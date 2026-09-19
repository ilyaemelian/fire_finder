"""Финальный инференс BS: два прохода XGBoost, U-Net и смесь вероятностей."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import segmentation_models_pytorch as smp

from models.burn_severity.core import (
    SIZE,
    build_cache,
    context,
    predict_probability,
    rle_encode,
    unet_tensor,
)

SEED = 42
HERE = Path(__file__).resolve().parent


def select_device(requested: str) -> str:
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Запрошена CUDA, но torch.cuda.is_available() == False")
    return device


def main() -> None:
    parser = argparse.ArgumentParser(description="BS: XGBoost + U-Net → submission.csv")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=HERE / "models")
    parser.add_argument("--work", type=Path, default=None,
                        help="промежуточный файл признаков на диске; по умолчанию расчёт идёт в памяти")
    parser.add_argument("--unet-weight", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--sample", type=Path,
                        help="шаблон состава строк; по умолчанию "
                             "<data-dir>/sample_submission.csv")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    device = select_device(args.device)
    started = time.perf_counter()
    test_root = args.data_dir / "bs"
    template_path = args.sample or args.data_dir / "sample_submission.csv"
    if not template_path.is_file():
        raise SystemExit(
            f"Не найден шаблон состава строк: {template_path}\n"
            f"Укажите его параметром --sample."
        )
    template = pd.read_csv(template_path)
    chip_ids = sorted(template.loc[template.chip_id.str.startswith("BS_"), "chip_id"].unique())
    cache = build_cache(test_root, chip_ids, args.work, "test", progress=True)
    expected_features = json.loads((args.models / "features.json").read_text(encoding="utf-8"))
    if cache["names"] != expected_features:
        raise RuntimeError("Порядок признаков теста не совпадает с обучением")
    print(f"    {len(chip_ids)} чипов, {len(cache['sev']):,} пикселей-кандидатов")

    config = json.loads((args.models / "config.json").read_text(encoding="utf-8"))
    folds = config["n_folds"]
    print(f"    вычислитель {device}; бустинг {folds + 1} моделей по 800 деревьев, "
          f"затем свёрточная сеть на {len(chip_ids)} чипах", flush=True)
    pass1 = np.zeros(len(cache["sev"]), np.float32)
    for fold in range(folds):
        model = xgb.XGBClassifier()
        model.load_model(args.models / f"pass1_fold{fold}.ubj")
        model.set_params(device=device)
        pass1 += predict_probability(model, cache["X"]) / folds
        print(f"    бустинг, первый этап: фолд {fold + 1} из {folds}, "
              f"{time.perf_counter() - started:.0f} с", flush=True)
    second = xgb.XGBClassifier()
    second.load_model(args.models / "pass2.ubj")
    second.set_params(device=device)
    boost = predict_probability(second, cache["X"], context(pass1, cache))
    print(f"    бустинг готов, {time.perf_counter() - started:.0f} с")

    norm = json.loads((args.models / "unet_norm.json").read_text(encoding="utf-8"))
    mean = torch.tensor(norm["mean"]).view(1, -1, 1, 1).to(device)
    std = torch.tensor(norm["std"]).view(1, -1, 1, 1).to(device)
    network = smp.Unet(
        "resnet34", encoder_weights=None, in_channels=len(norm["mean"]), classes=1,
    ).to(device)
    network.load_state_dict(torch.load(args.models / "unet_full_ep25.pt", map_location=device))
    network.eval()
    unet_probability = np.zeros(len(cache["sev"]), np.float32)
    with torch.no_grad():
        for index, chip_id in enumerate(chip_ids):
            start, stop = cache["offsets"][index:index + 2]
            tensor = torch.from_numpy(unet_tensor(test_root, chip_id).astype(np.float32))[None].to(device)
            tensor = (tensor - mean) / std
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                probability = (
                    torch.sigmoid(network(tensor).float())
                    + torch.sigmoid(network(tensor.flip(-1)).float()).flip(-1)
                    + torch.sigmoid(network(tensor.flip(-2)).float()).flip(-2)
                ) / 3
            unet_probability[start:stop] = (
                probability[0, 0].cpu().numpy().ravel()[cache["idx"][start:stop]]
            )
            if (index + 1) % 20 == 0 or index + 1 == len(chip_ids):
                print(f"    свёрточная сеть: {index + 1} из {len(chip_ids)}, "
                      f"{time.perf_counter() - started:.0f} с", flush=True)

    blend = args.unet_weight * unet_probability + (1 - args.unet_weight) * boost
    records = []
    area = np.zeros(4, np.int64)
    for index, chip_id in enumerate(chip_ids):
        start, stop = cache["offsets"][index:index + 2]
        prediction = np.zeros(SIZE * SIZE, np.uint8)
        prediction[cache["idx"][start:stop]] = np.where(
            blend[start:stop] >= args.threshold, cache["sev"][start:stop], 0,
        )
        area += np.bincount(prediction, minlength=4)
        for class_id in (1, 2, 3):
            records.append({
                "chip_id": chip_id,
                "class_id": class_id,
                "rle": rle_encode(prediction == class_id),
            })
    submission = pd.DataFrame(records, columns=["chip_id", "class_id", "rle"])
    expected = template[template.chip_id.str.startswith("BS_")]
    if set(zip(submission.chip_id, submission.class_id)) != set(zip(expected.chip_id, expected.class_id)):
        raise RuntimeError("Набор строк BS не совпадает с sample_submission.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    burnt = int(area[1:].sum())
    print(
        f"{args.output}: {len(submission)} строк · гарь {100 * burnt / area.sum():.2f}% · "
        f"классы 1/2/3: {area[1]} / {area[2]} / {area[3]} · "
        f"{time.perf_counter() - started:.1f} с"
    )


if __name__ == "__main__":
    main()

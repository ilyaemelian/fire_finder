"""Полное воспроизводимое обучение финальной схемы BS."""

import argparse
import hashlib
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
import torch
import torch.nn.functional as functional
import xgboost as xgb
import segmentation_models_pytorch as smp

from models.burn_severity.core import (
    SIZE,
    build_cache,
    context,
    data_folder,
    predict_probability,
    unet_tensor,
)

SEED = 42
UNET_CHANNELS = 40
PROTECTED = Path(__file__).resolve().parent / "models"


def guard_output(output: Path, force: bool) -> Path:
    """Запретить запись поверх исходных весов, переданных вместе с решением."""
    output = output.resolve()
    if output == PROTECTED and not force:
        raise SystemExit(
            f"Каталог {PROTECTED} содержит исходные веса модуля BS и не перезаписывается.\n"
            f"Укажите другой --output-dir, например "
            f"outputs/training/burn_severity/models, либо добавьте --force "
            f"для сознательной замены."
        )
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def chip_ids(root: Path, maximum: int | None = None) -> list[str]:
    result = sorted(path.name.removesuffix("_MASK.tif") for path in (root / "masks").glob("*_MASK.tif"))
    if maximum:
        result = result[:maximum]
    if not result:
        raise ValueError(f"В {root / 'masks'} нет файлов *_MASK.tif")
    return result


def geographic_block(root: Path, chip_id: str) -> tuple[int, int]:
    path = root / "sentinel2_pre" / f"{chip_id}_Sentinel-2_pre.tif"
    with rasterio.open(path) as source:
        center_x, center_y = source.transform * (source.width / 2, source.height / 2)
        longitude, latitude = transform(source.crs, "EPSG:4326", [center_x], [center_y])
    return int(np.floor(longitude[0])), int(np.floor(latitude[0]))


def folds_by_block(root: Path, ids: list[str], folds: int, seed: int) -> np.ndarray:
    groups = [geographic_block(root, chip_id) for chip_id in ids]
    unique = sorted(set(groups))
    if len(unique) < folds:
        raise ValueError(f"Для {folds} территориальных фолдов найдено только {len(unique)} блоков 1°×1°")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    assignment = {group: index % folds for index, group in enumerate(unique)}
    return np.asarray([assignment[group] for group in groups], dtype=np.int16)


def rows_for_chips(offsets: np.ndarray, chips) -> np.ndarray:
    parts = [np.arange(offsets[chip], offsets[chip + 1]) for chip in chips]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def sample_rows(offsets: np.ndarray, chips, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    parts = []
    for chip in chips:
        start, stop = offsets[chip:chip + 2]
        count = stop - start
        local = np.arange(count) if count <= maximum else rng.choice(count, maximum, replace=False)
        parts.append(start + local)
    return np.sort(np.concatenate(parts))


def classifier(seed: int, estimators: int, device: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=estimators,
        learning_rate=0.05,
        max_depth=9,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        device=device,
        random_state=seed,
        eval_metric="logloss",
    )


def train_boosting(root: Path, ids: list[str], models: Path, work: Path, *,
                   folds: int, estimators: int, samples_per_chip: int,
                   seed: int, device: str) -> dict:
    cache = build_cache(root, ids, work, "train", masks=root / "masks")
    labels = cache["gt"] > 0
    fold_of_chip = folds_by_block(root, ids, folds, seed)
    oof = np.zeros(len(labels), np.float32)
    checksums = {}
    for fold in range(folds):
        train_chips = np.flatnonzero(fold_of_chip != fold)
        valid_chips = np.flatnonzero(fold_of_chip == fold)
        train_rows = sample_rows(cache["offsets"], train_chips, samples_per_chip, seed + fold)
        valid_rows = rows_for_chips(cache["offsets"], valid_chips)
        model = classifier(seed + fold, estimators, device).fit(
            np.asarray(cache["X"][train_rows]), labels[train_rows],
        )
        oof[valid_rows] = predict_probability(model, cache["X"][valid_rows])
        path = models / f"pass1_fold{fold}.ubj"
        model.save_model(path)
        checksums[path.name] = sha256(path)
        print(f"BS XGBoost: первый проход {fold + 1}/{folds}", flush=True)

    contextual = context(oof, cache)
    all_rows = sample_rows(cache["offsets"], range(len(ids)), samples_per_chip, seed)
    second_matrix = np.hstack([
        np.asarray(cache["X"][all_rows]),
        contextual[all_rows],
    ])
    second = classifier(seed, estimators, device).fit(second_matrix, labels[all_rows])
    second_path = models / "pass2.ubj"
    second.save_model(second_path)
    (models / "features.json").write_text(
        json.dumps(cache["names"], ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("BS XGBoost: второй проход завершён", flush=True)
    return {
        "folds": folds,
        "estimators": estimators,
        "pass1_sha256": checksums,
        "pass2_sha256": sha256(second_path),
        "training_candidates": int(len(labels)),
        "training_rows": int(len(all_rows)),
        "territorial_blocks": int(len({geographic_block(root, chip_id) for chip_id in ids})),
    }


def prepare_unet_cache(root: Path, ids: list[str], work: Path):
    work.mkdir(parents=True, exist_ok=True)
    tensors = np.memmap(
        work / "unet_X.float16.bin", dtype=np.float16, mode="w+",
        shape=(len(ids), UNET_CHANNELS, SIZE, SIZE),
    )
    targets = np.memmap(
        work / "unet_y.uint8.bin", dtype=np.uint8, mode="w+",
        shape=(len(ids), SIZE, SIZE),
    )
    for index, chip_id in enumerate(ids):
        tensors[index] = unet_tensor(root, chip_id)
        with rasterio.open(root / "masks" / f"{chip_id}_MASK.tif") as source:
            mask = source.read(1)
        targets[index] = ((mask >= 1) & (mask <= 3)).astype(np.uint8)
        print(f"BS U-Net: тензор {index + 1}/{len(ids)}", flush=True)
    tensors.flush()
    targets.flush()
    return tensors, targets


def batch_stream(tensors, targets, rng, *, crop: int, crops_per_chip: int, batch: int,
                 pin_memory: bool):
    output = queue.Queue(maxsize=8)

    def produce():
        items = [
            (chip, rng.randint(0, SIZE - crop + 1), rng.randint(0, SIZE - crop + 1))
            for chip in range(len(tensors)) for _ in range(crops_per_chip)
        ]
        rng.shuffle(items)
        for start in range(0, len(items), batch):
            xs, ys = [], []
            for chip, row, column in items[start:start + batch]:
                x = np.asarray(tensors[chip, :, row:row + crop, column:column + crop], np.float32)
                y = np.asarray(targets[chip, row:row + crop, column:column + crop], np.float32)
                rotation = rng.randint(4)
                x, y = np.rot90(x, rotation, (1, 2)), np.rot90(y, rotation)
                if rng.rand() < 0.5:
                    x, y = x[:, :, ::-1], y[:, ::-1]
                xs.append(np.ascontiguousarray(x))
                ys.append(np.ascontiguousarray(y))
            x_tensor = torch.from_numpy(np.stack(xs))
            y_tensor = torch.from_numpy(np.stack(ys))[:, None]
            if pin_memory:
                x_tensor, y_tensor = x_tensor.pin_memory(), y_tensor.pin_memory()
            output.put((x_tensor, y_tensor))
        output.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (item := output.get()) is not None:
        yield item


def train_unet(root: Path, ids: list[str], models: Path, work: Path, *,
               epochs: int, epoch_schedule: int, crop: int, crops_per_chip: int,
               batch: int, seed: int, device: str, encoder_weights: str | None) -> dict:
    tensors, targets = prepare_unet_cache(root, ids, work)
    sample = np.asarray(tensors[:, :, ::8, ::8], np.float32)
    mean_values = sample.mean((0, 2, 3))
    std_values = sample.std((0, 2, 3)) + 1e-6
    del sample
    norm_path = models / "unet_norm.json"
    norm_path.write_text(
        json.dumps({"mean": mean_values.tolist(), "std": std_values.tolist()}), encoding="utf-8",
    )
    mean = torch.tensor(mean_values).view(1, -1, 1, 1).to(device)
    std = torch.tensor(std_values).view(1, -1, 1, 1).to(device)
    network = smp.Unet(
        "resnet34", encoder_weights=encoder_weights,
        in_channels=UNET_CHANNELS, classes=1,
    ).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = (len(ids) * crops_per_chip + batch - 1) // batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=epoch_schedule * steps, pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.RandomState(seed)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        network.train()
        total_loss = seen = 0
        for x, y in batch_stream(
            tensors, targets, rng, crop=crop, crops_per_chip=crops_per_chip,
            batch=batch, pin_memory=device == "cuda",
        ):
            x = (x.to(device, non_blocking=True) - mean) / std
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits = network(x).float()
                probability = torch.sigmoid(logits)
                dice = 1 - (2 * (probability * y).sum() + 1) / (probability.sum() + y.sum() + 1)
                loss = functional.binary_cross_entropy_with_logits(logits, y) + dice
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(x)
            seen += len(x)
        value = total_loss / seen
        history.append({"epoch": epoch, "loss": value, "lr": scheduler.get_last_lr()[0]})
        print(f"BS U-Net: эпоха {epoch}/{epochs}, loss={value:.5f}", flush=True)
    weights = models / f"unet_full_ep{epochs}.pt"
    torch.save(network.state_dict(), weights)
    if weights.name != "unet_full_ep25.pt":
        target = models / "unet_full_ep25.pt"
        target.write_bytes(weights.read_bytes())
        weights = target
    return {
        "weights": weights.name,
        "normalisation": norm_path.name,
        "sha256": sha256(weights),
        "norm_sha256": sha256(norm_path),
        "epochs": epochs,
        "epoch_schedule": epoch_schedule,
        "batch": batch,
        "crop": crop,
        "crops_per_chip": crops_per_chip,
        "final_loss": history[-1]["loss"],
        "history": history,
        "seconds": time.perf_counter() - started,
        "encoder_weights": encoder_weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить полный BS: XGBoost + U-Net")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--estimators", type=int, default=800)
    parser.add_argument("--samples-per-chip", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--epoch-schedule", type=int, default=60)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--crops-per-chip", type=int, default=4)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--encoder-weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--max-chips", type=int)
    parser.add_argument("--force", action="store_true",
                        help="разрешить запись поверх исходных весов модуля")
    args = parser.parse_args()

    models = guard_output(args.output_dir, args.force)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Запрошена CUDA, но CUDA недоступна")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    root = args.data_dir / "bs" if (args.data_dir / "bs").is_dir() else args.data_dir
    ids = chip_ids(root, args.max_chips)
    work = args.work or args.output_dir.parent / "work"
    models.mkdir(parents=True, exist_ok=True)
    print(f"BS: {len(ids)} обучающих чипов, device={device}, seed={args.seed}")

    boosting = train_boosting(
        root, ids, models, work / "boosting", folds=args.folds,
        estimators=args.estimators, samples_per_chip=args.samples_per_chip,
        seed=args.seed, device=device,
    )
    unet = train_unet(
        root, ids, models, work / "unet", epochs=args.epochs,
        epoch_schedule=args.epoch_schedule, crop=args.crop,
        crops_per_chip=args.crops_per_chip, batch=args.batch,
        seed=args.seed, device=device,
        encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
    )
    config = {
        "n_folds": args.folds,
        "pass1_weights": "pass1_fold{fold}.ubj",
        "pass2_weights": "pass2.ubj",
        "unet_weights": "unet_full_ep25.pt",
        "unet_norm": "unet_norm.json",
        "unet_weight": 0.45,
        "threshold": 0.4,
        "seed": args.seed,
    }
    (models / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    manifest = {
        "module": "burn_severity",
        "algorithm": "5 XGBoost + XGBoost с контекстом + U-Net(resnet34)",
        "seed": args.seed,
        "training_chips": len(ids),
        "validation": f"{args.folds} территориальных фолдов по блокам 1°×1°",
        "pass1": {
            "weights": [f"pass1_fold{fold}.ubj" for fold in range(args.folds)],
            "sha256": boosting["pass1_sha256"],
        },
        "pass2": {"weights": "pass2.ubj", "sha256": boosting["pass2_sha256"]},
        "unet": unet,
        "blend": {"unet_weight": 0.45, "threshold": 0.4},
        "boosting": boosting,
    }
    (models / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("\nBS обучение завершено")
    print(f"  XGBoost: {args.folds} + 1 моделей, {args.estimators} деревьев")
    print(f"  U-Net: {args.epochs} эпох из расписания {args.epoch_schedule}, loss={unet['final_loss']:.5f}")
    print(f"  Артефакты: {models}")


if __name__ == "__main__":
    main()

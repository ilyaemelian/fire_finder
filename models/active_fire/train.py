"""Воспроизводимое обучение модели активного горения из командной строки."""

import argparse
import hashlib
import json
import os
import pickle
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio

from models.active_fire.features import pixel_features, read_channels

THRESHOLDS = (0.90, 0.99, 0.995, 0.999, 0.9995, 0.9999)
PROTECTED = Path(__file__).resolve().parent / "artifacts"


def guard_output(artifacts: Path, force: bool) -> Path:
    """Запретить запись поверх исходных весов, переданных вместе с решением."""
    artifacts = artifacts.resolve()
    if artifacts == PROTECTED and not force:
        raise SystemExit(
            f"Каталог {PROTECTED} содержит исходные веса модуля AF и не перезаписывается.\n"
            f"Укажите другой --output-dir, например outputs/training/active_fire, "
            f"либо добавьте --force для сознательной замены."
        )
    return artifacts


def mask_path(root: Path, chip_id: str) -> Path:
    candidates = [root / "masks" / f"{chip_id}_MASK.tif", root / chip_id / f"{chip_id}_MASK.tif"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Маска AF не найдена для {chip_id}")


def chip_ids(root: Path) -> list[str]:
    ids = sorted(path.name.removesuffix("_MASK.tif") for path in (root / "masks").glob("*_MASK.tif"))
    if not ids:
        raise ValueError(f"В {root / 'masks'} нет файлов *_MASK.tif")
    return ids


def read_target(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        mask = source.read(1)
    return (mask == 1).ravel()


def classifier(seed: int, estimators: int):
    return lgb.LGBMClassifier(
        n_estimators=estimators, learning_rate=.08, num_leaves=63, min_child_samples=40,
        reg_lambda=1.0, random_state=seed, deterministic=True, force_col_wise=True,
        verbose=-1, n_jobs=-1,
    )


def score_thresholds(truth: np.ndarray, probability: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for threshold in THRESHOLDS:
        predicted = probability >= threshold
        tp = int(np.sum(predicted & truth))
        fp = int(np.sum(predicted & ~truth))
        fn = int(np.sum(~predicted & truth))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1})
    best = max(rows, key=lambda row: row["f1"])
    return float(best["threshold"]), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить модель AF")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training/active_fire"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--negatives-per-chip", type=int, default=1500)
    parser.add_argument("--estimators", type=int, default=500)
    parser.add_argument("--max-chips", type=int)
    parser.add_argument("--force", action="store_true",
                        help="разрешить запись поверх исходных весов модуля")
    args = parser.parse_args()

    artifacts = guard_output(args.output_dir / "artifacts", args.force)
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    root = args.data_dir / "af" if (args.data_dir / "af").is_dir() else args.data_dir
    ids = chip_ids(root)
    if args.max_chips:
        ids = ids[:args.max_chips]
    if len(ids) < args.folds:
        raise SystemExit(f"Для {args.folds} фолдов требуется не менее {args.folds} чипов")

    rng = np.random.RandomState(args.seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    fold_of = {chip_id: index % args.folds for index, chip_id in enumerate(shuffled)}
    samples = {}
    feature_names = None
    print(f"AF: подготовка выборки из {len(ids)} чипов")
    for index, chip_id in enumerate(ids, 1):
        matrix, names = pixel_features(read_channels(root / chip_id))
        feature_names = feature_names or names
        if names != feature_names:
            raise ValueError(f"{chip_id}: изменился порядок признаков")
        target = read_target(mask_path(root, chip_id))
        positive = np.flatnonzero(target)
        negative = np.flatnonzero(~target)
        take = rng.choice(negative, min(args.negatives_per_chip, negative.size), replace=False)
        selected = np.concatenate([positive, take])
        samples[chip_id] = (matrix[selected], target[selected])
        print(f"  {index}/{len(ids)} {chip_id}: {len(positive)} положительных", flush=True)

    truth_parts, probability_parts = [], []
    for fold in range(args.folds):
        train_ids = [chip for chip in ids if fold_of[chip] != fold]
        valid_ids = [chip for chip in ids if fold_of[chip] == fold]
        x = np.concatenate([samples[chip][0] for chip in train_ids])
        y = np.concatenate([samples[chip][1] for chip in train_ids])
        model = classifier(args.seed + fold, args.estimators).fit(x, y)
        for chip_id in valid_ids:
            matrix, _ = pixel_features(read_channels(root / chip_id))
            truth_parts.append(read_target(mask_path(root, chip_id)))
            probability_parts.append(model.predict_proba(matrix)[:, 1].astype(np.float32))
        print(f"AF: фолд {fold + 1}/{args.folds} завершён", flush=True)

    threshold, metrics = score_thresholds(np.concatenate(truth_parts), np.concatenate(probability_parts))
    x = np.concatenate([samples[chip][0] for chip in ids])
    y = np.concatenate([samples[chip][1] for chip in ids])
    model = classifier(args.seed, args.estimators).fit(x, y)

    artifacts.mkdir(parents=True, exist_ok=True)
    weight = artifacts / "model.pkl"
    with weight.open("wb") as stream:
        pickle.dump({"model": model, "threshold": threshold, "features": feature_names,
                     "windows": (5, 11, 21), "seed": args.seed}, stream)
    digest = hashlib.sha256(weight.read_bytes()).hexdigest().upper()
    best = max(metrics, key=lambda row: row["f1"])
    manifest = {
        "module": "active_fire", "algorithm": "LightGBM, попиксельная бинарная классификация",
        "weights": weight.name, "weights_sha256": digest, "features": len(feature_names),
        "probability_threshold": threshold, "seed": args.seed, "training_chips": len(ids),
        "validation": f"{args.folds} детерминированных фолдов по chip_id",
        "f1_cross_validation": best["f1"], "precision_cross_validation": best["precision"],
        "recall_cross_validation": best["recall"],
        "n_estimators": args.estimators,
        "note": "Обучение CLI не использует NASA FIRMS; переданный командой инференса артефакт не изменён.",
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"AF: веса {weight}; threshold={threshold}; F1={best['f1']:.4f}")


if __name__ == "__main__":
    main()

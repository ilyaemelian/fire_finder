"""Обучение бейзлайна BS: CatBoost MultiClass по признакам dNBR / NBR_pre / NBR_post.

Повторяет notebooks/baseline_catboost_bs.ipynb, но без ручной правки путей:
каталог чипов и список идентификаторов задаются аргументами.

    python -m models.burn_severity.train --data-dir C:/data/train --limit 30
"""

import argparse
import time
from pathlib import Path

import numpy as np

from models.burn_severity import WEIGHTS
from models.burn_severity.predict import (
    BS_CLASSES,
    BS_SHAPE,
    extract_features,
    read_optical,
    resolve_chip_paths,
)

DEFAULT_ITERATIONS = 150
DEFAULT_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.1


def discover_chips(data_dir: Path) -> list[Path]:
    """Найти чипы как <data-dir>/**/<chip_id>_pre.tif и вернуть их префиксы."""
    chips = sorted(path.with_name(path.name[: -len("_pre.tif")]) for path in data_dir.rglob("*_pre.tif"))
    if not chips:
        raise FileNotFoundError(f"No *_pre.tif found under {data_dir}")
    return chips


def read_mask(chip_prefix: Path) -> np.ndarray:
    import rasterio

    candidates = [Path(f"{chip_prefix}_mask.tif"), Path(f"{chip_prefix}_target.tif")]
    for path in candidates:
        if path.is_file():
            with rasterio.open(path) as source:
                mask = source.read(1).astype(np.uint8)
            if mask.shape != BS_SHAPE:
                raise ValueError(f"{path}: expected {BS_SHAPE}, got {mask.shape}")
            if not np.isin(mask, BS_CLASSES).all():
                raise ValueError(f"{path}: labels outside {BS_CLASSES}")
            return mask
    raise FileNotFoundError(f"Mask not found for {chip_prefix} (tried: {', '.join(map(str, candidates))})")


def build_dataset(chips: list[Path], pixels_per_chip: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for index, chip in enumerate(chips, 1):
        pre_path, post_path = resolve_chip_paths(chip)
        chip_features = extract_features(read_optical(pre_path), read_optical(post_path))
        chip_labels = read_mask(chip).reshape(-1)
        if pixels_per_chip and pixels_per_chip < chip_labels.size:
            picked = rng.choice(chip_labels.size, size=pixels_per_chip, replace=False)
            chip_features, chip_labels = chip_features[picked], chip_labels[picked]
        features.append(chip_features)
        labels.append(chip_labels)
        print(f"dataset: {index}/{len(chips)} chips", flush=True)
    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BS CatBoost baseline")
    parser.add_argument("--data-dir", type=Path, required=True, help="Каталог с train-чипами BS")
    parser.add_argument("--limit", type=int, default=30, help="Сколько чипов взять (0 — все)")
    parser.add_argument("--pixels-per-chip", type=int, default=0,
                        help="Подвыборка пикселей на чип, 0 — все 262144")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=WEIGHTS)
    args = parser.parse_args()

    from catboost import CatBoostClassifier

    chips = discover_chips(args.data_dir)
    if args.limit:
        chips = chips[: args.limit]
    features, labels = build_dataset(chips, args.pixels_per_chip or None, args.seed)
    print(f"Train matrix: {features.shape[0]} rows x {features.shape[1]} features; "
          f"class counts: {np.bincount(labels, minlength=4).tolist()}")

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="MultiClass",
        task_type="CPU",
        random_seed=args.seed,
        verbose=25,
    )
    started = time.perf_counter()
    model.fit(features, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output))
    print(f"Saved {args.output} in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    main()

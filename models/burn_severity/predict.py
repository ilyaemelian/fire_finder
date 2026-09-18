"""Inference adapter for the BS CatBoost baseline (степень поражения гари).

Контракт повторяет модуль AF: один путь к чипу на вход, одна провалидированная
маска на выход. Никакой записи CSV здесь нет — сабмишн собирает run_inference.py.

    model = load_model(Path("models/burn_severity/artifacts/baseline_bs_model.cbm"))
    mask = predict_bs(Path(".../BS_0001"), model)   # uint8 (512, 512), классы 0/1/2/3
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BS_SHAPE = (512, 512)
BS_CLASSES = (0, 1, 2, 3)
# Индексы каналов Sentinel-2 L2A в порядке GeoTIFF кейса (см. документацию данных).
BAND_B8A = 6
BAND_B12 = 8
EPS = 1e-8
FEATURE_NAMES = ("dNBR", "NBR_pre", "NBR_post")
DEFAULT_MODEL_NAME = "baseline_bs_model.cbm"


@dataclass(frozen=True)
class BurnSeverityModel:
    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES


def resolve_chip_paths(chip_path: Path) -> tuple[Path, Path]:
    """Принять либо каталог чипа, либо префикс <dir>/<chip_id> и вернуть (pre, post)."""
    chip_path = Path(chip_path)
    if chip_path.is_dir():
        pre = sorted(chip_path.glob("*_pre.tif"))
        post = sorted(chip_path.glob("*_post.tif"))
        if len(pre) != 1 or len(post) != 1:
            raise ValueError(
                f"Expected exactly one *_pre.tif and one *_post.tif in {chip_path}, "
                f"found {len(pre)} and {len(post)}"
            )
        return pre[0], post[0]
    pre, post = Path(f"{chip_path}_pre.tif"), Path(f"{chip_path}_post.tif")
    missing = [path for path in (pre, post) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"BS chip files not found: {', '.join(map(str, missing))}")
    return pre, post


def read_optical(path: Path) -> np.ndarray:
    """Прочитать снимок Sentinel-2 как float32 (bands, 512, 512)."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install BS inference dependencies first (requirements-bs.txt)") from exc
    with rasterio.open(path) as source:
        image = source.read().astype(np.float32)
    if image.ndim != 3 or image.shape[1:] != BS_SHAPE:
        raise ValueError(f"{path}: expected bands x {BS_SHAPE}, got {image.shape}")
    if image.shape[0] <= BAND_B12:
        raise ValueError(f"{path}: expected at least {BAND_B12 + 1} bands, got {image.shape[0]}")
    return image


def nbr(image: np.ndarray) -> np.ndarray:
    b8a, b12 = image[BAND_B8A], image[BAND_B12]
    return (b8a - b12) / (b8a + b12 + EPS)


def extract_features(pre_image: np.ndarray, post_image: np.ndarray) -> np.ndarray:
    """(bands, 512, 512) x2 -> таблица (262144, 3) в порядке FEATURE_NAMES."""
    nbr_pre, nbr_post = nbr(pre_image), nbr(post_image)
    stacked = np.stack([nbr_pre - nbr_post, nbr_pre, nbr_post], axis=-1)
    return np.nan_to_num(stacked.reshape(-1, len(FEATURE_NAMES)), copy=False)


def validate_bs_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("BS prediction must be a numpy.ndarray")
    if mask.shape != BS_SHAPE:
        raise ValueError(f"BS prediction must have shape {BS_SHAPE}, got {mask.shape}")
    if mask.dtype != np.uint8:
        raise TypeError(f"BS prediction must be uint8, got {mask.dtype}")
    if not np.isin(mask, BS_CLASSES).all():
        raise ValueError(f"BS prediction may contain only {BS_CLASSES}")


def load_model(model_path: Path) -> BurnSeverityModel:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"BS weights not found: {path}")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is not installed. Install requirements-bs.txt in the ML environment."
        ) from exc
    model = CatBoostClassifier()
    model.load_model(str(path))
    return BurnSeverityModel(model=model)


def predict_bs(chip_path: Path, model: BurnSeverityModel) -> np.ndarray:
    """Вернуть маску степени поражения для одного BS-чипа."""
    pre_path, post_path = resolve_chip_paths(chip_path)
    features = extract_features(read_optical(pre_path), read_optical(post_path))
    predictions = np.asarray(model.model.predict(features)).astype(np.uint8).reshape(BS_SHAPE)
    validate_bs_mask(predictions)
    return predictions

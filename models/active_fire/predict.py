"""Применение модели AF к одному чипу.

Контракт модуля: на входе путь к чипу, на выходе проверенная бинарная матрица
256×256 типа uint8. Этим пользуются и расчёт таблицы предсказаний, и
веб-приложение, которому нужна матрица `.npy` для отображения на карте.

    from models.active_fire import WEIGHTS, load_model, predict_af

    model = load_model(WEIGHTS)
    mask = predict_af(path_to_chip, model)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import AF_SHAPE, pixel_features, read_channels


@dataclass(frozen=True)
class ActiveFireModel:
    """Обученная модель вместе с порогом и порядком признаков."""

    model: Any
    threshold: float
    features: list[str]
    windows: tuple[int, ...]
    seed: int | None = None


def load_model(model_path: Path) -> ActiveFireModel:
    import pickle

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Модель AF не найдена: {path}")
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    missing = {"model", "threshold", "features", "windows"} - bundle.keys()
    if missing:
        raise ValueError(f"В файле модели нет полей: {', '.join(sorted(missing))}")
    return ActiveFireModel(
        model=bundle["model"],
        threshold=float(bundle["threshold"]),
        features=list(bundle["features"]),
        windows=tuple(bundle["windows"]),
        seed=bundle.get("seed"),
    )


def validate_af_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("Предсказание AF должно быть numpy.ndarray")
    if mask.shape != AF_SHAPE:
        raise ValueError(f"Предсказание AF должно иметь размер {AF_SHAPE}, получено {mask.shape}")
    if mask.dtype != np.uint8:
        raise TypeError(f"Предсказание AF должно быть uint8, получено {mask.dtype}")
    if not np.isin(mask, (0, 1)).all():
        raise ValueError("Предсказание AF может содержать только значения 0 и 1")


def predict_proba(chip_path: Path, model: ActiveFireModel) -> np.ndarray:
    """Вероятность горения для каждого пикселя, матрица 256×256 типа float32."""
    matrix, names = pixel_features(read_channels(chip_path), model.windows)
    if names != model.features:
        raise ValueError(
            "Порядок признаков не совпадает с сохранённым при обучении: "
            f"ожидалось {len(model.features)}, получено {len(names)}"
        )
    probability = model.model.predict_proba(matrix)[:, 1]
    return probability.reshape(AF_SHAPE).astype(np.float32)


def predict_af(chip_path: Path, model: ActiveFireModel,
               threshold: float | None = None) -> np.ndarray:
    """Бинарная матрица активного горения для одного чипа."""
    level = model.threshold if threshold is None else float(threshold)
    mask = (predict_proba(chip_path, model) >= level).astype(np.uint8)
    validate_af_mask(mask)
    return mask

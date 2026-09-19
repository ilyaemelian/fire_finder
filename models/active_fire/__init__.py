"""Модуль AF — детекция активного горения по снимкам VIIRS.

Попиксельный классификатор LightGBM на 38 признаках. Сущность самодостаточна:
код расчёта признаков, адаптер применения модели, командный расчёт таблицы
предсказаний, весовые коэффициенты, тетрадь обучения, документация и готовый
результат лежат внутри этого пакета.
"""

from pathlib import Path

from .features import AF_SHAPE, CHANNELS, WINDOWS, pixel_features, read_channels
from .predict import ActiveFireModel, load_model, predict_af, predict_proba, validate_af_mask

MODULE_ROOT = Path(__file__).resolve().parent
WEIGHTS = MODULE_ROOT / "artifacts" / "model.pkl"
RESULTS = MODULE_ROOT / "results" / "af_submission.csv"

__all__ = [
    "AF_SHAPE",
    "CHANNELS",
    "MODULE_ROOT",
    "RESULTS",
    "WEIGHTS",
    "WINDOWS",
    "ActiveFireModel",
    "load_model",
    "pixel_features",
    "predict_af",
    "predict_proba",
    "read_channels",
    "validate_af_mask",
]

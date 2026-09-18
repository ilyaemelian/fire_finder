"""Модель AF (active fire) — детекция активного горения по VIIRS, YOLO11n.

Сущность самодостаточна: код адаптера, веса, журнал обучения, notebook,
описание и готовый results/active_fire_yolo.csv лежат внутри этого пакета.
"""

from pathlib import Path

from .predict import ActiveFireModel, load_model, predict_af, validate_af_mask

MODULE_ROOT = Path(__file__).resolve().parent
WEIGHTS = MODULE_ROOT / "artifacts" / "best.pt"
RESULTS = MODULE_ROOT / "results" / "active_fire_yolo.csv"

__all__ = [
    "ActiveFireModel",
    "MODULE_ROOT",
    "RESULTS",
    "WEIGHTS",
    "load_model",
    "predict_af",
    "validate_af_mask",
]

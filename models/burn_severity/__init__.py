"""Модель BS (burn severity) — степень поражения гари по Sentinel-2, CatBoost на dNBR.

Сущность самодостаточна: адаптер инференса, обучение, notebook-бейзлайн,
веса в artifacts/ и результат в results/ лежат внутри этого пакета.
"""

from pathlib import Path

from .predict import (
    BS_CLASSES,
    BS_SHAPE,
    BurnSeverityModel,
    extract_features,
    load_model,
    predict_bs,
    validate_bs_mask,
)

MODULE_ROOT = Path(__file__).resolve().parent
WEIGHTS = MODULE_ROOT / "artifacts" / "baseline_bs_model.cbm"
RESULTS = MODULE_ROOT / "results" / "burn_severity_catboost.csv"

__all__ = [
    "BS_CLASSES",
    "BS_SHAPE",
    "BurnSeverityModel",
    "MODULE_ROOT",
    "RESULTS",
    "WEIGHTS",
    "extract_features",
    "load_model",
    "predict_bs",
    "validate_bs_mask",
]

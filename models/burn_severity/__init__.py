"""Финальный модуль BS, переданный автором модели и встроенный без смены алгоритма."""

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
MODELS = MODULE_ROOT / "models"
RESULTS = MODULE_ROOT / "results" / "bs_submission.csv"

__all__ = ["MODELS", "MODULE_ROOT", "RESULTS"]

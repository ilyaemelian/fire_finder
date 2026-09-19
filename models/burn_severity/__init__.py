"""Модуль BS — степень поражения гари по снимкам Sentinel-2.

Двухэтапная схема: бинарная модель контура гари, затем уточнение; степень
внутри контура определяется пороговым правилом по индексу dNBR с учётом типа
покрова. Переданы весовые коэффициенты XGBoost и результаты изучения данных;
код расчёта предсказаний ожидается — см. README.md, раздел о недостающих файлах.
"""

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent
ARTIFACTS = MODULE_ROOT / "artifacts"
RESULTS = MODULE_ROOT / "results" / "bs_submission.csv"

BS_SHAPE = (512, 512)
BS_CLASSES = (0, 1, 2, 3)

# Пороги dNBR из постановки задачи: код WorldCover -> (слабое, среднее, сильное).
SEVERITY_THRESHOLDS = {
    10: (0.100, 0.270, 0.660),   # лес
    20: (0.100, 0.270, 0.660),   # кустарник
    30: (0.062, 0.204, 0.386),   # травы
    40: (0.070, 0.177, 0.380),   # пашня
    90: (0.075, 0.331, 0.677),   # болото, пойма
}

PASS1_WEIGHTS = tuple(sorted(ARTIFACTS.glob("pass1_fold*.ubj")))
PASS2_WEIGHTS = ARTIFACTS / "pass2.ubj"

__all__ = [
    "ARTIFACTS",
    "BS_CLASSES",
    "BS_SHAPE",
    "MODULE_ROOT",
    "PASS1_WEIGHTS",
    "PASS2_WEIGHTS",
    "RESULTS",
    "SEVERITY_THRESHOLDS",
]

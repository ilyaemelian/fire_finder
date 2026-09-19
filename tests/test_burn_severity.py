"""Проверки модуля BS: состав поставки, сохранность весов и параметров смеси.

Документы автора модуля (`IDEAS.md`, `PIPELINE.md`) и отдельный список
зависимостей объединены в `models/burn_severity/README.md` и корневой
`requirements-bs.txt`. Проверки следят за тем, чтобы при этом объединении не
потерялось содержание: обоснование свёрточной сети, параметры смеси, описание
расчёта и состав зависимостей.
"""

import hashlib
import json
import re
from pathlib import Path


MODULE = Path("models/burn_severity")
MODELS = MODULE / "models"
REQUIREMENTS = Path("requirements-bs.txt")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_author_delivery_structure_is_preserved():
    """Все переданные автором части модуля на месте."""
    required = {
        "README.md",
        "unet_train.ipynb",
        "research_bs.ipynb",
        "inference_new.py",
        "train.py",
        "core.py",
        "models",
    }
    assert required <= {path.name for path in MODULE.iterdir()}


def test_module_readme_keeps_the_rationale_for_the_neural_network():
    """Обоснование свёрточной сети из IDEAS.md перенесено в README модуля."""
    readme = (MODULE / "README.md").read_text(encoding="utf-8")
    assert "U-Net" in readme
    assert "ResNet34" in readme
    assert "Ronneberger" in readme                    # первоисточник архитектуры
    assert "40" in readme                             # состав входных каналов
    assert "0.445" in readme and "0.750" in readme    # измеренные IoU по классам
    assert "0.018" in readme                          # измеренный прирост смеси


def test_module_readme_describes_the_inference_pipeline():
    """Описание расчёта из PIPELINE.md перенесено в README модуля."""
    readme = (MODULE / "README.md").read_text(encoding="utf-8")
    assert "inference_new.py" in readme
    assert "models.burn_severity.inference_new" in readme
    for argument in ("--data-dir", "--output", "--models", "--unet-weight",
                     "--threshold", "--device"):
        assert argument in readme
    assert "dNBR" in readme
    assert "pass1" in readme and "pass2.ubj" in readme


def test_bs_requirements_cover_xgboost_runtime():
    """Зависимости модуля закреплены по версиям в корневом файле требований."""
    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    for package in ("xgboost==3.4.1", "scikit-learn==1.9.0", "torch==2.11.0",
                    "segmentation-models-pytorch==0.5.0", "rasterio==1.5.1"):
        assert package in requirements


def test_notebooks_have_no_author_machine_paths():
    for name in ("unet_train.ipynb", "research_bs.ipynb"):
        source = (MODULE / name).read_text(encoding="utf-8")
        assert "/home/admin/" not in source
        assert 'Path("../..")' not in source
        assert "data" in source and "train" in source


def test_all_author_weights_match_manifest():
    manifest = json.loads((MODELS / "manifest.json").read_text(encoding="utf-8"))
    expected = dict(manifest["pass1"]["sha256"])
    expected[manifest["pass2"]["weights"]] = manifest["pass2"]["sha256"]
    expected[manifest["unet"]["weights"]] = manifest["unet"]["sha256"]
    expected[manifest["unet"]["normalisation"]] = manifest["unet"]["norm_sha256"]
    assert {name: digest(MODELS / name) for name in expected} == {
        name: value.upper() for name, value in expected.items()
    }


def test_integration_keeps_author_blend_parameters():
    """Параметры смеси и её формула не изменены при встраивании модуля."""
    source = (MODULE / "inference_new.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--unet-weight", type=float, default=0.45)' in source
    assert 'parser.add_argument("--threshold", type=float, default=0.4)' in source
    # Смесь: p = w · p_сеть + (1 − w) · p_бустинг. Имена переменных менялись
    # при встраивании, поэтому сверяется сама форма выражения.
    blend = re.search(
        r"=\s*args\.unet_weight\s*\*\s*\w+\s*\+\s*\(\s*1\s*-\s*args\.unet_weight\s*\)\s*\*\s*\w+",
        source,
    )
    assert blend, "выражение смеси вероятностей не найдено в inference_new.py"
    assert "args.threshold" in source


def test_blend_parameters_match_the_configuration_of_the_weights():
    """Значения по умолчанию совпадают с конфигурацией переданных весов."""
    config = json.loads((MODELS / "config.json").read_text(encoding="utf-8"))
    assert config["unet_weight"] == 0.45
    assert config["threshold"] == 0.4

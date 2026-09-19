import hashlib
import json
from pathlib import Path


MODULE = Path("models/burn_severity")
MODELS = MODULE / "models"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def test_author_delivery_structure_is_preserved():
    required = {
        "IDEAS.md",
        "PIPELINE.md",
        "requirements.txt",
        "unet_train.ipynb",
        "research_bs.ipynb",
        "inference_new.py",
        "train.py",
        "core.py",
        "models",
    }
    assert required <= {path.name for path in MODULE.iterdir()}


def test_author_documents_and_requirements_are_preserved():
    expected = {
        "IDEAS.md": "F94DB60DF2D220D4A6D8ED9699D1AAF7A47121735738256682BF7F89E18FFB0D",
    }
    assert {name: digest(MODULE / name) for name in expected} == expected


def test_bs_requirements_cover_xgboost_runtime():
    requirements = (MODULE / "requirements.txt").read_text(encoding="utf-8")
    assert "xgboost==3.4.1" in requirements
    assert "scikit-learn==1.9.0" in requirements


def test_pipeline_names_the_delivered_inference_file():
    pipeline = (MODULE / "PIPELINE.md").read_text(encoding="utf-8")
    assert "python inference_new.py" in pipeline
    assert "`inference_new.py`" in pipeline


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
    source = (MODULE / "inference_new.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--unet-weight", type=float, default=0.45)' in source
    assert 'parser.add_argument("--threshold", type=float, default=0.4)' in source
    assert "p2_test = args.unet_weight * p_unet + (1 - args.unet_weight) * p2_boost" in source

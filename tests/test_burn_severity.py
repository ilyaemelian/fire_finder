"""Контракт модуля BS и сборка финального сабмишна из results/*.csv."""

import numpy as np
import pytest

from models.burn_severity.predict import (
    BAND_B12,
    BAND_B8A,
    BS_SHAPE,
    extract_features,
    resolve_chip_paths,
    validate_bs_mask,
)
from models.common.submission import kinds_present, read_rows, write_rows
from scripts.collect_results import collect


def fake_optical(b8a: float, b12: float) -> np.ndarray:
    image = np.zeros((10, *BS_SHAPE), dtype=np.float32)
    image[BAND_B8A] = b8a
    image[BAND_B12] = b12
    return image


def test_features_are_dnbr_nbr_pre_nbr_post():
    pre = fake_optical(b8a=3.0, b12=1.0)      # NBR_pre  = 0.5
    post = fake_optical(b8a=1.0, b12=3.0)     # NBR_post = -0.5
    features = extract_features(pre, post)
    assert features.shape == (BS_SHAPE[0] * BS_SHAPE[1], 3)
    assert features[0] == pytest.approx([1.0, 0.5, -0.5], abs=1e-6)


def test_features_survive_zero_reflectance():
    features = extract_features(fake_optical(0.0, 0.0), fake_optical(0.0, 0.0))
    assert np.isfinite(features).all()


def test_mask_contract_rejects_wrong_shape_dtype_and_classes():
    validate_bs_mask(np.zeros(BS_SHAPE, dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_bs_mask(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(TypeError):
        validate_bs_mask(np.zeros(BS_SHAPE, dtype=np.int16))
    with pytest.raises(ValueError):
        validate_bs_mask(np.full(BS_SHAPE, 4, dtype=np.uint8))


def test_chip_paths_resolve_from_prefix_and_directory(tmp_path):
    (tmp_path / "BS_1_pre.tif").write_bytes(b"")
    (tmp_path / "BS_1_post.tif").write_bytes(b"")
    assert resolve_chip_paths(tmp_path / "BS_1") == (tmp_path / "BS_1_pre.tif", tmp_path / "BS_1_post.tif")
    assert resolve_chip_paths(tmp_path) == (tmp_path / "BS_1_pre.tif", tmp_path / "BS_1_post.tif")
    with pytest.raises(FileNotFoundError):
        resolve_chip_paths(tmp_path / "BS_2")


def test_kinds_are_taken_from_chip_ids():
    assert kinds_present([{"chip_id": "AF_1", "class_id": 1}]) == {"af"}
    assert kinds_present([{"chip_id": "AF_1", "class_id": 1},
                          {"chip_id": "BS_1", "class_id": 1}]) == {"af", "bs"}
    with pytest.raises(ValueError):
        kinds_present([{"chip_id": "XX_1", "class_id": 1}])


def prepare_models_root(tmp_path, *, with_bs: bool = True):
    root = tmp_path / "models"
    write_rows(root / "active_fire" / "results" / "active_fire_yolo.csv",
               [{"chip_id": "AF_1", "class_id": 1, "rle": "1 1"}])
    if with_bs:
        write_rows(root / "burn_severity" / "results" / "burn_severity_catboost.csv",
                   [{"chip_id": "BS_1", "class_id": 1, "rle": "2 1"},
                    {"chip_id": "BS_1", "class_id": 2, "rle": "4 1"},
                    {"chip_id": "BS_1", "class_id": 3, "rle": ""}])
    sample = tmp_path / "sample.csv"
    write_rows(sample, [{"chip_id": "AF_1", "class_id": 1, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 1, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 2, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 3, "rle": ""}])
    return root, sample


def test_collect_merges_module_csv_in_template_order(tmp_path):
    root, sample = prepare_models_root(tmp_path)
    output = tmp_path / "submission.csv"
    stats = collect(sample, root, output)
    assert stats == {"rows": 4, "nonempty": 3, "pixels": 3}
    rows = read_rows(output)
    assert [(row["chip_id"], row["class_id"]) for row in rows] == [
        ("AF_1", 1), ("BS_1", 1), ("BS_1", 2), ("BS_1", 3)
    ]


def test_collect_refuses_partial_answer_unless_explicitly_allowed(tmp_path):
    root, sample = prepare_models_root(tmp_path, with_bs=False)
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="Нет предсказаний"):
        collect(sample, root, output)
    stats = collect(sample, root, output, fill_from_sample=True)
    assert stats == {"rows": 4, "nonempty": 1, "pixels": 1}

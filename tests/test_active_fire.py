"""Проверки модуля AF: расчёт признаков, контракт применения модели, результат."""

import os
from pathlib import Path

import numpy as np
import pytest

from models.active_fire.features import AF_SHAPE, CHANNELS, WINDOWS, pixel_features, resolve_chip_paths
from models.active_fire.predict import validate_af_mask
from models.common.submission import read_rows, validate_rows


def synthetic_channels(hot: bool = False) -> dict[str, np.ndarray]:
    channels = {name: np.full(AF_SHAPE, 280.0, np.float32) for name in CHANNELS}
    channels["I5"] = np.full(AF_SHAPE, 275.0, np.float32)
    channels["valid"] = np.ones(AF_SHAPE, np.float32)
    if hot:
        channels["I4"][128, 128] = 350.0
    return channels


def test_feature_count_and_order_are_fixed():
    matrix, names = pixel_features(synthetic_channels())
    assert matrix.shape == (AF_SHAPE[0] * AF_SHAPE[1], 38)
    assert matrix.dtype == np.float32
    assert names[: len(CHANNELS)] == list(CHANNELS)
    assert names[len(CHANNELS) : len(CHANNELS) + 3] == ["diff", "I4_over_I5", "I3_minus_I2"]
    for size in WINDOWS:
        assert f"I4_dev{size}" in names and f"diff_z{size}" in names
    assert names[-4:] == ["diff_peak_gap", "I4_peak_gap", "I4_min3", "I4_median5"]


def test_hot_pixel_stands_out_against_its_neighbourhood():
    names = pixel_features(synthetic_channels())[1]
    column = names.index("I4_z21")
    calm = pixel_features(synthetic_channels())[0][:, column]
    hot = pixel_features(synthetic_channels(hot=True))[0][:, column]
    centre = 128 * AF_SHAPE[1] + 128
    assert abs(calm[centre]) < 1.0
    assert hot[centre] > 10.0


def test_missing_channel_is_rejected():
    channels = synthetic_channels()
    del channels["landcover"]
    with pytest.raises(ValueError, match="landcover"):
        pixel_features(channels)


def test_nan_and_infinity_do_not_reach_the_model():
    channels = synthetic_channels()
    channels["I4"][0, 0] = np.nan
    channels["I5"][1, 1] = np.inf
    matrix, _ = pixel_features(channels)
    assert np.isfinite(matrix).all()


def test_mask_contract_rejects_wrong_shape_dtype_and_values():
    validate_af_mask(np.zeros(AF_SHAPE, dtype=np.uint8))
    with pytest.raises(ValueError):
        validate_af_mask(np.zeros((512, 512), dtype=np.uint8))
    with pytest.raises(TypeError):
        validate_af_mask(np.zeros(AF_SHAPE, dtype=np.int16))
    with pytest.raises(ValueError):
        validate_af_mask(np.full(AF_SHAPE, 2, dtype=np.uint8))


def test_chip_paths_resolve_from_directory_and_from_prefix(tmp_path):
    (tmp_path / "AF_te_000001_VIIRS_I1-I5.tif").write_bytes(b"")
    (tmp_path / "AF_te_000001_AUX.tif").write_bytes(b"")
    expected = (tmp_path / "AF_te_000001_VIIRS_I1-I5.tif", tmp_path / "AF_te_000001_AUX.tif")
    assert resolve_chip_paths(tmp_path) == expected
    assert resolve_chip_paths(tmp_path / "AF_te_000001") == expected
    with pytest.raises(FileNotFoundError):
        resolve_chip_paths(tmp_path / "AF_te_000002")


@pytest.mark.skipif(os.name == "nt", reason="в Windows AUX — зарезервированное имя устройства, "
                                            "каталог с таким именем создать нельзя")
def test_chip_paths_resolve_from_test_root_layout(tmp_path):
    root = tmp_path / "af"
    (root / "viirs").mkdir(parents=True)
    (root / "aux").mkdir()
    (root / "viirs" / "AF_te_000002_VIIRS_I1-I5.tif").write_bytes(b"")
    (root / "aux" / "AF_te_000002_AUX.tif").write_bytes(b"")
    assert resolve_chip_paths(root / "AF_te_000002") == (
        root / "viirs" / "AF_te_000002_VIIRS_I1-I5.tif",
        root / "aux" / "AF_te_000002_AUX.tif",
    )


def test_supplied_af_result_matches_declared_numbers():
    rows = read_rows(Path("models/active_fire/results/af_submission.csv"))
    assert validate_rows(rows, kind="af") == {"rows": 180, "nonempty": 131, "pixels": 3877}

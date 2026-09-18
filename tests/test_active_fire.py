from pathlib import Path

import numpy as np
import pytest

from models.active_fire.predict import AF_SHAPE, boxes_to_mask, render_channels
from models.common.rle import decode_rle, encode_rle
from models.common.submission import read_rows, validate_rows, write_rows


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.array([[4, 8, 12, 16], [100, 100, 104, 104]], dtype=float)
        self.conf = np.array([0.9, 0.2], dtype=float)

    def __len__(self):
        return len(self.conf)


class FakeResult:
    boxes = FakeBoxes()


def test_render_preserves_af_shape_and_invalid_pixels():
    channels = {
        "I3": np.full(AF_SHAPE, 0.3, np.float32),
        "I4": np.full(AF_SHAPE, 367.0, np.float32),
        "I5": np.full(AF_SHAPE, 300.0, np.float32),
        "valid": np.ones(AF_SHAPE, np.float32),
    }
    channels["valid"][0, 0] = 0
    image = render_channels(channels)
    assert image.shape == (1024, 1024, 3)
    assert image.dtype == np.uint8
    assert not image[:4, :4].any()
    assert image[4:, 4:].max() == 255


def test_boxes_are_returned_as_binary_uint8_mask():
    mask = boxes_to_mask(FakeResult(), confidence=0.5)
    assert mask.shape == AF_SHAPE
    assert mask.dtype == np.uint8
    assert mask.sum() == 4
    assert mask[2:4, 1:3].all()


def test_rle_is_row_major_and_canonical():
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 2:4] = 1
    mask[2, 0] = 1
    encoded = encode_rle(mask)
    assert encoded == "3 2 9 1"
    assert np.array_equal(decode_rle(encoded, mask.shape), mask)
    with pytest.raises(ValueError, match="touch"):
        decode_rle("1 2 3 1", (3, 4))


def test_supplied_af_submission_is_valid():
    rows = read_rows(Path("models/active_fire/results/active_fire_yolo.csv"))
    assert validate_rows(rows, kind="af") == {
        "rows": 180,
        "nonempty": 120,
        "pixels": 9246,
    }


def test_rejects_overlapping_bs_classes():
    rows = [
        {"chip_id": "BS_x", "class_id": 1, "rle": "10 2"},
        {"chip_id": "BS_x", "class_id": 2, "rle": "11 1"},
        {"chip_id": "BS_x", "class_id": 3, "rle": ""},
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_rows(rows, kind="bs")

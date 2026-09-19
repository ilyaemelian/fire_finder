"""Проверки общего контракта обмена и сборки итогового файла ответа."""

import numpy as np
import pytest

from models.common.rle import decode_rle, encode_rle
from models.common.submission import kinds_present, read_rows, validate_rows, write_rows
from scripts.collect_results import collect


def test_rle_is_row_major_and_canonical():
    mask = np.zeros((3, 4), dtype=np.uint8)
    mask[0, 2:4] = 1
    mask[2, 0] = 1
    encoded = encode_rle(mask)
    assert encoded == "3 2 9 1"
    assert np.array_equal(decode_rle(encoded, mask.shape), mask)
    with pytest.raises(ValueError, match="touch"):
        decode_rle("1 2 3 1", (3, 4))


def test_large_masks_survive_the_csv_field_limit(tmp_path):
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 2, (512, 512)).astype(np.uint8)
    path = tmp_path / "bs.csv"
    write_rows(path, [{"chip_id": "BS_1", "class_id": 1, "rle": encode_rle(noisy)},
                      {"chip_id": "BS_1", "class_id": 2, "rle": ""},
                      {"chip_id": "BS_1", "class_id": 3, "rle": ""}])
    rows = read_rows(path)
    assert len(rows[0]["rle"]) > 131_072
    assert validate_rows(rows, kind="bs")["pixels"] == int(noisy.sum())


def test_class_is_checked_against_the_kind_of_each_row():
    with pytest.raises(ValueError, match="AF"):
        validate_rows([{"chip_id": "AF_1", "class_id": 2, "rle": ""}])
    with pytest.raises(ValueError, match="BS"):
        validate_rows([{"chip_id": "BS_1", "class_id": 4, "rle": ""}])


def test_overlapping_bs_classes_are_rejected():
    rows = [
        {"chip_id": "BS_x", "class_id": 1, "rle": "10 2"},
        {"chip_id": "BS_x", "class_id": 2, "rle": "11 1"},
        {"chip_id": "BS_x", "class_id": 3, "rle": ""},
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_rows(rows, kind="bs")


def test_kinds_are_taken_from_chip_ids():
    assert kinds_present([{"chip_id": "AF_1", "class_id": 1}]) == {"af"}
    assert kinds_present([{"chip_id": "AF_1", "class_id": 1},
                          {"chip_id": "BS_1", "class_id": 1}]) == {"af", "bs"}
    with pytest.raises(ValueError):
        kinds_present([{"chip_id": "XX_1", "class_id": 1}])


def prepare(tmp_path, *, with_bs: bool = True, mixed: bool = False):
    root = tmp_path / "models"
    write_rows(root / "active_fire" / "results" / "af.csv",
               [{"chip_id": "AF_1", "class_id": 1, "rle": "1 1"}])
    if with_bs:
        rows = [{"chip_id": "BS_1", "class_id": 1, "rle": "2 1"},
                {"chip_id": "BS_1", "class_id": 2, "rle": "4 1"},
                {"chip_id": "BS_1", "class_id": 3, "rle": ""}]
        if mixed:                      # модуль отдал полную таблицу с пустыми чужими строками
            rows.insert(0, {"chip_id": "AF_1", "class_id": 1, "rle": ""})
        write_rows(root / "burn_severity" / "results" / "bs.csv", rows)
    sample = tmp_path / "sample.csv"
    write_rows(sample, [{"chip_id": "AF_1", "class_id": 1, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 1, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 2, "rle": ""},
                        {"chip_id": "BS_1", "class_id": 3, "rle": ""}])
    return root, sample


def test_collect_merges_modules_in_template_order(tmp_path):
    root, sample = prepare(tmp_path)
    output = tmp_path / "submission.csv"
    assert collect(sample, root, output) == {"rows": 4, "nonempty": 3, "pixels": 3}
    assert [(row["chip_id"], row["class_id"]) for row in read_rows(output)] == [
        ("AF_1", 1), ("BS_1", 1), ("BS_1", 2), ("BS_1", 3)
    ]


def test_empty_row_never_displaces_a_prediction(tmp_path):
    root, sample = prepare(tmp_path, mixed=True)
    output = tmp_path / "submission.csv"
    collect(sample, root, output)
    rows = {(r["chip_id"], r["class_id"]): r["rle"] for r in read_rows(output)}
    assert rows[("AF_1", 1)] == "1 1"


def test_conflicting_predictions_stop_the_assembly(tmp_path):
    root, sample = prepare(tmp_path)
    write_rows(root / "burn_severity" / "results" / "bs.csv",
               [{"chip_id": "AF_1", "class_id": 1, "rle": "5 1"}])
    with pytest.raises(ValueError, match="Противоречие"):
        collect(sample, root, tmp_path / "submission.csv")


def test_partial_answer_requires_an_explicit_flag(tmp_path):
    root, sample = prepare(tmp_path, with_bs=False)
    output = tmp_path / "submission.csv"
    with pytest.raises(ValueError, match="Нет предсказаний"):
        collect(sample, root, output)
    assert collect(sample, root, output, fill_from_sample=True) == {
        "rows": 4, "nonempty": 1, "pixels": 1
    }

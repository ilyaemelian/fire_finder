"""Validate module predictions and assemble the final competition CSV.

Общий для обеих моделей контракт обмена: каждая модель кладёт свой результат
в models/<module>/results/*.csv, а scripts/collect_results.py собирает из этих
файлов один outputs/submission/submission.csv строго по официальному шаблону.
"""

import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from .rle import decode_rle, encode_rle


def _raise_csv_field_limit() -> None:
    """RLE одной маски 512x512 легко превышает стандартные 128 КБ поля csv."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:  # на Windows sys.maxsize не влезает в C long
            limit //= 2


_raise_csv_field_limit()

FIELDS = ("chip_id", "class_id", "rle")
KIND_PREFIX = {"af": "AF_", "bs": "BS_"}
KIND_SHAPE = {"af": (256, 256), "bs": (512, 512)}


def kind_of(chip_id: str) -> str:
    """Вид предсказания (af/bs) по префиксу одного chip_id."""
    prefix = str(chip_id).split("_", 1)[0] + "_"
    for kind, known in KIND_PREFIX.items():
        if prefix == known:
            return kind
    raise ValueError(f"Неизвестный префикс chip_id: {chip_id!r}; ожидается AF_ или BS_")


def kinds_present(rows: Iterable[dict[str, str | int]]) -> set[str]:
    """Какие виды предсказаний есть в таблице. Файл может содержать оба."""
    return {kind_of(str(row["chip_id"])) for row in rows}


def read_rows(path: Path) -> list[dict[str, str | int]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path}: expected columns {FIELDS}")
        rows = []
        for line, source in enumerate(reader, 2):
            chip_id = (source.get("chip_id") or "").strip()
            if not chip_id:
                raise ValueError(f"{path}:{line}: empty chip_id")
            try:
                class_id = int(source.get("class_id") or "")
            except ValueError as exc:
                raise ValueError(f"{path}:{line}: invalid class_id") from exc
            rows.append({"chip_id": chip_id, "class_id": class_id, "rle": source.get("rle") or ""})
    return rows


def validate_rows(rows: Iterable[dict[str, str | int]], *, kind: str | None = None,
                  expected_pairs: set[tuple[str, int]] | None = None) -> dict[str, int]:
    seen: set[tuple[str, int]] = set()
    pixels = nonempty = count = 0
    bs_occupied: dict[str, np.ndarray] = {}
    for row in rows:
        count += 1
        chip_id = str(row["chip_id"])
        class_id = int(row["class_id"])
        pair = (chip_id, class_id)
        if pair in seen:
            raise ValueError(f"Duplicate prediction: {pair}")
        seen.add(pair)
        row_kind = kind_of(chip_id)
        if kind is not None and row_kind != kind:
            raise ValueError(f"Ожидались только предсказания {kind.upper()}, встречено: {pair}")
        if row_kind == "af" and class_id != 1:
            raise ValueError(f"Для AF допустим только class_id=1, встречено: {pair}")
        if row_kind == "bs" and class_id not in (1, 2, 3):
            raise ValueError(f"Для BS допустимы class_id 1, 2, 3; встречено: {pair}")
        shape = KIND_SHAPE[row_kind]
        mask = decode_rle(str(row.get("rle") or ""), shape)
        if row_kind == "bs":
            occupied = bs_occupied.setdefault(chip_id, np.zeros(shape, dtype=np.uint8))
            if np.logical_and(occupied, mask).any():
                raise ValueError(f"BS class masks overlap for {chip_id}")
            occupied |= mask
        if encode_rle(mask) != str(row.get("rle") or "").strip():
            raise ValueError(f"Non-canonical RLE for {pair}")
        positive = int(mask.sum())
        pixels += positive
        nonempty += int(positive > 0)
    if expected_pairs is not None and seen != expected_pairs:
        missing = sorted(expected_pairs - seen)[:5]
        extra = sorted(seen - expected_pairs)[:5]
        raise ValueError(f"Prediction/template mismatch; missing={missing}, extra={extra}")
    return {"rows": count, "nonempty": nonempty, "pixels": pixels}


def write_rows(path: Path, rows: Iterable[dict[str, str | int]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

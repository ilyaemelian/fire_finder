"""CLI модуля BS: прогон CatBoost по тестовым чипам и запись results/burn_severity_catboost.csv.

Строк на чип ровно три — по одной на классы 1, 2 и 3; класс 0 в сабмишн не идёт.
"""

import argparse
import csv
import time
from pathlib import Path

from models.burn_severity import RESULTS, WEIGHTS
from models.burn_severity.predict import load_model, predict_bs
from models.common.rle import encode_rle
from models.common.submission import validate_rows, write_rows

CLASS_IDS = (1, 2, 3)


def bs_chip_ids(sample_path: Path) -> list[str]:
    with sample_path.open(encoding="utf-8-sig", newline="") as stream:
        template = list(csv.DictReader(stream))
    ids, seen = [], set()
    for row in template:
        chip_id = (row.get("chip_id") or "").strip()
        if chip_id.startswith("BS_") and chip_id not in seen:
            ids.append(chip_id)
            seen.add(chip_id)
    if not ids:
        raise ValueError("The sample submission contains no BS chips")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BS CatBoost module on competition test chips")
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Корень тестовых данных: <data-dir>/bs/<chip_id>_pre.tif и _post.tif")
    parser.add_argument("--sample", type=Path, required=True, help="Официальный sample_submission.csv")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    chip_ids = bs_chip_ids(args.sample)
    model = load_model(args.weights)
    chips_dir = args.data_dir / "bs"
    if not chips_dir.is_dir():
        chips_dir = args.data_dir
    rows = []
    started = time.perf_counter()
    for index, chip_id in enumerate(chip_ids, 1):
        mask = predict_bs(chips_dir / chip_id, model)
        rows.extend({"chip_id": chip_id, "class_id": class_id,
                     "rle": encode_rle((mask == class_id).astype("uint8"))}
                    for class_id in CLASS_IDS)
        if index % 10 == 0 or index == len(chip_ids):
            elapsed = time.perf_counter() - started
            print(f"BS inference: {index}/{len(chip_ids)} · {elapsed:.0f} s "
                  f"({elapsed / index:.2f} s/chip)", flush=True)
    expected = {(chip_id, class_id) for chip_id in chip_ids for class_id in CLASS_IDS}
    stats = validate_rows(rows, kind="bs", expected_pairs=expected)
    write_rows(args.output, rows)
    print(f"Saved {args.output}: {stats['rows']} rows, {stats['nonempty']} non-empty, "
          f"{stats['pixels']} positive pixels")


if __name__ == "__main__":
    main()

"""CLI модуля AF: прогон YOLO по тестовым чипам и запись results/active_fire_yolo.csv."""

import argparse
import csv
import time
from pathlib import Path

from models.active_fire import RESULTS, WEIGHTS
from models.active_fire.predict import load_model, predict_af
from models.common.rle import encode_rle
from models.common.submission import validate_rows, write_rows


def af_chip_ids(sample_path: Path) -> list[str]:
    with sample_path.open(encoding="utf-8-sig", newline="") as stream:
        template = list(csv.DictReader(stream))
    ids, seen = [], set()
    for row in template:
        chip_id = (row.get("chip_id") or "").strip()
        if chip_id.startswith("AF_") and chip_id not in seen:
            ids.append(chip_id)
            seen.add(chip_id)
    if not ids:
        raise ValueError("The sample submission contains no AF chips")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AF YOLO module on competition test chips")
    parser.add_argument("--data-dir", type=Path, required=True, help="Корень тестовых данных: <data-dir>/af/viirs/*.tif")
    parser.add_argument("--sample", type=Path, required=True, help="Официальный sample_submission.csv")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--device", default="cpu", help="cpu или индекс CUDA, например 0")
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    chip_ids = af_chip_ids(args.sample)
    model = load_model(args.weights, args.device)
    viirs_dir = args.data_dir / "af" / "viirs"
    rows = []
    started = time.perf_counter()
    for index, chip_id in enumerate(chip_ids, 1):
        chip_path = viirs_dir / f"{chip_id}_VIIRS_I1-I5.tif"
        rows.append({"chip_id": chip_id, "class_id": 1,
                     "rle": encode_rle(predict_af(chip_path, model))})
        if index % 20 == 0 or index == len(chip_ids):
            elapsed = time.perf_counter() - started
            print(f"AF inference: {index}/{len(chip_ids)} · {elapsed:.0f} s "
                  f"({elapsed / index:.2f} s/chip)", flush=True)
    stats = validate_rows(rows, kind="af", expected_pairs={(chip_id, 1) for chip_id in chip_ids})
    write_rows(args.output, rows)
    print(f"Saved {args.output}: {stats['rows']} rows, {stats['nonempty']} non-empty, "
          f"{stats['pixels']} positive pixels")


if __name__ == "__main__":
    main()

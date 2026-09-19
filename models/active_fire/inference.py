"""Финальный расчёт предсказаний модуля AF.

    python -m models.active_fire.inference --data-dir <каталог теста>

Каталог теста должен содержать `sample_submission.csv` и подкаталог `af/` с
`viirs/` и `aux/`. Результат по умолчанию — `models/active_fire/results/af_submission.csv`,
проверенный по правилам конкурса до записи.
"""

import argparse
import csv
import time
from pathlib import Path

from models.active_fire import RESULTS, WEIGHTS
from models.active_fire.predict import load_model, predict_af
from models.common.rle import encode_rle
from models.common.submission import validate_rows, write_rows


def af_chip_ids(sample_path: Path) -> list[str]:
    """Идентификаторы AF-чипов в порядке официального шаблона."""
    with Path(sample_path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids, seen = [], set()
    for row in rows:
        chip_id = (row.get("chip_id") or "").strip()
        if chip_id.startswith("AF_") and chip_id not in seen:
            ids.append(chip_id)
            seen.add(chip_id)
    if not ids:
        raise ValueError(f"{sample_path}: в шаблоне нет AF-чипов")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Расчёт предсказаний модуля AF по тестовым чипам"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="каталог теста: внутри af/viirs, af/aux и sample_submission.csv")
    parser.add_argument("--sample", type=Path,
                        help="путь к шаблону, по умолчанию <data-dir>/sample_submission.csv")
    parser.add_argument("--model", type=Path, default=WEIGHTS)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--threshold", type=float,
                        help="заменить порог, сохранённый в модели")
    args = parser.parse_args()

    sample_path = args.sample or args.data_dir / "sample_submission.csv"
    chip_ids = af_chip_ids(sample_path)
    model = load_model(args.model)
    threshold = model.threshold if args.threshold is None else args.threshold
    chips_root = args.data_dir / "af" if (args.data_dir / "af").is_dir() else args.data_dir

    rows = []
    started = time.perf_counter()
    for index, chip_id in enumerate(chip_ids, 1):
        mask = predict_af(chips_root / chip_id, model, threshold)
        rows.append({"chip_id": chip_id, "class_id": 1, "rle": encode_rle(mask)})
        if index % 20 == 0 or index == len(chip_ids):
            elapsed = time.perf_counter() - started
            print(f"обработано {index} из {len(chip_ids)} · {elapsed:.0f} с "
                  f"({elapsed / index:.2f} с на чип)", flush=True)

    stats = validate_rows(rows, kind="af",
                          expected_pairs={(chip_id, 1) for chip_id in chip_ids})
    write_rows(args.output, rows)
    print(f"Записан {args.output}: {stats['rows']} строк, {stats['nonempty']} непустых матриц, "
          f"{stats['pixels']} пикселей горения | порог {threshold}")


if __name__ == "__main__":
    main()

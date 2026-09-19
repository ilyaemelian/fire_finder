"""Финальный расчёт предсказаний модуля AF.

    python -m models.active_fire.inference --data-dir <каталог теста>

Каталог теста должен содержать `sample_submission.csv` и подкаталог `af/` с
`viirs/` и `aux/`. Результат по умолчанию — `models/active_fire/results/af_submission.csv`,
проверенный по правилам конкурса до записи.

Расчёт вынесен в функцию `run`: ею пользуется и эта команда, и полный пайплайн
`scripts/run_af_pipeline.py`, который вокруг расчёта добавляет проверку среды,
весов и сборку сабмита по шаблону.
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


def run(data_dir: Path, *, sample: Path | None = None, model_path: Path = WEIGHTS,
        output: Path = RESULTS, threshold: float | None = None,
        progress: bool = True) -> dict:
    """Расчёт по всем AF-чипам шаблона и запись проверенной таблицы предсказаний."""
    data_dir = Path(data_dir)
    sample_path = Path(sample) if sample else data_dir / "sample_submission.csv"
    chip_ids = af_chip_ids(sample_path)
    model = load_model(model_path)
    level = model.threshold if threshold is None else float(threshold)
    chips_root = data_dir / "af" if (data_dir / "af").is_dir() else data_dir

    rows = []
    started = time.perf_counter()
    for index, chip_id in enumerate(chip_ids, 1):
        mask = predict_af(chips_root / chip_id, model, level)
        rows.append({"chip_id": chip_id, "class_id": 1, "rle": encode_rle(mask)})
        if progress and (index % 60 == 0 or index == len(chip_ids)):
            elapsed = time.perf_counter() - started
            print(f"    обработано {index} из {len(chip_ids)}, {elapsed:.0f} с", flush=True)

    stats = validate_rows(rows, kind="af", expected_pairs={(chip_id, 1) for chip_id in chip_ids})
    write_rows(output, rows)
    return {"rows": rows, "stats": stats, "threshold": level, "output": Path(output),
            "chip_ids": chip_ids, "seconds": time.perf_counter() - started}


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

    result = run(args.data_dir, sample=args.sample, model_path=args.model,
                 output=args.output, threshold=args.threshold)
    stats = result["stats"]
    print(f"Записан {result['output']}: {stats['rows']} строк, {stats['nonempty']} непустых матриц, "
          f"{stats['pixels']} пикселей горения | порог {result['threshold']}")


if __name__ == "__main__":
    main()

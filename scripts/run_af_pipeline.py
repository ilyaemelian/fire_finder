"""Полный пайплайн модуля AF: от тестовых чипов до готового файла ответа.

    .venv-af\\Scripts\\python.exe -m scripts.run_af_pipeline --data-dir <каталог теста>

Пять шагов, каждый прекращает работу при первой же неувязке:

1. среда — библиотеки из `requirements-af.txt` и их версии;
2. вход — шаблон, каталог чипов, число AF-фрагментов, наличие файлов первого чипа;
3. веса — `artifacts/model.pkl` на месте и совпадает с контрольной суммой паспорта;
4. расчёт — `models/active_fire/results/af_submission.csv`, проверенный по правилам конкурса;
5. сабмит — `outputs/submission/submission_af.csv`: строки AF заполнены, строки BS
   взяты из шаблона пустыми.

Пятый шаг и даёт «первый сабмит»: файл полон по структуре, его можно отправить
на лидерборд и увидеть вклад одного модуля отдельно. Общий ответ из результатов
обеих моделей собирает `scripts/collect_results.py`.
"""

import argparse
import json
import time
from pathlib import Path

from models.active_fire import MODULE_ROOT, RESULTS, WEIGHTS
from models.active_fire.features import resolve_chip_paths
from models.active_fire.inference import af_chip_ids, run
from models.common.submission import read_rows
from scripts.collect_results import fill_template
from scripts.pipeline_common import describe_test_dir, require, step, verify_weights

PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "rasterio": "rasterio",
    "lightgbm": "lightgbm",
    "sklearn": "scikit-learn",
}
DEFAULT_SUBMISSION = Path("outputs/submission/submission_af.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Полный пайплайн модуля AF: проверки, расчёт, файл ответа"
    )
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="каталог теста: внутри af/ и sample_submission.csv")
    parser.add_argument("--sample", type=Path,
                        help="путь к шаблону, по умолчанию <data-dir>/sample_submission.csv")
    parser.add_argument("--model", type=Path, default=WEIGHTS)
    parser.add_argument("--results", type=Path, default=RESULTS,
                        help="таблица предсказаний модуля")
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION,
                        help="файл ответа по шаблону")
    parser.add_argument("--threshold", type=float,
                        help="заменить порог, сохранённый в модели")
    parser.add_argument("--reuse-results", action="store_true",
                        help="не считать заново: взять готовую таблицу из --results")
    parser.add_argument("--skip-checksums", action="store_true",
                        help="не сверять контрольные суммы весов")
    args = parser.parse_args()
    started = time.perf_counter()
    sample_path = args.sample or args.data_dir / "sample_submission.csv"

    step(1, "Среда")
    require(PACKAGES, "requirements-af.txt")

    step(2, "Вход")
    if not Path(sample_path).is_file():
        raise SystemExit(f"Шаблон не найден: {sample_path}")
    chip_ids = af_chip_ids(sample_path)
    describe_test_dir(args.data_dir, Path(sample_path), "af", len(chip_ids))
    if not args.reuse_results:
        chips_root = args.data_dir / "af" if (args.data_dir / "af").is_dir() else args.data_dir
        viirs, aux = resolve_chip_paths(chips_root / chip_ids[0])
        print(f"    первый чип      {viirs.name}, {aux.name}")

    step(3, "Веса")
    manifest_path = Path(args.model).with_name("manifest.json")
    if not manifest_path.is_file():
        manifest_path = MODULE_ROOT / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    verify_weights({Path(args.model): manifest.get("weights_sha256")},
                   checksums=not args.skip_checksums)
    print(f"    {manifest['algorithm']}, признаков {manifest['features']}, "
          f"порог {manifest['probability_threshold']}")

    step(4, "Расчёт предсказаний")
    if args.reuse_results:
        rows = read_rows(args.results)
        print(f"    взята готовая таблица {args.results}: {len(rows)} строк")
    else:
        result = run(args.data_dir, sample=sample_path, model_path=args.model,
                     output=args.results, threshold=args.threshold)
        rows, stats = result["rows"], result["stats"]
        print(f"    {result['output']}: {stats['rows']} строк, {stats['nonempty']} непустых "
              f"матриц, {stats['pixels']} пикселей горения | порог {result['threshold']} | "
              f"{result['seconds']:.0f} с")

    step(5, "Файл ответа по шаблону")
    stats = fill_template(sample_path, rows, args.output)
    print(f"    {args.output}: {stats['rows']} строк, {stats['nonempty']} непустых матриц, "
          f"{stats['pixels']} положительных пикселей")

    print(f"\nГотово за {time.perf_counter() - started:.0f} с. "
          f"Отправить на лидерборд: {args.output}")
    print("Общий ответ обеих моделей: python -m scripts.collect_results "
          f"--sample {sample_path}")


if __name__ == "__main__":
    main()

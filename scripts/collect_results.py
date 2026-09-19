"""Формирование итогового файла ответа из результатов моделей.

Единственное место, где составляющие системы встречаются. Скрипт находит
таблицы предсказаний models/<модуль>/results/*.csv, проверяет каждую отдельно и
объединяет строки строго в порядке официального шаблона.

    python -m scripts.collect_results --sample data/sample_submission.csv

Файл модуля может содержать строки обоих видов: например, модуль BS выдал полную
таблицу, где строки AF пустые. Пустая строка никогда не вытесняет непустую;
противоречие двух непустых предсказаний по одной паре прекращает работу.

Если результат одной модели ещё не получен, добавьте --fill-from-sample:
недостающие строки будут взяты из шаблона, то есть с пустым полем rle.

Проверить одну таблицу, ничего не формируя:

    python -m scripts.collect_results --check models/active_fire/results/af_submission.csv
"""

import argparse
from collections.abc import Iterable
from pathlib import Path

from models.common.submission import (
    kinds_present,
    read_rows,
    validate_rows,
    write_rows,
)

DEFAULT_MODELS_ROOT = Path("models")
DEFAULT_OUTPUT = Path("outputs/submission/submission.csv")

Row = dict[str, str | int]
Pair = tuple[str, int]


def discover_result_files(models_root: Path) -> dict[str, Path]:
    """models/<модуль>/results/*.csv -> {модуль: самый свежий файл}."""
    found: dict[str, Path] = {}
    for results_dir in sorted(models_root.glob("*/results")):
        candidates = sorted(results_dir.glob("*.csv"))
        if not candidates:
            continue
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        if len(candidates) > 1:
            others = ", ".join(path.name for path in candidates if path != newest)
            print(f"[{results_dir.parent.name}] в каталоге несколько таблиц, взята последняя "
                  f"по времени изменения: {newest.name}; не взяты: {others}")
        found[results_dir.parent.name] = newest
    return found


def check_file(path: Path) -> dict[str, int]:
    """Проверить одну таблицу предсказаний и вывести статистику."""
    rows = read_rows(path)
    stats = validate_rows(rows)
    kinds = "+".join(sorted(kind.upper() for kind in kinds_present(rows)))
    print(f"Ошибок нет: {path} · {kinds} · {stats['rows']} строк, "
          f"{stats['nonempty']} непустых матриц, {stats['pixels']} положительных пикселей")
    return stats


def fill_template(sample_path: Path, rows: Iterable[Row], output_path: Path) -> dict[str, int]:
    """Разложить строки одного модуля по шаблону и записать файл ответа.

    Строки, которых модуль не даёт, берутся из шаблона, то есть с пустым полем
    `rle`. Получается файл, полный по структуре и заведомо неполный по
    содержанию: его можно отправить на лидерборд, чтобы увидеть вклад одного
    модуля отдельно. Этим пользуются `scripts/run_af_pipeline.py` и
    `scripts/run_bs_pipeline.py`; общий ответ по-прежнему собирает `collect`.
    """
    template = read_rows(sample_path)
    order = [(str(row["chip_id"]), int(row["class_id"])) for row in template]
    template_rows = {pair: row for pair, row in zip(order, template)}
    predictions = {(str(row["chip_id"]), int(row["class_id"])): row for row in rows}

    extra = sorted(set(predictions) - set(order))[:5]
    if extra:
        raise ValueError(f"В таблице есть строки, отсутствующие в шаблоне: {extra}")

    merged = [predictions.get(pair, template_rows[pair]) for pair in order]
    stats = validate_rows(merged, expected_pairs=set(order))
    write_rows(output_path, merged)
    borrowed = len(order) - len(predictions)
    if borrowed:
        print(f"Из шаблона взято {borrowed} строк с пустым полем rle: "
              f"по ним предсказаний этого модуля нет.")
    return stats


def load_predictions(models_root: Path) -> dict[Pair, Row]:
    """Прочитать таблицы всех модулей и свести их в одно отображение пара -> строка."""
    discovered = discover_result_files(models_root)
    if not discovered:
        raise FileNotFoundError(
            f"Таблицы предсказаний не найдены: {models_root}/*/results/*.csv"
        )
    predictions: dict[Pair, Row] = {}
    origin: dict[Pair, Path] = {}
    overridden = 0
    for module, path in sorted(discovered.items()):
        rows = read_rows(path)
        stats = validate_rows(rows)
        kinds = "+".join(sorted(kind.upper() for kind in kinds_present(rows)))
        print(f"[{module}] {path} · {kinds} · {stats['rows']} строк, "
              f"{stats['nonempty']} непустых матриц, {stats['pixels']} положительных пикселей")
        for row in rows:
            pair = (str(row["chip_id"]), int(row["class_id"]))
            value = str(row.get("rle") or "").strip()
            if pair not in predictions:
                predictions[pair] = row
                origin[pair] = path
                continue
            previous = str(predictions[pair].get("rle") or "").strip()
            if previous == value:
                continue
            if not previous:                      # пустое место занимает непустое
                predictions[pair] = row
                origin[pair] = path
                overridden += 1
            elif value:                           # два непустых предсказания
                raise ValueError(
                    f"Противоречие по {pair}: непустые предсказания в {origin[pair]} и {path}. "
                    f"Оставьте одну таблицу для этого вида предсказаний."
                )
    if overridden:
        print(f"Пустых строк замещено непустыми из других таблиц: {overridden}")
    return predictions


def collect(sample_path: Path, models_root: Path, output_path: Path,
            fill_from_sample: bool = False) -> dict[str, int]:
    template = read_rows(sample_path)
    order = [(str(row["chip_id"]), int(row["class_id"])) for row in template]
    template_rows = {pair: row for pair, row in zip(order, template)}

    predictions = load_predictions(models_root)

    extra = sorted(set(predictions) - set(order))[:5]
    if extra:
        raise ValueError(f"В таблицах есть строки, отсутствующие в шаблоне: {extra}")

    missing = [pair for pair in order if pair not in predictions]
    empty = [pair for pair in order
             if pair in predictions and not str(predictions[pair].get("rle") or "").strip()]
    if missing:
        by_kind: dict[str, int] = {}
        for chip_id, _ in missing:
            prefix = chip_id.split("_", 1)[0]
            by_kind[prefix] = by_kind.get(prefix, 0) + 1
        breakdown = ", ".join(f"{name}: {count}" for name, count in sorted(by_kind.items()))
        if not fill_from_sample:
            raise ValueError(
                f"Нет предсказаний для {len(missing)} строк шаблона ({breakdown}). "
                f"Первые: {missing[:3]}. Взять их из шаблона: --fill-from-sample"
            )
        print(f"Внимание: {len(missing)} строк взяты из шаблона ({breakdown}). "
              f"Ответ неполон по содержанию: по этим фрагментам сдаются пустые матрицы.")
        for pair in missing:
            predictions[pair] = template_rows[pair]

    merged = [predictions[pair] for pair in order]
    stats = validate_rows(merged, expected_pairs=set(order))
    write_rows(output_path, merged)
    if empty:
        print(f"Пустых матриц среди предсказаний моделей: {len(empty)} "
              f"(обнаружений на этих фрагментах нет)")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сформировать итоговый файл ответа из models/*/results/*.csv"
    )
    parser.add_argument("--check", type=Path, metavar="CSV",
                        help="проверить одну таблицу предсказаний и завершить работу")
    parser.add_argument("--sample", type=Path, help="официальный шаблон sample_submission.csv")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fill-from-sample", action="store_true",
                        help="недостающие строки взять из шаблона (пустые матрицы)")
    args = parser.parse_args()

    if args.check:
        check_file(args.check)
        return
    if not args.sample:
        parser.error("требуется --sample (или --check для проверки одной таблицы)")
    stats = collect(args.sample, args.models_root, args.output, args.fill_from_sample)
    print(f"Создан {args.output}: {stats['rows']} строк, {stats['nonempty']} непустых матриц, "
          f"{stats['pixels']} положительных пикселей")


if __name__ == "__main__":
    main()

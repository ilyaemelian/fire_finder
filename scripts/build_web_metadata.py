"""Сформировать отдельную геопривязку для показа submission.csv в веб-сервисе.

    .venv\\Scripts\\python.exe -m scripts.build_web_metadata --data-dir data/train

Конкурсный CSV содержит только `chip_id`, `class_id` и `rle`: ни координат, ни
дат в нём нет. Чтобы показать маски на карте, веб-сервису нужен отдельный файл
`web_metadata.json` с системой координат, аффинным преобразованием и датами
съёмки каждого фрагмента. Структура CSV при этом не меняется.

Источник сведений — `meta.csv` рядом с фрагментами. Нужны столбцы `epsg`,
`x_min`, `y_max`, `gsd` для геопривязки, `acq_datetime` для фрагментов AF и
`date_pre` с `date_post` для фрагментов BS. Если в `meta.csv` таких столбцов
нет, сведения берутся из самих растров, а при их отсутствии — из файла,
указанного параметром `--meta`.

Замечание по составу выданных данных: полный `meta.csv` присутствует только у
обучающего набора `data/train`. У тестового набора `data/test` из `meta.csv`
удалены столбцы геопривязки и дат, а растры записаны без системы координат,
поэтому построить карту по тестовым фрагментам без внешнего файла `--meta`
невозможно. Команда сообщает об этом явно и называет недостающие столбцы.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import rasterio

GEOMETRY_COLUMNS = ("epsg", "x_min", "y_max", "gsd")
AF_RASTER = ("viirs/{chip}_VIIRS_I1-I5.tif", "{chip}/{chip}_VIIRS_I1-I5.tif")
BS_RASTER = ("sentinel2_post/{chip}_Sentinel-2_post.tif",
             "{chip}/{chip}_Sentinel-2_post.tif")


def blank(value: object) -> bool:
    """Пустое значение: пропуск, пустая строка или строковое «nan»."""
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def read_table(path: Path) -> dict[str, dict[str, str]]:
    """Прочитать таблицу, ключ — chip_id. Отсутствующий файл даёт пустой словарь."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {
            (row.get("chip_id") or "").strip(): row
            for row in csv.DictReader(stream)
            if (row.get("chip_id") or "").strip()
        }


def collect_meta(data_dir: Path, override: Path | None) -> dict[str, dict[str, str]]:
    """Свести `meta.csv` обоих модулей и внешний файл. Внешний имеет приоритет."""
    table: dict[str, dict[str, str]] = {}
    for candidate in (data_dir / "meta.csv",
                      data_dir / "af" / "meta.csv",
                      data_dir / "bs" / "meta.csv"):
        for chip_id, row in read_table(candidate).items():
            table.setdefault(chip_id, {}).update(
                {key: value for key, value in row.items() if not blank(value)}
            )
    if override is not None:
        if not override.is_file():
            raise SystemExit(f"Не найден файл сведений: {override}")
        for chip_id, row in read_table(override).items():
            table.setdefault(chip_id, {}).update(
                {key: value for key, value in row.items() if not blank(value)}
            )
    return table


def chip_ids(data_dir: Path, sample: Path | None,
             table: dict[str, dict[str, str]]) -> list[str]:
    """Идентификаторы в порядке шаблона, а при его отсутствии — в порядке meta.csv."""
    if sample is not None:
        with sample.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            ids = list(dict.fromkeys((row["chip_id"] or "").strip() for row in rows))
        if ids:
            return ids
    ids = sorted(table)
    if not ids:
        raise SystemExit(
            f"В {data_dir} не найдены ни sample_submission.csv, ни meta.csv "
            f"с идентификаторами фрагментов."
        )
    return ids


def module_root(data_dir: Path, kind: str) -> Path:
    inner = data_dir / kind
    return inner if inner.is_dir() else data_dir


def raster_path(data_dir: Path, chip_id: str, kind: str) -> Path | None:
    root = module_root(data_dir, kind)
    for pattern in (AF_RASTER if kind == "af" else BS_RASTER):
        candidate = root / pattern.format(chip=chip_id)
        if candidate.is_file():
            return candidate
    return None


def geometry_from_meta(row: dict[str, str]) -> tuple[str, list[float]] | None:
    """Система координат и аффинное преобразование по столбцам meta.csv."""
    if any(blank(row.get(column)) for column in GEOMETRY_COLUMNS):
        return None
    try:
        code = int(float(row["epsg"]))
        x_min = float(row["x_min"])
        y_max = float(row["y_max"])
        gsd = float(row["gsd"])
    except (TypeError, ValueError):
        return None
    if not all(map(math.isfinite, (x_min, y_max, gsd))) or gsd <= 0:
        return None
    # Растр ориентирован на север: шаг по строкам отрицательный, начало — верхний
    # левый угол. Порядок значений совпадает с классом Affine.
    return f"EPSG:{code}", [gsd, 0.0, x_min, 0.0, -gsd, y_max]


def geometry_from_raster(path: Path | None) -> tuple[str, list[float]] | None:
    """Система координат и преобразование из самого растра, если они записаны."""
    if path is None:
        return None
    with rasterio.open(path) as source:
        transform = list(source.transform)[:6]
        if source.crs is None or transform == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
            return None
        return source.crs.to_string(), transform


def describe(chip_id: str, row: dict[str, str], kind: str) -> list[str]:
    """Перечислить, каких именно сведений не хватает для фрагмента."""
    missing = [column for column in GEOMETRY_COLUMNS if blank(row.get(column))]
    if kind == "af" and blank(row.get("acq_datetime")):
        missing.append("acq_datetime")
    if kind == "bs":
        missing += [name for name in ("date_pre", "date_post") if blank(row.get(name))]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Создать web_metadata.json — геопривязку и даты для submission.csv"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"),
                        help="каталог с фрагментами и meta.csv")
    parser.add_argument("--meta", "--dates", dest="meta", type=Path,
                        help="внешний CSV: chip_id,epsg,x_min,y_max,gsd,"
                             "acq_datetime,date_pre,date_post; дополняет meta.csv")
    parser.add_argument("--sample", type=Path,
                        help="шаблон, задающий состав и порядок фрагментов")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/submission/web_metadata.json"))
    parser.add_argument("--kind", choices=("af", "bs", "all"), default="all",
                        help="включить только фрагменты одного модуля")
    parser.add_argument("--submission", type=Path,
                        help="оставить только фрагменты, присутствующие в этом CSV")
    parser.add_argument("--limit", type=int,
                        help="ограничить число фрагментов каждого модуля")
    parser.add_argument("--limit-af", type=int,
                        help="ограничить число фрагментов AF отдельно от BS")
    parser.add_argument("--limit-bs", type=int,
                        help="ограничить число фрагментов BS; расчёт BS "
                             "существенно тяжелее расчёта AF")
    parser.add_argument("--make-sample", type=Path,
                        help="дополнительно записать шаблон состава строк "
                             "в конкурсном формате для расчёта по этим фрагментам")
    parser.add_argument("--title", default="Результаты моделей")
    parser.add_argument("--source", default="Расчёт моделей AF и BS")
    parser.add_argument("--af-version", default="af-model")
    parser.add_argument("--bs-version", default="bs-model")
    arguments = parser.parse_args()

    data_dir = arguments.data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Не найден каталог данных: {data_dir}")

    sample = arguments.sample
    if sample is None and (data_dir / "sample_submission.csv").is_file():
        sample = data_dir / "sample_submission.csv"

    table = collect_meta(data_dir, arguments.meta)
    identifiers = chip_ids(data_dir, sample, table)

    if arguments.submission is not None:
        if not arguments.submission.is_file():
            raise SystemExit(f"Не найден файл предсказаний: {arguments.submission}")
        csv.field_size_limit(1 << 30)
        with arguments.submission.open(encoding="utf-8-sig", newline="") as stream:
            wanted = {(row.get("chip_id") or "").strip()
                      for row in csv.DictReader(stream)}
        identifiers = [chip_id for chip_id in identifiers if chip_id in wanted]
        if not identifiers:
            raise SystemExit(
                f"Ни один фрагмент из {arguments.submission} не описан сведениями "
                f"каталога {data_dir}.\nИмена обучающих фрагментов начинаются с "
                f"AF_tr и BS_tr, тестовых — с AF_te и BS_te; оба файла должны "
                f"относиться к одному набору."
            )

    ceiling = {
        "af": arguments.limit_af if arguments.limit_af is not None else arguments.limit,
        "bs": arguments.limit_bs if arguments.limit_bs is not None else arguments.limit,
    }
    if any(value is not None for value in ceiling.values()):
        taken = {"af": 0, "bs": 0}
        selected = []
        for chip_id in identifiers:
            group = "af" if chip_id.startswith("AF_") else "bs"
            allowed = ceiling[group]
            if allowed is None or taken[group] < allowed:
                taken[group] += 1
                selected.append(chip_id)
        identifiers = selected

    chips: dict[str, dict] = {}
    incomplete: list[tuple[str, list[str]]] = []
    for chip_id in identifiers:
        if chip_id.startswith("AF_"):
            kind = "af"
        elif chip_id.startswith("BS_"):
            kind = "bs"
        else:
            raise SystemExit(f"Неизвестный вид фрагмента по имени: {chip_id}")
        if arguments.kind != "all" and arguments.kind != kind:
            continue

        row = table.get(chip_id, {})
        geometry = geometry_from_meta(row) or geometry_from_raster(
            raster_path(data_dir, chip_id, kind)
        )
        if kind == "af":
            moment = (row.get("acq_datetime") or "").strip()
            complete = geometry is not None and not blank(moment)
        else:
            date_pre = (row.get("date_pre") or "").strip()
            date_post = (row.get("date_post") or "").strip()
            complete = geometry is not None and not blank(date_pre) and not blank(date_post)

        if not complete:
            incomplete.append((chip_id, describe(chip_id, row, kind)))
            continue

        crs, transform = geometry
        record = {"kind": kind, "observation_id": chip_id, "crs": crs,
                  "transform": transform}
        if kind == "af":
            record["model_version"] = arguments.af_version
            record["acquisition_datetime"] = moment
        else:
            record["model_version"] = arguments.bs_version
            record["date_pre"] = date_pre
            record["date_post"] = date_post
        chips[chip_id] = record

    if not chips:
        columns = sorted({name for _, names in incomplete for name in names})
        raise SystemExit(
            f"Ни для одного фрагмента в {data_dir} не нашлось геопривязки и дат.\n"
            f"Не хватает столбцов: {', '.join(columns) or 'неизвестно'}.\n"
            f"У выданного тестового набора эти столбцы удалены из meta.csv, а растры\n"
            f"записаны без системы координат. Возможные действия:\n"
            f"  1. Построить набор по обучающим данным: --data-dir data/train\n"
            f"  2. Передать сведения отдельным файлом: --meta <файл.csv> со\n"
            f"     столбцами chip_id, epsg, x_min, y_max, gsd, acq_datetime,\n"
            f"     date_pre, date_post"
        )

    payload = {"title": arguments.title, "data_mode": "real",
               "source": arguments.source, "chips": chips}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if arguments.make_sample is not None:
        arguments.make_sample.parent.mkdir(parents=True, exist_ok=True)
        with arguments.make_sample.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["chip_id", "class_id", "rle"])
            for chip_id, record in chips.items():
                for class_id in ((1,) if record["kind"] == "af" else (1, 2, 3)):
                    writer.writerow([chip_id, class_id, ""])
        print(f"Создан {arguments.make_sample}: шаблон состава строк")

    print(f"Создан {arguments.output}: {len(chips)} фрагментов "
          f"({sum(1 for r in chips.values() if r['kind'] == 'af')} AF, "
          f"{sum(1 for r in chips.values() if r['kind'] == 'bs')} BS)")
    if incomplete:
        print(f"Пропущено без геопривязки или дат: {len(incomplete)}")
        for chip_id, names in incomplete[:5]:
            print(f"  {chip_id}: нет {', '.join(names) or 'сведений'}")
        if len(incomplete) > 5:
            print(f"  и ещё {len(incomplete) - 5}")
        print("Загружаемый CSV должен содержать те же фрагменты, что и этот файл: "
              "отберите строки по перечисленным идентификаторам "
              "либо дополните сведения параметром --meta.")


if __name__ == "__main__":
    main()

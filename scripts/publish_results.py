"""Поместить таблицу предсказаний и геопривязку прямо в хранилище сервиса.

    .venv\\Scripts\\python.exe -m scripts.publish_results --submission outputs/demo/demo.csv --metadata outputs/demo/web_metadata.json --title "Предсказания моделей"

Делает то же, что загрузка двух файлов через интерфейс, но из командной строки
и с подробным разбором причин отказа. Набор появляется в списке сразу после
запуска сервиса, вручную загружать ничего не нужно.

Полезно и при проверке: команда печатает число термоточек, контуров, границы
набора и диапазон дат, поэтому сразу видно, будет ли на карте что показывать.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError
from shapely.geometry import shape
from shapely.ops import unary_union

from src.analytics.masks import convert_submission, read_submission
from src.service.repository import save_json, data_root
from src.service.schemas import SubmissionMetadata


def summarise(data: dict) -> None:
    """Напечатать то, по чему сразу видно, появится ли что-нибудь на карте."""
    points = data["active_fires"]["features"]
    contours = data["burned_areas"]["features"]
    observations = data["observations"]["features"]
    print(f"  термоточек: {len(points)}")
    print(f"  контуров гари: {len(contours)}")

    if observations:
        bounds = unary_union([shape(f["geometry"]) for f in observations]).bounds
        print("  границы набора: "
              f"{bounds[0]:.3f}, {bounds[1]:.3f} — {bounds[2]:.3f}, {bounds[3]:.3f}")
        dates = sorted(
            (f["properties"].get("acquisition_datetime")
             or f["properties"].get("date_post"))[:10]
            for f in observations
        )
        print(f"  даты наблюдений: {dates[0]} — {dates[-1]}")

    if not points and not contours:
        print("  Внимание: обнаружений нет. Карта будет пустой — это значит, "
              "что во всех масках нет ни одного положительного пикселя.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Зарегистрировать результаты расчёта в веб-сервисе без "
                    "загрузки через интерфейс"
    )
    parser.add_argument("--submission", type=Path, required=True,
                        help="таблица предсказаний в конкурсном формате")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="web_metadata.json из scripts.build_web_metadata")
    parser.add_argument("--name", default="results",
                        help="имя набора в хранилище; повторный запуск с тем же "
                             "именем заменяет набор")
    parser.add_argument("--title",
                        help="заголовок в списке наборов; по умолчанию берётся "
                             "из web_metadata.json")
    arguments = parser.parse_args()

    for path in (arguments.submission, arguments.metadata):
        if not path.is_file():
            raise SystemExit(f"Не найден файл: {path}")

    csv.field_size_limit(1 << 30)
    try:
        payload = json.loads(arguments.metadata.read_text(encoding="utf-8-sig"))
        if arguments.title:
            payload["title"] = arguments.title
        metadata = SubmissionMetadata.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as error:
        raise SystemExit(f"Файл геопривязки {arguments.metadata} не принят:\n{error}")

    try:
        rows = read_submission(arguments.submission.read_bytes())
    except ValueError as error:
        raise SystemExit(f"Таблица {arguments.submission} не принята:\n{error}")

    try:
        data = convert_submission(rows, metadata)
    except ValueError as error:
        raise SystemExit(f"Соединить таблицу и геопривязку не удалось:\n{error}")

    save_json("datasets", arguments.name, data)
    shown = data["metadata"].get("chips_shown", len(metadata.chips))
    skipped = data["metadata"].get("chips_skipped", 0)

    print(f"Набор «{metadata.title}» помещён в {data_root() / 'datasets' / arguments.name}.json")
    print(f"  фрагментов показано: {shown}"
          + (f", пропущено без геопривязки: {skipped}" if skipped else ""))
    summarise(data)
    print("\nЗапустите сервис и выберите набор в списке слева:")
    print("  .venv\\Scripts\\python.exe run.py")


if __name__ == "__main__":
    main()

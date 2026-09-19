"""Сформировать отдельную геопривязку для показа submission.csv в веб-сервисе."""

import argparse
import csv
import json
from pathlib import Path

import rasterio

from models.active_fire.features import resolve_chip_paths as af_paths


def bs_post_path(root: Path, chip_id: str) -> Path:
    path = root / "sentinel2_post" / f"{chip_id}_Sentinel-2_post.tif"
    if not path.is_file():
        raise FileNotFoundError(f"{chip_id}: не найден {path}")
    return path


def read_ids(sample: Path) -> list[str]:
    with sample.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        return list(dict.fromkeys((row["chip_id"] or "").strip() for row in rows))


def read_dates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {(row["chip_id"] or "").strip(): row for row in csv.DictReader(stream)}


def raster_metadata(path: Path) -> tuple[str, list[float]]:
    with rasterio.open(path) as source:
        if source.crs is None:
            raise ValueError(f"{path}: отсутствует CRS")
        return source.crs.to_string(), list(source.transform)[:6]


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать JSON геопривязки для submission.csv")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dates", type=Path, required=True,
                        help="CSV: chip_id,acquisition_datetime,date_pre,date_post")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/submission/web_metadata.json"))
    parser.add_argument("--title", default="Результаты моделей")
    parser.add_argument("--source", default="AF + BS inference")
    parser.add_argument("--af-version", default="af-model")
    parser.add_argument("--bs-version", default="bs-model")
    args = parser.parse_args()

    sample = args.sample or args.data_dir / "sample_submission.csv"
    dates = read_dates(args.dates)
    chips = {}
    for chip_id in read_ids(sample):
        if chip_id not in dates:
            raise ValueError(f"{chip_id}: нет строки в {args.dates}")
        row = dates[chip_id]
        if chip_id.startswith("AF_"):
            root = args.data_dir / "af" if (args.data_dir / "af").is_dir() else args.data_dir
            raster = af_paths(root / chip_id)[0]
            timestamp = (row.get("acquisition_datetime") or "").strip()
            if not timestamp:
                raise ValueError(f"{chip_id}: не заполнен acquisition_datetime")
            crs, transform = raster_metadata(raster)
            chips[chip_id] = {
                "kind": "af", "observation_id": chip_id, "crs": crs,
                "transform": transform, "model_version": args.af_version,
                "acquisition_datetime": timestamp,
            }
        elif chip_id.startswith("BS_"):
            root = args.data_dir / "bs" if (args.data_dir / "bs").is_dir() else args.data_dir
            raster = bs_post_path(root, chip_id)
            date_pre = (row.get("date_pre") or "").strip()
            date_post = (row.get("date_post") or "").strip()
            if not date_pre or not date_post:
                raise ValueError(f"{chip_id}: не заполнены date_pre и date_post")
            crs, transform = raster_metadata(raster)
            chips[chip_id] = {
                "kind": "bs", "observation_id": chip_id, "crs": crs,
                "transform": transform, "model_version": args.bs_version,
                "date_pre": date_pre, "date_post": date_post,
            }
        else:
            raise ValueError(f"Неизвестный префикс chip_id: {chip_id}")

    payload = {"title": args.title, "data_mode": "real", "source": args.source, "chips": chips}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Создан {args.output}: {len(chips)} чипов")


if __name__ == "__main__":
    main()

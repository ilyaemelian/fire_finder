import csv
import io
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import Polygon, Point, mapping, shape
from shapely.ops import transform as transform_geometry

from models.common.rle import decode_rle
from models.common.submission import validate_rows
from src.service.schemas import Metadata, SubmissionMetadata

MAX_FILE = 4 * 1024 * 1024
MAX_SUBMISSION = 64 * 1024 * 1024


def read_mask(raw: bytes, kind: str) -> np.ndarray:
    if len(raw) > MAX_FILE:
        raise ValueError("Файл маски превышает 4 МБ")
    try:
        mask = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise ValueError("Маска должна быть файлом .npy без объектов Python") from exc
    expected = (256, 256) if kind == "af" else (512, 512)
    classes = [0, 1] if kind == "af" else [0, 1, 2, 3]
    if not isinstance(mask, np.ndarray) or mask.shape != expected or mask.dtype != np.uint8:
        raise ValueError(f"{kind}: требуется uint8 {expected}")
    if not np.isin(mask, classes).all():
        raise ValueError(f"{kind}: допустимы только классы {classes}")
    return mask


def read_submission(raw: bytes) -> list[dict]:
    """Прочитать конкурсный CSV без изменения его схемы."""
    if len(raw) > MAX_SUBMISSION:
        raise ValueError("submission.csv превышает 64 МБ")
    try:
        csv.field_size_limit(MAX_SUBMISSION)
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        if reader.fieldnames != ["chip_id", "class_id", "rle"]:
            raise ValueError("Ожидаются ровно столбцы chip_id,class_id,rle")
        rows = [{
            "chip_id": (row["chip_id"] or "").strip(),
            "class_id": int(row["class_id"]),
            "rle": (row["rle"] or "").strip(),
        } for row in reader]
    except (UnicodeDecodeError, csv.Error, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("Ожидаются"):
            raise
        raise ValueError(f"Некорректный submission.csv: {exc}") from exc
    validate_rows(rows)
    return rows


def collection(features):
    return {"type": "FeatureCollection", "features": features}


def _convert_raster(kind, mask, meta):
    observations, points, polygons = [], [], []
    affine = Affine(*meta.transform)
    project = Transformer.from_crs(meta.crs, 4326, always_xy=True).transform
    height, width = mask.shape
    footprint = transform_geometry(
        project,
        Polygon([affine @ point for point in [(0, 0), (width, 0), (width, height), (0, height)]]),
    )
    bounds = footprint.bounds
    if (
        not footprint.is_valid
        or footprint.is_empty
        or not np.isfinite(bounds).all()
        or not (-180 <= bounds[0] < bounds[2] <= 180 and -85 <= bounds[1] < bounds[3] <= 85)
    ):
        raise ValueError("Геопривязка выходит за поддерживаемые координаты WGS84")
    props = meta.model_dump(mode="json", exclude_none=True) | {"kind": kind}
    observations.append({"type": "Feature", "geometry": mapping(footprint), "properties": props})
    if kind == "af":
        for row, col in np.argwhere(mask == 1):
            coordinates = project(*(affine @ (float(col) + .5, float(row) + .5)))
            points.append({
                "type": "Feature",
                "geometry": mapping(Point(coordinates)),
                "properties": props | {"detection_id": f"{meta.observation_id}-{row}-{col}"},
            })
    else:
        for index, (geometry, value) in enumerate(shapes(mask, mask=mask > 0, transform=affine)):
            projected = transform_geometry(project, shape(geometry))
            polygons.append({
                "type": "Feature",
                "geometry": mapping(projected),
                "properties": props | {
                    "severity": int(value),
                    "contour_id": f"{meta.observation_id}-{index}",
                },
            })
    return observations, points, polygons


def convert_masks(af, bs, metadata: Metadata):
    observations, points, polygons = [], [], []
    for kind, mask, meta in [
        ("af", af, metadata.active_fire),
        ("bs", bs, metadata.burn_severity),
    ]:
        obs, detected, burned = _convert_raster(kind, mask, meta)
        observations.extend(obs)
        points.extend(detected)
        polygons.extend(burned)
    return {
        "metadata": metadata.model_dump(mode="json"),
        "observations": collection(observations),
        "active_fires": collection(points),
        "burned_areas": collection(polygons),
    }


def convert_submission(rows: list[dict], metadata: SubmissionMetadata):
    """Декодировать неизменённый submission.csv с отдельной геопривязкой."""
    grouped: dict[str, dict[int, str]] = {}
    for row in rows:
        grouped.setdefault(str(row["chip_id"]), {})[int(row["class_id"])] = str(row["rle"])
    if set(grouped) != set(metadata.chips):
        missing = sorted(set(grouped) - set(metadata.chips))[:5]
        extra = sorted(set(metadata.chips) - set(grouped))[:5]
        raise ValueError(
            f"chip_id в CSV и JSON не совпадают; без метаданных: {missing}, без строк CSV: {extra}"
        )

    observations, points, polygons = [], [], []
    for chip_id, classes in grouped.items():
        meta = metadata.chips[chip_id]
        if meta.kind == "af":
            if set(classes) != {1}:
                raise ValueError(f"{chip_id}: ожидается одна строка класса 1")
            mask = decode_rle(classes[1], (256, 256)).astype(np.uint8)
        else:
            if set(classes) != {1, 2, 3}:
                raise ValueError(f"{chip_id}: ожидаются строки классов 1, 2 и 3")
            mask = np.zeros((512, 512), np.uint8)
            for class_id in (1, 2, 3):
                binary = decode_rle(classes[class_id], mask.shape).astype(bool)
                if np.any(mask[binary]):
                    raise ValueError(f"{chip_id}: маски классов пересекаются")
                mask[binary] = class_id
        obs, detected, burned = _convert_raster(meta.kind, mask, meta)
        observations.extend(obs)
        points.extend(detected)
        polygons.extend(burned)

    top = metadata.model_dump(mode="json", exclude={"chips"})
    return {
        "metadata": top,
        "observations": collection(observations),
        "active_fires": collection(points),
        "burned_areas": collection(polygons),
    }

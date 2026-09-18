import io
import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.features import shapes
from shapely.geometry import Polygon, Point, mapping, shape
from shapely.ops import transform as transform_geometry
from src.service.schemas import Metadata

MAX_FILE = 4 * 1024 * 1024


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


def collection(features):
    return {"type": "FeatureCollection", "features": features}


def convert_masks(af, bs, metadata: Metadata):
    observations, points, polygons = [], [], []
    for kind, mask, meta in [("af", af, metadata.active_fire), ("bs", bs, metadata.burn_severity)]:
        affine = Affine(*meta.transform)
        project = Transformer.from_crs(meta.crs, 4326, always_xy=True).transform
        height, width = mask.shape
        footprint = transform_geometry(project, Polygon([affine @ p for p in [(0,0),(width,0),(width,height),(0,height)]]))
        bounds = footprint.bounds
        if not footprint.is_valid or footprint.is_empty or not np.isfinite(bounds).all() or not (-180 <= bounds[0] < bounds[2] <= 180 and -85 <= bounds[1] < bounds[3] <= 85):
            raise ValueError("Геопривязка выходит за поддерживаемые координаты WGS84")
        props = meta.model_dump(mode="json") | {"kind": kind}
        observations.append({"type": "Feature", "geometry": mapping(footprint), "properties": props})
        if kind == "af":
            for row, col in np.argwhere(mask == 1):
                xy = project(*(affine @ (float(col)+.5, float(row)+.5)))
                points.append({"type": "Feature", "geometry": mapping(Point(xy)), "properties": props | {"detection_id": f"{meta.observation_id}-{row}-{col}"}})
        else:
            for i, (geom, value) in enumerate(shapes(mask, mask=mask > 0, transform=affine)):
                projected = transform_geometry(project, shape(geom))
                polygons.append({"type": "Feature", "geometry": mapping(projected), "properties": props | {"severity": int(value), "contour_id": f"{meta.observation_id}-{i}"}})
    return {"metadata": metadata.model_dump(mode="json"), "observations": collection(observations), "active_fires": collection(points), "burned_areas": collection(polygons)}

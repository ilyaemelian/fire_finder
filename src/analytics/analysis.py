from datetime import datetime, timezone
from shapely.geometry import shape, mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer
from src.analytics.masks import collection

TO_AREA = Transformer.from_crs(4326, 6933, always_xy=True).transform


def area_ha(geometry):
    return transform(TO_AREA, geometry).area / 10000


def observation_date(props):
    if props["kind"] == "af":
        return datetime.fromisoformat(props["acquisition_datetime"].replace("Z", "+00:00")).astimezone(timezone.utc).date()
    return datetime.fromisoformat(props["date_post"]).date()


def analyze(dataset, query):
    region = shape(query.geometry)
    def relevant(f):
        return query.date_from <= observation_date(f["properties"]) <= query.date_to
    coverage = {}
    for kind in ["af", "bs"]:
        footprints = [shape(f["geometry"]).intersection(region) for f in dataset["observations"]["features"] if f["properties"]["kind"] == kind and relevant(f)]
        footprints = [g for g in footprints if g.area > 0]
        covered = unary_union(footprints)
        ratio = min(1.0, area_ha(covered) / area_ha(region)) if footprints else 0
        coverage[kind] = {"has_observations": bool(footprints), "fraction": ratio, "status": "none" if not footprints else "full" if ratio > .999 else "partial"}
    points = [f for f in dataset["active_fires"]["features"] if relevant(f) and region.covers(shape(f["geometry"]))]
    polygons = []
    areas = {str(k): 0.0 for k in [1,2,3]}
    for feature in dataset["burned_areas"]["features"]:
        if not relevant(feature):
            continue
        clipped = shape(feature["geometry"]).intersection(region)
        if clipped.is_empty or clipped.area == 0:
            continue
        if clipped.geom_type == "GeometryCollection":
            clipped = unary_union([g for g in clipped.geoms if g.geom_type in ("Polygon", "MultiPolygon")])
        if clipped.is_empty:
            continue
        area = area_ha(clipped)
        props = feature["properties"] | {"area_ha": area}
        areas[str(props["severity"])] += area
        polygons.append({"type": "Feature", "geometry": mapping(clipped), "properties": props})
    return {"query": query.model_dump(mode="json"), "data_mode": dataset["metadata"]["data_mode"], "source": dataset["metadata"]["source"], "coverage": coverage, "summary": {"total_burned_area_ha": sum(areas.values()), "area_by_severity_ha": areas, "active_detection_count": len(points), "burned_contour_count": len(polygons)}, "active_fires": collection(points), "burned_areas": collection(polygons)}

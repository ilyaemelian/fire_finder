import json
import os
import uuid
from pathlib import Path
from shapely.geometry import shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]


def data_root():
    root = Path(os.environ.get("FIRE_DATA_DIR", str(ROOT / "data")))
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_json(folder, ident, data):
    target = data_root() / folder
    target.mkdir(exist_ok=True)
    temp = target / f"{ident}.{uuid.uuid4().hex}.tmp"
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(target / f"{ident}.json")


def load_json(folder, ident):
    path = data_root() / folder / f"{ident}.json"
    if not path.is_file():
        raise FileNotFoundError(ident)
    return json.loads(path.read_text(encoding="utf-8"))


def catalog():
    out = []
    for path in sorted((data_root() / "datasets").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["metadata"]
        features = data["observations"]["features"]
        bs_features = [feature for feature in features if feature["properties"]["kind"] == "bs"]
        footprint = unary_union([shape(feature["geometry"]) for feature in (bs_features or features)])
        dates = []
        for feature in features:
            props = feature["properties"]
            dates.append((props.get("acquisition_datetime") or props.get("date_post"))[:10])
        out.append({
            "id": path.stem,
            "title": meta["title"],
            "data_mode": meta["data_mode"],
            "source": meta["source"],
            "bounds": list(footprint.bounds),
            "date_from": min(dates),
            "date_to": max(dates),
            "observations": data["observations"],
        })
    return out

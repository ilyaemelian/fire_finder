import json
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.analytics.masks import (
    read_mask, convert_masks, read_submission, convert_submission, MAX_FILE, MAX_SUBMISSION,
)
from src.analytics.analysis import analyze
from src.service.schemas import Metadata, SubmissionMetadata, AnalysisQuery, ExportQuery
from src.service.repository import save_json, load_json, catalog, ROOT

app = FastAPI(title="Мониторинг природных пожаров", version="0.2.0")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/observations")
def observations():
    return {"datasets": catalog()}


@app.post("/api/results", status_code=201)
async def upload_results(active_fire: UploadFile = File(...), burn_severity: UploadFile = File(...), metadata: UploadFile = File(...)):
    try:
        raw = await metadata.read(65537)
        if len(raw) > 65536:
            raise ValueError("Метаданные превышают 64 КБ")
        meta = Metadata.model_validate_json(raw)
        af = read_mask(await active_fire.read(MAX_FILE + 1), "af")
        bs = read_mask(await burn_severity.read(MAX_FILE + 1), "bs")
        data = convert_masks(af, bs, meta)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    ident = uuid.uuid4().hex
    save_json("datasets", ident, data)
    return {"dataset_id": ident}


@app.post("/api/results/submission", status_code=201)
async def upload_submission(submission: UploadFile = File(...), metadata: UploadFile = File(...)):
    """Принять конкурсный CSV без изменения схемы и отдельную геопривязку."""
    try:
        raw_meta = await metadata.read(4 * 1024 * 1024 + 1)
        if len(raw_meta) > 4 * 1024 * 1024:
            raise ValueError("Метаданные submission превышают 4 МБ")
        meta = SubmissionMetadata.model_validate_json(raw_meta)
        rows = read_submission(await submission.read(MAX_SUBMISSION + 1))
        data = convert_submission(rows, meta)
    except (ValueError, ValidationError) as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    ident = uuid.uuid4().hex
    save_json("datasets", ident, data)
    return {"dataset_id": ident}


@app.post("/api/analysis")
def analysis(query: AnalysisQuery):
    try:
        dataset = load_json("datasets", query.dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Набор не найден") from exc
    result = analyze(dataset, query)
    result["result_id"] = uuid.uuid4().hex
    save_json("results", result["result_id"], result)
    return result


def exported(query, kind):
    try:
        result = load_json("results", query.result_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Результат не найден. Повторите расчёт") from exc
    if kind == "contours":
        # Система координат объявляется явно: геометрия в WGS84, площадь
        # посчитана в равновеликой проекции EPSG:6933.
        data = result["burned_areas"] | {
            "crs": {"type": "name",
                    "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "area_crs": "EPSG:6933",
            "query": result["query"],
            "data_mode": result["data_mode"],
            "source": result["source"],
        }
    else:
        data = {key: value for key, value in result.items()
                if key not in ("active_fires", "burned_areas")}
        data["crs"] = "EPSG:4326"
        data["area_crs"] = "EPSG:6933"
    return Response(json.dumps(data, ensure_ascii=False, allow_nan=False), media_type="application/geo+json" if kind == "contours" else "application/json", headers={"Content-Disposition": f'attachment; filename="{kind}-{query.result_id}.json"'})


@app.post("/api/export/contours")
def contours(query: ExportQuery):
    return exported(query, "contours")


@app.post("/api/export/summary")
def summary(query: ExportQuery):
    return exported(query, "summary")


DIST = ROOT / "frontend" / "dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    if not (DIST / "index.html").exists():
        raise HTTPException(503, "Соберите интерфейс: npm --prefix frontend run build")
    return FileResponse(DIST / "index.html")

import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from src.service.main import app
from src.service.repository import load_json, data_root
from scripts.seed_demo import main as seed
from src.analytics.analysis import area_ha


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRE_DATA_DIR", str(tmp_path))
    seed()
    with TestClient(app) as client:
        yield client


def query(client):
    dataset = client.get("/api/observations").json()["datasets"][0]
    return {"dataset_id": "demo", "geometry": mapping(box(*dataset["bounds"])), "date_from": "2025-09-01", "date_to": "2025-09-30"}


def test_analysis_exports_and_partition(client):
    response=client.post("/api/analysis",json=query(client))
    assert response.status_code == 200
    result=response.json()
    summary=result["summary"]
    assert summary["active_detection_count"] == 9
    assert summary["total_burned_area_ha"] > 0
    assert sum(summary["area_by_severity_ha"].values()) == pytest.approx(summary["total_burned_area_ha"])
    geoms=[shape(f["geometry"]) for f in result["burned_areas"]["features"]]
    assert area_ha(unary_union(geoms)) == pytest.approx(summary["total_burned_area_ha"], rel=1e-7)
    for kind in ["summary", "contours"]:
        download=client.post("/api/export/"+kind,json={"result_id":result["result_id"]})
        assert download.status_code == 200
        assert "attachment" in download.headers["content-disposition"]
        if kind == "summary": assert download.json()["summary"] == summary
        else: assert download.json()["features"] == result["burned_areas"]["features"]


def test_clipping_and_empty_period(client):
    q=query(client)
    full=client.post("/api/analysis",json=q).json()
    w,s,e,n=shape(q["geometry"]).bounds
    q["geometry"]=mapping(box(w,s,(w+e)/2,n))
    half=client.post("/api/analysis",json=q).json()
    assert 0 < half["summary"]["total_burned_area_ha"] < full["summary"]["total_burned_area_ha"]
    for f in half["burned_areas"]["features"]:
        assert shape(q["geometry"]).buffer(1e-10).covers(shape(f["geometry"]))
    q["date_from"]=q["date_to"]="2024-01-01"
    empty=client.post("/api/analysis",json=q).json()
    assert empty["summary"]["total_burned_area_ha"] == 0
    assert not empty["coverage"]["bs"]["has_observations"]


def upload_files(af=None,bs=None,meta=None):
    folder=data_root()/"demo-input"
    def data(name,arr):
        if arr is None:return (folder/name).read_bytes()
        stream=io.BytesIO();np.save(stream,arr);return stream.getvalue()
    return {"active_fire":("active_fire.npy",data("active_fire.npy",af),"application/octet-stream"),"burn_severity":("burn_severity.npy",data("burn_severity.npy",bs),"application/octet-stream"),"metadata":("metadata.json",json.dumps(meta).encode() if meta else (folder/"metadata.json").read_bytes(),"application/json")}


def test_upload_roundtrip_and_zero_observations(client):
    files=upload_files(af=np.zeros((256,256),dtype=np.uint8),bs=np.zeros((512,512),dtype=np.uint8))
    response=client.post("/api/results",files=files)
    assert response.status_code == 201
    q=query(client);q["dataset_id"]=response.json()["dataset_id"]
    result=client.post("/api/analysis",json=q).json()
    assert result["coverage"]["bs"]["has_observations"]
    assert result["summary"]["total_burned_area_ha"] == 0
    assert result["summary"]["active_detection_count"] == 0


@pytest.mark.parametrize("kind,mask",[("af",np.zeros((20,20),dtype=np.uint8)),("af",np.full((256,256),2,dtype=np.uint8)),("bs",np.full((512,512),4,dtype=np.uint8)),("bs",np.zeros((512,512),dtype=float))])
def test_reject_invalid_masks(client,kind,mask):
    assert client.post("/api/results",files=upload_files(**{kind:mask})).status_code == 422


def test_invalid_metadata_query_and_paths(client):
    metadata=json.loads((data_root()/"demo-input"/"metadata.json").read_text(encoding="utf-8"))
    metadata["burn_severity"]["transform"]=[0,0,0,0,0,0]
    assert client.post("/api/results",files=upload_files(meta=metadata)).status_code == 422
    q=query(client);q["date_from"]="2026-01-01"
    assert client.post("/api/analysis",json=q).status_code == 422
    q=query(client);q["dataset_id"]="../other"
    assert client.post("/api/analysis",json=q).status_code == 422
    q=query(client);q["geometry"]={"type":"Point","coordinates":[39,46]}
    assert client.post("/api/analysis",json=q).status_code == 422
    assert client.get("/api/missing").status_code == 404

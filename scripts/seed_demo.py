"""Explicit synthetic data generator, never a substitute for ML inference."""
import io
import json
import numpy as np
from src.service.schemas import Metadata
from src.analytics.masks import convert_masks
from src.service.repository import save_json, data_root


def main():
    af = np.zeros((256,256), dtype=np.uint8)
    for row,col in [(125,123),(125,124),(126,125),(127,124),(131,132),(132,132),(132,133),(128,136),(129,136)]:
        af[row,col] = 1
    y,x = np.ogrid[:512,:512]
    bs = np.zeros((512,512), dtype=np.uint8)
    for cx,cy,rx,ry in [(165,185,87,65),(345,330,58,80),(395,160,34,51)]:
        distance = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
        bs[distance < 1] = 1
        bs[distance < .57] = 2
        bs[distance < .20] = 3
    meta = Metadata.model_validate({"title": "Учебный участок · сентябрь 2025", "data_mode": "synthetic", "source": "Искусственные маски для проверки интерфейса. Не сведения о реальных пожарах.", "active_fire": {"observation_id": "demo-af", "crs": "EPSG:32637", "transform": [375,0,452000,0,-375,5148000], "model_version": "stub-1", "acquisition_datetime": "2025-09-12T10:30:00Z"}, "burn_severity": {"observation_id": "demo-bs", "crs": "EPSG:32637", "transform": [20,0,494880,0,-20,5105120], "model_version": "stub-1", "date_pre": "2025-08-20", "date_post": "2025-09-15"}})
    save_json("datasets", "demo", convert_masks(af, bs, meta))
    folder = data_root() / "demo-input"
    folder.mkdir(exist_ok=True)
    np.save(folder / "active_fire.npy", af)
    np.save(folder / "burn_severity.npy", bs)
    (folder / "metadata.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    print(f"Synthetic demo created: {folder}")


if __name__ == "__main__":
    main()

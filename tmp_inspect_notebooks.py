import json
from pathlib import Path

paths = [
    Path(r"C:\Users\Lenovo\Downloads\research_bs.ipynb"),
    Path(r"C:\Users\Lenovo\Downloads\Telegram Desktop\unet_train.ipynb"),
    Path(r"models\active_fire\notebooks\pipeline.ipynb"),
]
needles = (
    "XGBClassifier", "pass1", "pass2", "GroupKFold", "fold", "save_model",
    "features.json", "config.json", "train", "validation", "score", "baseline",
)
for path in paths:
    print(f"\n### {path}")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    print("cells", len(notebook.get("cells", [])))
    for index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        first = next((line.strip() for line in source.splitlines() if line.strip()), "")
        hits = [needle for needle in needles if needle.lower() in source.lower()]
        print(index, cell.get("cell_type"), first[:140], "hits=" + ",".join(hits))
        if path.name == "research_bs.ipynb" and hits:
            print(source[:5000])

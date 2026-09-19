import json
import shutil
from pathlib import Path


module = Path("models/burn_severity")
research_source = Path(r"C:\Users\Lenovo\Downloads\research_bs.ipynb")
research_target = module / "research_bs.ipynb"
shutil.copyfile(research_source, research_target)

research = json.loads(research_target.read_text(encoding="utf-8"))
cell = "".join(research["cells"][1]["source"])
cell = cell.replace(
    'PROJECT = Path("/home/admin/PycharmProjects/space_hack/train")\nROOT = PROJECT / "bs"\nFIRMS_DIR = PROJECT\n',
    'PROJECT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "train" / "bs").is_dir()), None)\n'
    'if PROJECT is None:\n'
    '    raise FileNotFoundError("Сначала выполните prepare_data.py --scope train")\n'
    'ROOT = PROJECT / "data" / "train" / "bs"\n'
    'FIRMS_DIR = ROOT\n',
)
cell = cell.replace('ROOT / "aux" /', '(ROOT / ("aux" if (ROOT / "aux").is_dir() else "aux_data")) /')
research["cells"][1]["source"] = cell.splitlines(keepends=True)
for notebook_cell in research["cells"]:
    source = "".join(notebook_cell.get("source", []))
    source = source.replace('ROOT / "aux" /', '(ROOT / ("aux" if (ROOT / "aux").is_dir() else "aux_data")) /')
    notebook_cell["source"] = source.splitlines(keepends=True)
research_target.write_text(json.dumps(research, ensure_ascii=False, indent=1), encoding="utf-8")

training_path = module / "unet_train.ipynb"
training = json.loads(training_path.read_text(encoding="utf-8"))
setup = "".join(training["cells"][0]["source"])
setup = setup.replace(
    'PROJECT = Path("../..")\nTRAIN = PROJECT / "BS" / "bs"\nOUT = Path("weights")\n',
    'PROJECT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "train" / "bs").is_dir()), None)\n'
    'if PROJECT is None:\n'
    '    raise FileNotFoundError("Сначала выполните prepare_data.py --scope train")\n'
    'TRAIN = PROJECT / "data" / "train" / "bs"\n'
    'OUT = PROJECT / "outputs" / "training" / "burn_severity" / "notebook_weights"\n',
)
training["cells"][0]["source"] = setup.splitlines(keepends=True)
reader = "".join(training["cells"][1]["source"])
reader = reader.replace(
    'def read(folder, cid, suffix):\n    with rasterio.open(TRAIN / folder / f"{cid}_{suffix}.tif") as s:\n',
    'def read(folder, cid, suffix):\n'
    '    if folder == "aux" and not (TRAIN / folder).is_dir():\n'
    '        folder = "aux_data"\n'
    '    with rasterio.open(TRAIN / folder / f"{cid}_{suffix}.tif") as s:\n',
)
training["cells"][1]["source"] = reader.splitlines(keepends=True)
training_path.write_text(json.dumps(training, ensure_ascii=False, indent=1), encoding="utf-8")

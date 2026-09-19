"""Интеграционный запуск финального BS-инференса, переданного автором модели."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from models.common.submission import read_rows, validate_rows
from scripts.collect_results import fill_template
from scripts.pipeline_common import describe_test_dir, require, step, verify_weights

ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "models" / "burn_severity"
DEFAULT_MODELS = MODULE_ROOT / "models"
DEFAULT_RESULTS = MODULE_ROOT / "results" / "bs_submission.csv"
DEFAULT_SUBMISSION = ROOT / "outputs" / "submission" / "submission_bs.csv"

PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "rasterio": "rasterio",
    "xgboost": "xgboost",
    "torch": "torch",
    "segmentation_models_pytorch": "segmentation-models-pytorch",
}


def bs_pairs(sample: Path) -> set[tuple[str, int]]:
    return {
        (str(row["chip_id"]), int(row["class_id"]))
        for row in read_rows(sample)
        if str(row["chip_id"]).startswith("BS_")
    }


def weight_files(models: Path) -> dict[Path, str]:
    manifest = json.loads((models / "manifest.json").read_text(encoding="utf-8"))
    files = {models / name: digest for name, digest in manifest["pass1"]["sha256"].items()}
    files[models / manifest["pass2"]["weights"]] = manifest["pass2"]["sha256"]
    files[models / manifest["unet"]["weights"]] = manifest["unet"]["sha256"]
    files[models / manifest["unet"]["normalisation"]] = manifest["unet"]["norm_sha256"]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить финальный BS: XGBoost + U-Net")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--models", "--artifacts", dest="models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--work", type=Path, default=MODULE_ROOT / "work")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--unet-weight", type=float, default=0.45)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    sample = args.sample or args.data_dir / "sample_submission.csv"
    expected = bs_pairs(sample)

    step(1, "Среда")
    require(PACKAGES, "models/burn_severity/requirements.txt")

    step(2, "Вход")
    describe_test_dir(args.data_dir, sample, "bs", len(expected) // 3)

    step(3, "Авторские веса")
    verify_weights(weight_files(args.models), checksums=not args.skip_checksums)
    print("    финальная схема автора: 5 XGBoost -> XGBoost с контекстом + U-Net ResNet34")

    step(4, "Авторский inference_new.py")
    if not args.reuse_results:
        command = [
            sys.executable,
            "-m", "models.burn_severity.inference_new",
            "--data-dir", str(args.data_dir),
            "--output", str(args.results),
            "--models", str(args.models),
            "--work", str(args.work),
            "--device", args.device,
            "--unet-weight", str(args.unet_weight),
            "--threshold", str(args.threshold),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    rows = read_rows(args.results)
    stats = validate_rows(rows, kind="bs", expected_pairs=expected)
    print(f"    проверено: {stats['rows']} строк, {stats['nonempty']} непустых масок")

    step(5, "Файл по официальному шаблону")
    final = fill_template(sample, rows, args.output)
    print(f"    {args.output}: {final['rows']} строк, {final['nonempty']} непустых масок")
    print(f"\nBS готов за {time.perf_counter() - started:.0f} с")


if __name__ == "__main__":
    main()

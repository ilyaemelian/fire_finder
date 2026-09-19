"""Единая команда полного обучения AF и BS на заранее подготовленном train."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment_python(name: str) -> Path:
    candidates = [ROOT / name / "Scripts" / "python.exe", ROOT / name / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Среда {name} не найдена. Запустите python bootstrap.py")


def require_train_data(path: Path) -> Path:
    path = path.resolve()
    required = [path / "af" / "masks", path / "bs" / "masks"]
    missing = [item for item in required if not item.is_dir()]
    if missing:
        raise SystemExit(
            "Обучающие данные не подготовлены. Сначала выполните:\n"
            "  .venv\\Scripts\\python.exe prepare_data.py --scope train\n"
            f"Не найдены: {', '.join(map(str, missing))}"
        )
    return path


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить AF и BS")
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--af-python", type=Path)
    parser.add_argument("--bs-python", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="быстрая проверка всех стадий на пяти чипах")
    args = parser.parse_args()

    data_dir = require_train_data(args.data_dir)
    af_python = args.af_python or environment_python(".venv-af")
    bs_python = args.bs_python or environment_python(".venv-bs")
    af_output = args.output_dir / "active_fire"
    bs_output = args.output_dir / "burn_severity" / "models"

    af = [
        str(af_python), "-m", "models.active_fire.train",
        "--data-dir", str(data_dir), "--output-dir", str(af_output),
        "--seed", str(args.seed),
    ]
    bs = [
        str(bs_python), "-m", "models.burn_severity.train",
        "--data-dir", str(data_dir), "--output-dir", str(bs_output),
        "--seed", str(args.seed), "--device", args.device,
    ]
    if args.smoke:
        af += ["--max-chips", "5", "--folds", "2", "--estimators", "5",
               "--negatives-per-chip", "100"]
        bs += [
            "--max-chips", "5", "--folds", "2", "--estimators", "2",
            "--samples-per-chip", "500", "--epochs", "1", "--epoch-schedule", "2",
            "--crop", "64", "--crops-per-chip", "1", "--batch", "2",
            "--encoder-weights", "none",
        ]

    run(af)
    run(bs)
    af_manifest = json.loads((af_output / "artifacts" / "manifest.json").read_text("utf-8"))
    bs_manifest = json.loads((bs_output / "manifest.json").read_text("utf-8"))
    print("\nПолное обучение завершено")
    print(f"  AF: {af_manifest['algorithm']} · чипов={af_manifest['training_chips']} · "
          f"деревьев={af_manifest['n_estimators']} · CV F1={af_manifest['f1_cross_validation']:.4f}")
    print(f"  BS: {bs_manifest['algorithm']} · чипов={bs_manifest['training_chips']} · "
          f"фолдов={len(bs_manifest['pass1']['weights'])} · "
          f"U-Net эпох={bs_manifest['unet']['epochs']} · "
          f"loss={bs_manifest['unet']['final_loss']:.5f}")
    print(f"  Артефакты: {args.output_dir}")


if __name__ == "__main__":
    main()

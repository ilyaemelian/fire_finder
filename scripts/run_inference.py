"""Единый инференс AF и BS с формированием неизменённого конкурсного submission.csv."""

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.collect_results import collect

ROOT = Path(__file__).resolve().parents[1]


def environment_python(name: str) -> Path:
    folder = ROOT / name
    candidates = [folder / "Scripts" / "python.exe", folder / "bin" / "python"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Среда {name} не найдена. Создайте её по инструкции README.md")


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Рассчитать обе модели и собрать submission.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data/test"),
                        help="заранее подготовленный каталог test")
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/submission/submission.csv"))
    parser.add_argument("--af-model", type=Path, default=Path("models/active_fire/artifacts/model.pkl"))
    parser.add_argument(
        "--bs-models", "--bs-artifacts", dest="bs_models", type=Path,
        default=Path("models/burn_severity/models"),
    )
    parser.add_argument("--af-python", type=Path)
    parser.add_argument("--bs-python", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unet-weight", type=float)
    parser.add_argument("--skip-checksums", action="store_true")
    args = parser.parse_args()

    af_python = args.af_python or environment_python(".venv-af")
    bs_python = args.bs_python or environment_python(".venv-bs")
    data_dir = args.data_dir.resolve()
    required = [data_dir / "sample_submission.csv", data_dir / "af", data_dir / "bs"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Тестовые данные не подготовлены. Загрузка намеренно отделена от замера "
            "инференса. Сначала выполните:\n"
            "  .venv\\Scripts\\python.exe prepare_data.py --scope test\n"
            f"Не найдены: {', '.join(map(str, missing))}"
        )
    sample = args.sample or data_dir / "sample_submission.csv"
    af_result = Path("models/active_fire/results/af_submission.csv")
    bs_result = Path("models/burn_severity/results/bs_submission.csv")

    common = ["--data-dir", str(data_dir), "--sample", str(sample)]
    af = [str(af_python), "-m", "scripts.run_af_pipeline", *common,
          "--results", str(af_result), "--output", "outputs/submission/submission_af.csv",
          "--model", str(args.af_model)]
    bs = [str(bs_python), "-m", "scripts.run_bs_pipeline", *common,
          "--results", str(bs_result), "--output", "outputs/submission/submission_bs.csv",
          "--device", args.device, "--models", str(args.bs_models)]
    if args.unet_weight is not None:
        bs += ["--unet-weight", str(args.unet_weight)]
    if args.skip_checksums:
        af.append("--skip-checksums")
        bs.append("--skip-checksums")

    run(af)
    run(bs)
    stats = collect(sample, ROOT / "models", args.output)
    print(f"\nСоздан {args.output}: {stats['rows']} строк, "
          f"{stats['nonempty']} непустых масок, {stats['pixels']} положительных пикселей")


if __name__ == "__main__":
    main()

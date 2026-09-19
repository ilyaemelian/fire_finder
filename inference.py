"""Расчёт предсказаний обеих моделей и сборка итогового файла ответа.

    python inference.py --data-dir data/test

Три шага: модуль AF, модуль BS, объединение по официальному шаблону.
Результат — `outputs/submission/submission.csv`.

Модели живут в отдельных средах, поэтому каждый шаг выполняется своим
интерпретатором: `.venv-af` для AF, `.venv-bs` для BS. Если такой среды нет,
шаг выполняется текущим интерпретатором. Имеющиеся веса не изменяются.

Полезные параметры: `--only af`, `--only bs`, `--only merge` — выполнить один
шаг; `--data-dir` — каталог теста с `sample_submission.csv`, `af/` и `bs/`.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AF_RESULT = ROOT / "models" / "active_fire" / "results" / "af_submission.csv"
BS_RESULT = ROOT / "models" / "burn_severity" / "results" / "bs_submission.csv"
SUBMISSION = ROOT / "outputs" / "submission" / "submission.csv"


def interpreter(name: str) -> Path:
    """Интерпретатор среды, если она создана, иначе текущий."""
    for relative in (f"{name}/Scripts/python.exe", f"{name}/bin/python"):
        candidate = ROOT / relative
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def run(step: str, python: Path, module: str, arguments: list[str]) -> None:
    print(f"\n[{step}]  {python.parent.parent.name if python != Path(sys.executable) else 'текущая среда'}")
    command = [str(python), "-m", module, *arguments]
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nШаг «{step}» завершился с ошибкой (код {result.returncode}).", file=sys.stderr)
        print(f"Команда: {' '.join(command)}", file=sys.stderr)
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Предсказания обеих моделей и итоговый файл ответа"
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "test",
                        help="каталог теста: sample_submission.csv, af/, bs/")
    parser.add_argument("--sample", type=Path,
                        help="шаблон, по умолчанию <data-dir>/sample_submission.csv")
    parser.add_argument("--output", type=Path, default=SUBMISSION,
                        help="итоговый файл ответа")
    parser.add_argument("--only", choices=("af", "bs", "merge"),
                        help="выполнить только один шаг")
    parser.add_argument("--device", default="auto", help="для BS: auto, cuda или cpu")
    parser.add_argument("--af-model", type=Path,
                        help="иные веса AF, например после повторного обучения")
    parser.add_argument("--bs-models", type=Path,
                        help="иной каталог весов BS, например после повторного обучения")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    sample = (args.sample or data_dir / "sample_submission.csv").resolve()
    if not sample.is_file():
        raise SystemExit(f"Не найден шаблон: {sample}")

    started = time.perf_counter()
    steps = ("af", "bs", "merge") if args.only is None else (args.only,)

    if "af" in steps:
        arguments = ["--data-dir", str(data_dir), "--sample", str(sample),
                     "--output", str(AF_RESULT)]
        if args.af_model:
            arguments += ["--model", str(args.af_model.resolve())]
        run("AF, активное горение", interpreter(".venv-af"),
            "models.active_fire.inference", arguments)

    if "bs" in steps:
        arguments = ["--data-dir", str(data_dir), "--output", str(BS_RESULT),
                     "--device", args.device, "--sample", str(sample)]
        if args.bs_models:
            arguments += ["--models", str(args.bs_models.resolve())]
        run("BS, степень поражения", interpreter(".venv-bs"),
            "models.burn_severity.inference_new", arguments)

    if "merge" in steps:
        run("Объединение по шаблону", Path(sys.executable),
            "scripts.collect_results",
            ["--sample", str(sample), "--output", str(args.output)])

    print(f"\nГотово за {time.perf_counter() - started:.0f} с. Файл ответа: {args.output}")


if __name__ == "__main__":
    main()

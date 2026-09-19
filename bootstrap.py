"""Создать три изолированные среды и установить зафиксированные зависимости."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = (
    (".venv", "requirements.txt"),
    (".venv-af", "requirements-af.txt"),
    (".venv-bs", "requirements-bs.txt"),
)


def environment_python(folder: Path) -> Path:
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def exposes_system_packages(folder: Path) -> bool:
    config = folder / "pyvenv.cfg"
    if not config.is_file():
        return False
    return any(
        line.strip().lower() == "include-system-site-packages = true"
        for line in config.read_text(encoding="utf-8").splitlines()
    )


def main() -> None:
    for name, requirements in ENVIRONMENTS:
        folder = ROOT / name
        if exposes_system_packages(folder):
            print(f"Пересоздание {name}: старая среда подключает системные пакеты", flush=True)
            shutil.rmtree(folder)
        python = environment_python(folder)
        if not python.is_file():
            print(f"Создание {name}", flush=True)
            subprocess.run([sys.executable, "-m", "venv", str(folder)], check=True)
        print(f"Установка {requirements} в {name}", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(ROOT / requirements)],
            cwd=ROOT,
            check=True,
        )
    print("\nОкружения готовы: .venv, .venv-af, .venv-bs")


if __name__ == "__main__":
    main()

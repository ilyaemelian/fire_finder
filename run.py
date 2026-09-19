"""Запуск веб-приложения одной командой.

    python run.py

Скрипт проверяет зависимости, создаёт демонстрационный набор наблюдений, если
его ещё нет, и поднимает сервер. Собранный интерфейс включён в репозиторий,
поэтому Node.js для запуска не требуется.

Параметры: --host, --port, --reload, --no-demo.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

REQUIRED = ("fastapi", "uvicorn", "numpy", "rasterio", "shapely", "pyproj", "affine")


def check_dependencies() -> None:
    import importlib.util

    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if missing:
        print(f"Не установлены библиотеки: {', '.join(missing)}", file=sys.stderr)
        print(f"Выполните: {Path(sys.executable).name} -m pip install -r requirements.txt",
              file=sys.stderr)
        raise SystemExit(1)


def ensure_demo() -> None:
    from scripts.seed_demo import main as seed
    from src.service.repository import data_root

    if (data_root() / "datasets" / "demo.json").is_file():
        print("Демонстрационный набор уже создан.")
        return
    seed()


def check_interface() -> None:
    index = ROOT / "frontend" / "dist" / "index.html"
    if index.is_file():
        return
    print("Собранный интерфейс не найден: frontend/dist/index.html", file=sys.stderr)
    print("Соберите его командами: npm --prefix frontend ci "
          "и npm --prefix frontend run build", file=sys.stderr)
    print("Маршруты HTTP будут доступны и без интерфейса.", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить веб-приложение")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="перезапуск при изменении кода")
    parser.add_argument("--no-demo", action="store_true", help="не создавать демонстрационный набор")
    args = parser.parse_args()

    check_dependencies()
    if not args.no_demo:
        ensure_demo()
    check_interface()

    import uvicorn

    print(f"\nПриложение: http://{args.host}:{args.port}")
    print(f"Описание маршрутов: http://{args.host}:{args.port}/docs")
    print("Остановка: Ctrl+C\n")
    uvicorn.run("src.service.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()

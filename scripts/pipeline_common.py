"""Общие шаги пайплайнов моделей: проверка среды, целостности весов, оформление вывода.

Этим пользуются `scripts/run_af_pipeline.py` и `scripts/run_bs_pipeline.py`.
Модели живут в разных средах — `.venv-af` и `.venv-bs`, — поэтому сами пайплайны
раздельные, а совпадают у них только проверки, собранные здесь.
"""

import hashlib
import importlib
import importlib.util
from pathlib import Path


def step(number: int, title: str) -> None:
    print(f"\n[{number}] {title}")


def require(packages: dict[str, str], requirements: str) -> None:
    """Проверить наличие библиотек и напечатать версии.

    `packages` — имя для импорта -> имя в файле зависимостей. Отсутствие хотя бы
    одной прекращает работу: лучше сказать об этом сразу, чем через час расчёта.
    """
    missing = []
    for module, package in packages.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)
            print(f"    {package:<32} нет")
        else:
            version = getattr(importlib.import_module(module), "__version__", "версия неизвестна")
            print(f"    {package:<32} {version}")
    if missing:
        raise SystemExit(
            f"Не установлены: {', '.join(missing)}. Выполните: "
            f"python -m pip install -r {requirements}"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_weights(files: dict[Path, str | None], *, checksums: bool = True) -> None:
    """Проверить, что веса на месте и, если известна контрольная сумма, не подменены."""
    absent = [path for path in files if not Path(path).is_file()]
    if absent:
        raise SystemExit(
            "Веса не найдены:\n    " + "\n    ".join(str(path) for path in absent)
        )
    wrong = []
    for path, expected in files.items():
        size = Path(path).stat().st_size / 1e6
        if not checksums or not expected:
            print(f"    {Path(path).name:<24} {size:8.1f} МБ")
            continue
        actual = sha256(path)
        ok = actual == expected.upper()
        wrong += [] if ok else [Path(path).name]
        print(f"    {Path(path).name:<24} {size:8.1f} МБ   "
              f"{'сумма совпадает' if ok else 'СУММА НЕ СОВПАДАЕТ'}")
    if wrong:
        raise SystemExit(
            f"Контрольные суммы не совпали: {', '.join(wrong)}. "
            f"Файл повреждён или заменён другой версией; сверьте с artifacts/manifest.json."
        )


def describe_test_dir(data_dir: Path, sample: Path, subdir: str, chips: int) -> None:
    root = data_dir / subdir if (data_dir / subdir).is_dir() else data_dir
    print(f"    шаблон          {sample}")
    print(f"    каталог чипов   {root}"
          f"{'' if root != data_dir else '  (раскладка без подкаталога ' + subdir + ')'}")
    print(f"    чипов в шаблоне {chips}")

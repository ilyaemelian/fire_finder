"""Загрузка и безопасная распаковка конкурсных данных с публичного Яндекс Диска."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_PUBLIC_URL = "https://disk.yandex.ru/d/-rpmevTflbXZQg"
API = "https://cloud-api.yandex.net/v1/disk/public/resources"


@dataclass(frozen=True)
class Archive:
    remote_path: str
    filename: str
    dataset: str
    size: int
    sha256: str


ARCHIVES = {
    "train": Archive(
        "/fire-train-renamed.tar", "fire-train-renamed.tar", "train", 2_305_976_320,
        "9cc2d532312dd91d24beb8c6abcfdddac0cff6ca095501be28d81f751a8f159a",
    ),
    "test": Archive(
        "/fire-test-renamed.tar", "fire-test-renamed.tar", "test", 920_483_840,
        "0431f8ae3af3a266a99074f87ca4a4caf5b6ea2e3645048a905ccac10c51674f",
    ),
}


def api_json(endpoint: str, public_url: str, **parameters) -> dict:
    query = urllib.parse.urlencode({"public_key": public_url, **parameters})
    request = urllib.request.Request(f"{API}{endpoint}?{query}", headers={"User-Agent": "Kontur/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise RuntimeError(f"Не удалось обратиться к Яндекс Диску: {exc}") from exc
            if attempt == 4:
                raise RuntimeError(f"Яндекс Диск временно ограничил запросы: {exc}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = min(15, int(retry_after)) if retry_after and retry_after.isdigit() else 2 ** attempt
            print(f"Яндекс Диск временно недоступен ({exc.code}); повтор через {delay} с", flush=True)
            time.sleep(delay)
        except Exception as exc:
            if attempt == 4:
                raise RuntimeError(f"Не удалось обратиться к Яндекс Диску: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("недостижимая ветвь")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def human_size(value: int) -> str:
    return f"{value / 1024 ** 3:.2f} ГБ" if value >= 1024 ** 3 else f"{value / 1024 ** 2:.1f} МБ"


def dataset_ready(path: Path, kind: str) -> bool:
    common = (path / "af").is_dir() and (path / "bs").is_dir()
    if kind == "test":
        return common and (path / "sample_submission.csv").is_file()
    return common and (path / "af" / "masks").is_dir() and (path / "bs" / "masks").is_dir()


def check_remote(spec: Archive, public_url: str) -> None:
    metadata = api_json("", public_url, path=spec.remote_path)
    if int(metadata.get("size", -1)) != spec.size:
        raise RuntimeError(f"Размер {spec.remote_path} изменился: ожидалось {spec.size}, получено {metadata.get('size')}")
    remote_hash = str(metadata.get("sha256") or "").lower()
    if remote_hash and remote_hash != spec.sha256:
        raise RuntimeError(f"SHA-256 {spec.remote_path} изменился. Автоматическая загрузка остановлена")


def download_archive(spec: Archive, public_url: str, downloads: Path, force: bool) -> Path:
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / spec.filename
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_file():
        if target.stat().st_size == spec.size and sha256(target) == spec.sha256:
            print(f"Данные: архив уже загружен и проверен: {target}")
            return target
        if not force:
            raise RuntimeError(f"Повреждён кэш {target}. Повторная загрузка: --force-download")
        target.unlink()
    if partial.is_file() and partial.stat().st_size > spec.size:
        partial.unlink()

    free = shutil.disk_usage(downloads.parent).free
    remaining = spec.size - (partial.stat().st_size if partial.is_file() else 0)
    if free < remaining + 512 * 1024 ** 2:
        raise RuntimeError(f"Недостаточно места: свободно {human_size(free)}, для загрузки нужно не менее {human_size(remaining)}")

    href = api_json("/download", public_url, path=spec.remote_path)["href"]
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "Kontur/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(href, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = offset > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        if offset and not resumed:
            offset = 0
        downloaded = offset
        started = time.monotonic()
        last_report = started
        print(f"Данные: загрузка {spec.filename} с {human_size(offset)} из {human_size(spec.size)}")
        with partial.open(mode) as stream:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_report >= 2:
                    speed = (downloaded - offset) / max(now - started, .001)
                    print(f"  {100 * downloaded / spec.size:5.1f}% · {human_size(downloaded)} · {human_size(int(speed))}/с", flush=True)
                    last_report = now
    if partial.stat().st_size != spec.size:
        raise RuntimeError(f"Загружено {partial.stat().st_size} байт вместо {spec.size}; повторите команду для продолжения")
    actual = sha256(partial)
    if actual != spec.sha256:
        raise RuntimeError(f"SHA-256 не совпал для {partial}; файл не будет распакован")
    partial.replace(target)
    print(f"Данные: SHA-256 совпал: {target}")
    return target


def safe_member_path(name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise RuntimeError(f"Небезопасный путь в архиве: {name}")
    parts = ["aux_data" if os.name == "nt" and part.lower() == "aux" else part for part in pure.parts]
    return Path(*parts)


def extract_archive(archive: Path, data_root: Path, spec: Archive, force: bool,
                    public_url: str = DEFAULT_PUBLIC_URL) -> Path:
    target = data_root / spec.dataset
    if dataset_ready(target, spec.dataset) and not force:
        print(f"Данные: набор уже распакован: {target}")
        return target
    if target.exists():
        if not force:
            raise RuntimeError(f"Каталог {target} существует, но его структура неполна. Исправьте его или укажите --force-download")
        shutil.rmtree(target)

    staging = data_root / f".extract-{spec.dataset}-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    print(f"Данные: распаковка {archive} → {target}")
    try:
        with tarfile.open(archive, "r:") as source:
            for index, member in enumerate(source, 1):
                relative = safe_member_path(member.name)
                destination = staging / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    extracted = source.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Не удалось прочитать {member.name}")
                    with extracted, destination.open("wb") as output:
                        shutil.copyfileobj(extracted, output, length=8 * 1024 * 1024)
                else:
                    raise RuntimeError(f"Неподдерживаемый объект в архиве: {member.name}")
                if index % 500 == 0:
                    print(f"  распаковано объектов: {index}", flush=True)
        roots = [path for path in staging.iterdir() if path.is_dir()]
        extracted_root = staging / spec.dataset
        if not extracted_root.is_dir() and len(roots) == 1:
            extracted_root = roots[0]
        if not dataset_ready(extracted_root, spec.dataset):
            raise RuntimeError(f"После распаковки не найдена ожидаемая структура {spec.dataset}/af и {spec.dataset}/bs")
        shutil.move(str(extracted_root), str(target))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    receipt = {
        "public_url": public_url,
        "archive": spec.filename,
        "size": spec.size,
        "sha256": spec.sha256,
    }
    (target / ".source.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Данные: набор готов: {target}")
    return target


def download_aoi(public_url: str, data_root: Path) -> Path:
    target = data_root / "aoi"
    target.mkdir(parents=True, exist_ok=True)
    listing = api_json("", public_url, path="/fire-aoi", limit=100)
    required = {"README.md", "fire_monitoring_aoi.geojson"}
    for item in listing.get("_embedded", {}).get("items", []):
        if item.get("type") != "file" or item.get("name") not in required:
            continue
        path = target / item["name"]
        expected = str(item.get("sha256") or "").lower()
        if path.is_file() and (not expected or sha256(path) == expected):
            continue
        href = api_json("/download", public_url, path=item["path"])["href"]
        with urllib.request.urlopen(href, timeout=60) as response:
            path.write_bytes(response.read())
        if expected and sha256(path) != expected:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"SHA-256 не совпал для {item['name']}")
        time.sleep(.5)
    return target


def ensure_dataset(kind: str, *, data_root: Path = DEFAULT_DATA_ROOT,
                   public_url: str = DEFAULT_PUBLIC_URL, force: bool = False) -> Path:
    if kind not in ARCHIVES:
        raise ValueError(f"Неизвестный набор: {kind}")
    data_root = Path(data_root).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    target = data_root / kind
    if dataset_ready(target, kind) and not force:
        print(f"Данные: используется готовый набор {target}")
        return target
    spec = ARCHIVES[kind]
    archive = download_archive(spec, public_url, data_root / "downloads", force)
    return extract_archive(archive, data_root, spec, force, public_url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать и распаковать данные КосмоХакатона")
    parser.add_argument("--scope", choices=("train", "test", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--public-url", default=DEFAULT_PUBLIC_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="показать зафиксированные размеры и SHA-256 без сетевого запроса",
    )
    args = parser.parse_args()
    kinds = ("train", "test") if args.scope == "all" else (args.scope,)
    for kind in kinds:
        spec = ARCHIVES[kind]
        print(f"{kind}: {spec.filename}, {human_size(spec.size)}, SHA-256 {spec.sha256}")
        if not args.inspect:
            ensure_dataset(kind, data_root=args.data_root, public_url=args.public_url,
                           force=args.force_download)
    if not args.inspect:
        aoi = download_aoi(args.public_url, args.data_root)
        print(f"AOI: {aoi}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, tarfile.TarError) as exc:
        print(f"Ошибка подготовки данных: {exc}", file=sys.stderr)
        raise SystemExit(1)

"""Чтение каналов AF-чипа и расчёт признаков.

Единственное место, где описан состав признаков: им пользуются и адаптер
`predict.py` для веб-приложения, и расчёт таблицы предсказаний `inference.py`.
Порядок признаков обязан совпадать с порядком при обучении, иначе модель
получит перепутанные столбцы — поэтому список имён возвращается вместе с
матрицей и сверяется с сохранённым в `artifacts/model.pkl`.

Признаки (38 на пиксель):

- 13 исходных каналов GeoTIFF;
- 3 разности: `I4 − I5` (основной признак горения), `I4 / I5`, `I3 − I2`;
- 18 контекстных: для `I4` и `I4 − I5` в окнах 5, 11 и 21 пикселей —
  отклонение от локального среднего, нормированное отклонение и локальное СКО;
- 4 морфологических: разрыв до локального максимума, минимум и медиана.

Контекстная группа воспроизводит логику операционных алгоритмов NASA: пиксель
горяч не абсолютно, а относительно окружения.
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

AF_SHAPE = (256, 256)
GSD = 375.0
WINDOWS = (5, 11, 21)
CHANNELS = (
    "I1", "I2", "I3", "I4", "I5",
    "solar_zenith", "sensor_zenith", "valid",
    "landcover", "dem", "t2m", "rh2m", "wind_speed",
)


def window_stats(band: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Локальное среднее и среднеквадратичное отклонение в квадратном окне."""
    mean = ndimage.uniform_filter(band, size, mode="reflect")
    mean_square = ndimage.uniform_filter(band * band, size, mode="reflect")
    return mean, np.sqrt(np.maximum(mean_square - mean * mean, 1e-6))


def pixel_features(channels: dict[str, np.ndarray],
                   windows: tuple[int, ...] = WINDOWS) -> tuple[np.ndarray, list[str]]:
    """Словарь каналов -> матрица (пиксели × признаки) и список имён признаков."""
    missing = [name for name in CHANNELS if name not in channels]
    if missing:
        raise ValueError(f"В чипе нет каналов: {', '.join(missing)}")
    values = {name: np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
              for name, band in channels.items()}
    i4, i5 = values["I4"], values["I5"]
    difference = i4 - i5

    columns = [values[name] for name in CHANNELS]
    names = list(CHANNELS)
    columns += [difference, i4 / np.maximum(i5, 1e-3), values["I3"] - values["I2"]]
    names += ["diff", "I4_over_I5", "I3_minus_I2"]
    for size in windows:
        for band, tag in ((i4, "I4"), (difference, "diff")):
            mean, deviation = window_stats(band, size)
            columns += [band - mean, (band - mean) / deviation, deviation]
            names += [f"{tag}_dev{size}", f"{tag}_z{size}", f"{tag}_std{size}"]
    columns += [
        ndimage.maximum_filter(difference, 3) - difference,
        ndimage.maximum_filter(i4, 3) - i4,
        ndimage.minimum_filter(i4, 3),
        ndimage.median_filter(i4, 5),
    ]
    names += ["diff_peak_gap", "I4_peak_gap", "I4_min3", "I4_median5"]
    return np.stack(columns).reshape(len(columns), -1).T.astype(np.float32), names


def read_channels(chip_path: Path) -> dict[str, np.ndarray]:
    """Прочитать каналы одного чипа из файлов VIIRS и AUX.

    Принимает либо каталог чипа с ровно одной парой файлов, либо префикс
    `<каталог>/<chip_id>`, либо корневой каталог теста с подкаталогами
    `viirs/` и `aux/` — в последнем случае имя чипа берётся из префикса.
    Имена каналов читаются из описаний GeoTIFF, а не по порядковому номеру:
    в выданных данных имена файлов не соответствуют числу каналов внутри.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Установите зависимости расчёта: requirements-af.txt") from exc

    channels: dict[str, np.ndarray] = {}
    for path in resolve_chip_paths(chip_path):
        with rasterio.open(path) as source:
            for index, name in enumerate(source.descriptions, 1):
                if name:
                    channels[name] = source.read(index).astype(np.float32)
    shapes = {band.shape for band in channels.values()}
    if shapes != {AF_SHAPE}:
        raise ValueError(f"{chip_path}: каналы должны иметь размер {AF_SHAPE}, получено {shapes}")
    return channels


def resolve_chip_paths(chip_path: Path) -> tuple[Path, Path]:
    """Вернуть пути к файлам VIIRS и AUX для чипа."""
    chip_path = Path(chip_path)
    if chip_path.is_dir():
        viirs = sorted(chip_path.glob("*_VIIRS_I1-I5.tif"))
        aux = sorted(chip_path.glob("*_AUX.tif"))
        if len(viirs) != 1 or len(aux) != 1:
            raise ValueError(
                f"В каталоге {chip_path} ожидалась одна пара файлов VIIRS и AUX, "
                f"найдено {len(viirs)} и {len(aux)}"
            )
        return viirs[0], aux[0]
    chip_id = chip_path.name
    root = chip_path.parent
    candidates = [
        (root / "viirs" / f"{chip_id}_VIIRS_I1-I5.tif", root / "aux" / f"{chip_id}_AUX.tif"),
        (Path(f"{chip_path}_VIIRS_I1-I5.tif"), Path(f"{chip_path}_AUX.tif")),
    ]
    for viirs, aux in candidates:
        if viirs.is_file() and aux.is_file():
            return viirs, aux
    raise FileNotFoundError(
        f"Файлы чипа не найдены. Проверены: "
        f"{', '.join(str(path) for pair in candidates for path in pair)}"
    )

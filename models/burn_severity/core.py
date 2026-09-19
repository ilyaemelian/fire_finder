"""Общие признаки и операции финального BS-пайплайна для обучения и инференса."""

from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

SIZE = 512
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
LC_GROUP = {10: "forest", 20: "forest", 30: "grass", 40: "crop", 90: "wetland"}
DNBR_THRESH = {
    "forest": (0.1, 0.27, 0.66),
    "grass": (0.062, 0.204, 0.386),
    "crop": (0.07, 0.177, 0.38),
    "wetland": (0.075, 0.331, 0.677),
}


def data_folder(root: Path, folder: str) -> Path:
    path = Path(root) / folder
    if folder == "aux" and not path.is_dir() and (Path(root) / "aux_data").is_dir():
        return Path(root) / "aux_data"
    return path


def read(root: Path, folder: str, chip_id: str, suffix: str) -> dict[str, np.ndarray]:
    path = data_folder(root, folder) / f"{chip_id}_{suffix}.tif"
    with rasterio.open(path) as source:
        return {
            name: source.read(index).astype(np.float32)
            for index, name in enumerate(source.descriptions, 1)
        }


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / np.maximum(a + b, 1.0)


def severity_inputs(root: Path, chip_id: str):
    pre = read(root, "sentinel2_pre", chip_id, "Sentinel-2_pre")
    post = read(root, "sentinel2_post", chip_id, "Sentinel-2_post")
    s1_pre = read(root, "sentinel1_pre", chip_id, "Sentinel-1_pre")
    s1_post = read(root, "sentinel1_post", chip_id, "Sentinel-1_post")
    aux = read(root, "aux", chip_id, "AUX")
    smooth = lambda array: ndimage.uniform_filter(array, 3)
    nbr_pre = nd(smooth(pre["B8A"]), smooth(pre["B12"]))
    nbr_post = nd(smooth(post["B8A"]), smooth(post["B12"]))
    dnbr = nbr_pre - nbr_post
    landcover = aux["landcover"].astype(np.int32)
    thresholds = np.zeros((3,) + landcover.shape, np.float32)
    for index in range(3):
        thresholds[index] = DNBR_THRESH["grass"][index]
    for code, group in LC_GROUP.items():
        for index in range(3):
            thresholds[index][landcover == code] = DNBR_THRESH[group][index]
    severity = np.where(
        dnbr >= thresholds[2], 3,
        np.where(dnbr >= thresholds[1], 2, np.where(dnbr >= thresholds[0], 1, 0)),
    ).astype(np.uint8)
    return pre, post, s1_pre, s1_post, aux, nbr_pre, nbr_post, dnbr, landcover, thresholds, severity


def chip_features(root: Path, chip_id: str):
    (pre, post, s1_pre, s1_post, aux, nbr_pre, nbr_post, dnbr,
     landcover, thresholds, severity) = severity_inputs(root, chip_id)
    candidate = severity > 0
    ndvi_pre, ndvi_post = nd(pre["B8A"], pre["B4"]), nd(post["B8A"], post["B4"])
    nbr2_pre, nbr2_post = nd(pre["B11"], pre["B12"]), nd(post["B11"], post["B12"])
    dndvi = ndvi_pre - ndvi_post
    vv_diff = s1_post["VV"] - s1_pre["VV"]
    vh_diff = s1_post["VH"] - s1_pre["VH"]
    candidate_float = candidate.astype(np.float32)
    features = {}
    for band in BANDS:
        features[f"post_{band}"] = post[band]
        features[f"diff_{band}"] = post[band] - pre[band]
    features.update(
        nbr_pre=nbr_pre, nbr_post=nbr_post, dnbr=dnbr,
        rdnbr=dnbr / np.sqrt(np.maximum(np.abs(nbr_pre), 1e-3)),
        ndvi_pre=ndvi_pre, ndvi_post=ndvi_post, dndvi=dndvi,
        nbr2_pre=nbr2_pre, nbr2_post=nbr2_post, dnbr2=nbr2_pre - nbr2_post,
        ndwi_pre=nd(pre["B3"], pre["B8A"]), ndwi_post=nd(post["B3"], post["B8A"]),
        margin1=dnbr - thresholds[0], rel3=dnbr / thresholds[2],
        severity=severity.astype(np.float32),
        vv_pre=s1_pre["VV"], vh_pre=s1_pre["VH"],
        vv_post=s1_post["VV"], vh_post=s1_post["VH"],
        vv_diff=vv_diff, vh_diff=vh_diff,
        dem=aux["dem"], slope=aux["slope"], landcover=landcover.astype(np.float32),
        scl_pre=pre["SCL"], scl_post=post["SCL"],
    )
    for window in (5, 15, 31):
        features[f"dnbr_mean{window}"] = ndimage.uniform_filter(dnbr, window)
        features[f"dndvi_mean{window}"] = ndimage.uniform_filter(dndvi, window)
        features[f"cand_frac{window}"] = ndimage.uniform_filter(candidate_float, window)
    for window in (15, 31):
        mean = ndimage.uniform_filter(dnbr, window)
        features[f"dnbr_std{window}"] = np.sqrt(
            np.maximum(ndimage.uniform_filter(dnbr * dnbr, window) - mean * mean, 0)
        )
        features[f"vv_diff_mean{window}"] = ndimage.uniform_filter(vv_diff, window)
        features[f"vh_diff_mean{window}"] = ndimage.uniform_filter(vh_diff, window)
    features["cand_frac61"] = ndimage.uniform_filter(candidate_float, 61)
    names = list(features)
    matrix = np.stack([
        np.nan_to_num(features[name], nan=0.0, posinf=0.0, neginf=0.0).ravel()
        for name in names
    ], axis=1)
    return matrix, names, severity.ravel(), candidate.ravel()


def build_cache(root: Path, chip_ids: list[str], work: Path | None = None,
                prefix: str = "test", masks: Path | None = None, progress: bool = True):
    """Признаки всех чипов, сведённые в одну матрицу.

    В матрицу попадают только пиксели-кандидаты: те, где индекс dNBR выше порога
    своего типа покрова. По данным изучения там лежит 98.7 % гари, поэтому
    остальные пиксели считать не нужно, и матрица уменьшается примерно в пять раз.

    По умолчанию матрица собирается в оперативной памяти: для 89 тестовых чипов
    это порядка одного гигабайта. Аргумент `work` включает промежуточный файл на
    диске и нужен лишь при нехватке памяти на большом наборе.
    """
    offsets, severity, indices, truth, outside = [0], [], [], [], []
    blocks: list[np.ndarray] = []
    stream = None
    feature_path = None
    if work is not None:
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        feature_path = work / f"{prefix}_X.bin"
        stream = feature_path.open("wb")
    names = None
    try:
        for number, chip_id in enumerate(chip_ids, 1):
            matrix, current_names, current_severity, candidate = chip_features(root, chip_id)
            if names is None:
                names = current_names
            elif current_names != names:
                raise ValueError(f"{chip_id}: изменился порядок признаков")
            rows = np.flatnonzero(candidate)
            block = matrix[rows].astype(np.float32)
            if stream is not None:
                block.tofile(stream)
            else:
                blocks.append(block)
            severity.append(current_severity[rows])
            indices.append(rows.astype(np.int32))
            offsets.append(offsets[-1] + rows.size)
            if masks is not None:
                with rasterio.open(masks / f"{chip_id}_MASK.tif") as source:
                    mask = source.read(1).ravel().astype(np.uint8)
                truth.append(mask[rows])
                outside.append(np.bincount(mask[~candidate], minlength=4)[:4])
            if progress and (number % 20 == 0 or number == len(chip_ids)):
                print(f"    признаки {number}/{len(chip_ids)}", flush=True)
    finally:
        if stream is not None:
            stream.close()
    if names is None:
        raise ValueError("Список BS-чипов пуст")
    result = {
        "names": names,
        "offsets": np.asarray(offsets, dtype=np.int64),
        "sev": np.concatenate(severity),
        "idx": np.concatenate(indices),
    }
    if masks is not None:
        result.update(gt=np.concatenate(truth), outside=np.asarray(outside))
    if stream is not None:
        result["X"] = np.memmap(feature_path, dtype=np.float32, mode="r",
                                shape=(int(offsets[-1]), len(names)))
        return result

    # Матрица собирается копированием в заранее выделенный массив, а не
    # объединением списка: объединение на время работы держит в памяти две
    # полные копии, и на наборах в гигабайт и больше это приводит к обмену
    # страниц с диском и видимому зависанию. Каждый блок освобождается сразу
    # после копирования, поэтому пик равен размеру матрицы плюс один блок.
    total = int(offsets[-1])
    volume = total * len(names) * 4 / 1_073_741_824
    if progress:
        print(f"    матрица признаков: {total:,} строк × {len(names)} "
              f"признаков, {volume:.2f} ГБ".replace(",", " "), flush=True)
        if volume > 3.0:
            print("    при нехватке памяти укажите --work <каталог>: "
                  "признаки будут храниться на диске", flush=True)
    matrix = np.empty((total, len(names)), np.float32)
    position = 0
    for number, block in enumerate(blocks):
        matrix[position:position + len(block)] = block
        position += len(block)
        blocks[number] = None  # освободить блок сразу
    result["X"] = matrix
    return result


def context(probability: np.ndarray, cache: dict, chips=None) -> np.ndarray:
    indices, offsets = cache["idx"], cache["offsets"]
    output = np.zeros((len(probability), 10), np.float32)
    selected = range(len(offsets) - 1) if chips is None else chips
    for chip in selected:
        start, stop = offsets[chip], offsets[chip + 1]
        if start == stop:
            continue
        surface = np.zeros(SIZE * SIZE, np.float32)
        surface[indices[start:stop]] = probability[start:stop]
        surface = surface.reshape(SIZE, SIZE)
        candidate = np.zeros(SIZE * SIZE, bool)
        candidate[indices[start:stop]] = True
        candidate = candidate.reshape(SIZE, SIZE)
        high = surface >= 0.3
        labels, count = ndimage.label(high)
        area = np.bincount(labels.ravel(), minlength=count + 1).astype(np.float32)
        area[0] = 0
        means = np.divide(
            np.bincount(labels.ravel(), weights=surface.ravel(), minlength=count + 1),
            np.maximum(area, 1),
        )
        means[0] = 0
        features = [
            surface,
            ndimage.uniform_filter(surface, 7),
            ndimage.uniform_filter(surface, 15),
            ndimage.uniform_filter(surface, 31),
            ndimage.uniform_filter(surface, 61),
            ndimage.maximum_filter(surface, 5),
            ndimage.minimum_filter(np.where(candidate, surface, 1.0), 5),
            np.log1p(area[labels]),
            means[labels],
            ndimage.uniform_filter(high.astype(np.float32), 31),
        ]
        output[start:stop] = np.stack(
            [feature.ravel()[indices[start:stop]] for feature in features], axis=1
        )
    return output


def predict_probability(model, matrix, extra=None, chunk: int = 500_000) -> np.ndarray:
    output = np.zeros(len(matrix), np.float32)
    for start in range(0, len(matrix), chunk):
        stop = min(start + chunk, len(matrix))
        rows = np.asarray(matrix[start:stop])
        if extra is not None:
            rows = np.hstack([rows, extra[start:stop]])
        output[start:stop] = model.predict_proba(rows)[:, 1]
    return output


def unet_tensor(root: Path, chip_id: str) -> np.ndarray:
    (pre, post, s1_pre, s1_post, aux, nbr_pre, nbr_post, dnbr,
     landcover, thresholds, severity) = severity_inputs(root, chip_id)
    reflectance = lambda source, band: source[band] / 10000.0
    channels = [reflectance(pre, band) for band in BANDS]
    channels += [reflectance(post, band) for band in BANDS]
    channels += [
        dnbr,
        nd(pre["B8A"], pre["B4"]) - nd(post["B8A"], post["B4"]),
        nd(pre["B11"], pre["B12"]) - nd(post["B11"], post["B12"]),
        nbr_pre,
        nbr_post,
        (10 * reflectance(pre, "B12") - 9.8 * reflectance(pre, "B11"))
        - (10 * reflectance(post, "B12") - 9.8 * reflectance(post, "B11")),
        dnbr - thresholds[0],
        severity / 3.0,
        s1_pre["VV"] / 1000,
        s1_pre["VH"] / 1000,
        s1_post["VV"] / 1000,
        s1_post["VH"] / 1000,
        (s1_post["VV"] - s1_pre["VV"]) / 1000,
        (s1_post["VH"] - s1_pre["VH"]) / 1000,
        aux["dem"] / 500.0,
        aux["slope"] / 30.0,
        np.isin(landcover, [10, 20]),
        landcover == 30,
        landcover == 40,
        landcover == 90,
        ~np.isin(landcover, [10, 20, 30, 40, 90]),
        np.isin(post["SCL"], [3, 8, 9, 10]),
    ]
    return np.nan_to_num(
        np.stack([np.asarray(channel, np.float32) for channel in channels]),
        nan=0, posinf=0, neginf=0,
    ).astype(np.float16)


def rle_encode(mask: np.ndarray) -> str:
    flat = np.concatenate([[0], mask.astype(np.uint8).ravel(), [0]])
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    return " ".join(f"{start} {stop - start}" for start, stop in zip(edges[0::2], edges[1::2]))

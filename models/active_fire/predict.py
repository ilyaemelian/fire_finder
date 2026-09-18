"""Inference adapter for the AF YOLO model supplied by the teammate."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AF_SHAPE = (256, 256)
UPSCALE = 4
IMAGE_SIZE = AF_SHAPE[0] * UPSCALE
MODEL_CONFIDENCE_FLOOR = 0.001
MASK_CONFIDENCE = 0.50
I4_RANGE = (270.0, 367.0)
I4_I5_RANGE = (0.0, 60.0)
I3_RANGE = (0.0, 0.6)


@dataclass(frozen=True)
class ActiveFireModel:
    model: Any
    device: str | int
    confidence: float = MASK_CONFIDENCE


def _scale(values: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def _resolve_viirs_path(chip_path: Path) -> Path:
    chip_path = Path(chip_path)
    if chip_path.is_file():
        return chip_path
    if not chip_path.is_dir():
        raise FileNotFoundError(f"AF chip not found: {chip_path}")
    matches = sorted(chip_path.glob("*_VIIRS_I1-I5.tif"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one *_VIIRS_I1-I5.tif in {chip_path}, found {len(matches)}"
        )
    return matches[0]


def load_channels(chip_path: Path) -> dict[str, np.ndarray]:
    """Read VIIRS bands by GeoTIFF descriptions, not numeric position."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install AF inference dependencies first") from exc

    path = _resolve_viirs_path(chip_path)
    channels: dict[str, np.ndarray] = {}
    with rasterio.open(path) as source:
        for index, name in enumerate(source.descriptions, 1):
            if name:
                channels[name] = source.read(index).astype(np.float32)
    required = {"I3", "I4", "I5", "valid"}
    missing = sorted(required - channels.keys())
    if missing:
        raise ValueError(f"Missing VIIRS channel descriptions: {', '.join(missing)}")
    if any(channels[name].shape != AF_SHAPE for name in required):
        raise ValueError(f"Every AF channel must have shape {AF_SHAPE}")
    return channels


def render_channels(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Create the exact 1024x1024 BGR image used in the notebook."""
    red = _scale(channels["I4"], *I4_RANGE)
    green = _scale(channels["I4"] - channels["I5"], *I4_I5_RANGE)
    blue = _scale(channels["I3"], *I3_RANGE)
    invalid = channels["valid"] < 0.5
    red[invalid] = green[invalid] = blue[invalid] = 0
    image = np.dstack([blue, green, red])
    return np.repeat(np.repeat(image, UPSCALE, axis=0), UPSCALE, axis=1)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def boxes_to_mask(result: Any, confidence: float) -> np.ndarray:
    mask = np.zeros(AF_SHAPE, dtype=np.uint8)
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return mask
    coordinates = _as_numpy(boxes.xyxy) / UPSCALE
    scores = _as_numpy(boxes.conf)
    for (x0, y0, x1, y1), score in zip(coordinates, scores):
        if float(score) < confidence:
            continue
        row_start = max(int(np.floor(y0)), 0)
        row_stop = min(int(np.ceil(y1)), AF_SHAPE[0])
        col_start = max(int(np.floor(x0)), 0)
        col_stop = min(int(np.ceil(x1)), AF_SHAPE[1])
        if row_start < row_stop and col_start < col_stop:
            mask[row_start:row_stop, col_start:col_stop] = 1
    return mask


def validate_af_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("AF prediction must be a numpy.ndarray")
    if mask.shape != AF_SHAPE:
        raise ValueError(f"AF prediction must have shape {AF_SHAPE}, got {mask.shape}")
    if mask.dtype != np.uint8:
        raise TypeError(f"AF prediction must be uint8, got {mask.dtype}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("AF prediction may contain only 0 and 1")


def load_model(model_path: Path, device: str = "cpu") -> ActiveFireModel:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"AF weights not found: {path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install requirements-af.txt in the ML environment."
        ) from exc
    runtime_device: str | int = int(device) if device.isdigit() else device
    return ActiveFireModel(model=YOLO(str(path)), device=runtime_device)


def predict_af(chip_path: Path, model: ActiveFireModel) -> np.ndarray:
    """Return the final binary AF mask for one VIIRS GeoTIFF."""
    image = render_channels(load_channels(chip_path))
    result = model.model.predict(
        image,
        imgsz=IMAGE_SIZE,
        conf=MODEL_CONFIDENCE_FLOOR,
        device=model.device,
        verbose=False,
    )[0]
    mask = boxes_to_mask(result, model.confidence)
    validate_af_mask(mask)
    return mask

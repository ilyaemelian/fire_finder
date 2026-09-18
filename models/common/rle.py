"""Competition-compatible row-major RLE."""

import numpy as np


def encode_rle(mask: np.ndarray) -> str:
    if mask.ndim != 2:
        raise ValueError("RLE expects a two-dimensional mask")
    binary = (mask != 0).ravel(order="C").astype(np.uint8)
    padded = np.concatenate(([0], binary, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    return " ".join(
        f"{start} {stop - start}"
        for start, stop in zip(changes[0::2], changes[1::2])
    )


def decode_rle(value: str, shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    value = value.strip() if isinstance(value, str) else ""
    if not value:
        return output.reshape(shape, order="C")
    try:
        numbers = [int(part) for part in value.split()]
    except ValueError as exc:
        raise ValueError("RLE contains a non-integer value") from exc
    if len(numbers) % 2:
        raise ValueError("RLE must contain start/length pairs")
    previous_stop = 0
    for start, length in zip(numbers[0::2], numbers[1::2]):
        stop = start + length - 1
        if start < 1 or length < 1 or stop > output.size:
            raise ValueError("RLE run is outside the mask")
        if previous_stop and start <= previous_stop + 1:
            raise ValueError("RLE runs overlap, touch, or are not sorted")
        output[start - 1 : stop] = 1
        previous_stop = stop
    return output.reshape(shape, order="C")

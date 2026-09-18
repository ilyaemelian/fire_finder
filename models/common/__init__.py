"""Общий контракт моделей: RLE и работа с CSV предсказаний.

Обе модели пишут результат в один и тот же формат (chip_id, class_id, rle),
поэтому кодирование, валидация и сборка живут здесь, а не внутри модулей.
"""

from .rle import decode_rle, encode_rle
from .submission import (
    FIELDS,
    kind_of,
    kinds_present,
    read_rows,
    validate_rows,
    write_rows,
)

__all__ = [
    "FIELDS",
    "decode_rle",
    "encode_rle",
    "kind_of",
    "kinds_present",
    "read_rows",
    "validate_rows",
    "write_rows",
]

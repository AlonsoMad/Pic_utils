"""Color helpers: hex parsing, luminance, and the RGB/BGR boundary.

RGB is canonical everywhere; OpenCV's BGR order is confined to
:func:`to_bgr` / :func:`from_bgr` (architecture §4.3). ``luminance`` is the
Rec. 709 luma used by mask auto-thresholds.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import cv2
import numpy as np

__all__ = ["from_bgr", "luminance", "parse_hex_color", "to_bgr", "to_hex_color"]

_HEX_RE = re.compile(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
_LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def parse_hex_color(value: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB``, ``RRGGBB``, or the ``#RGB`` shorthand.

    Raises ``ValueError`` for anything else and ``TypeError`` for a non-string.
    """
    if not isinstance(value, str):
        raise TypeError(f"hex color must be a str, got {type(value).__name__}")
    match = _HEX_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid hex color: {value!r}")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(character * 2 for character in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def to_hex_color(rgb: Sequence[int] | np.ndarray) -> str:
    """Return ``#rrggbb`` for three integer components in 0..255."""
    array = np.asarray(rgb)
    if array.shape != (3,):
        raise ValueError(f"expected three color components, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"color components must be integers, got {array.dtype}")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError(f"color components must be in 0..255, got {array.tolist()}")
    red, green, blue = (int(component) for component in array)
    return f"#{red:02x}{green:02x}{blue:02x}"


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Return the Rec. 709 luma of an ``(H, W, 3)`` image as float32.

    uint8 input is interpreted on the 0..255 scale; float input is used as
    given. The output has the same scale as the input.
    """
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {array.shape}")
    if array.dtype not in (np.uint8, np.float32, np.float64):
        raise TypeError(f"unsupported image dtype: {array.dtype}")
    values = array.astype(np.float32, copy=False)
    return values @ _LUMA_709


def _validate_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) array, got shape {array.shape}")
    if array.dtype not in (np.uint8, np.float32, np.float64):
        raise TypeError(f"unsupported image dtype: {array.dtype}")
    return np.ascontiguousarray(array)


def to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB array to OpenCV's BGR order."""
    return cv2.cvtColor(_validate_rgb(np.asarray(rgb)), cv2.COLOR_RGB2BGR)


def from_bgr(bgr: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR array back to RGB."""
    return cv2.cvtColor(_validate_rgb(np.asarray(bgr)), cv2.COLOR_BGR2RGB)

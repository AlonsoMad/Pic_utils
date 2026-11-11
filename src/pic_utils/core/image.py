"""Canonical image value types (architecture §3).

``Image`` is the pipeline currency: 8-bit RGB plus an optional alpha plane.
Construction validates shape/dtype, normalizes to C-contiguous arrays, and
marks them read-only so a block cannot mutate its input. ``Mask`` is a
single-channel 8-bit plane, ``Layer`` pairs an image with an optional mask,
and ``BlendMode`` is the six-mode canvas enum.

No pixel-value scans happen here: at 48 MP they are too expensive. Binary
mask validation belongs to the executor's output validator (§5.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

__all__ = ["BlendMode", "Image", "Layer", "Mask"]


def _normalize_plane(
    value: object,
    *,
    name: str,
    ndim: int,
    channels: int | None = None,
) -> np.ndarray:
    """Validate one uint8 plane, make it C-contiguous, and freeze it."""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(value).__name__}")
    if value.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {value.dtype}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {value.shape}")
    if channels is not None and value.shape[-1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got shape {value.shape}")
    if value.shape[0] < 1 or value.shape[1] < 1:
        raise ValueError(f"{name} must be at least 1x1, got shape {value.shape}")
    if not value.flags.c_contiguous:
        value = np.ascontiguousarray(value)
    value.setflags(write=False)
    return value


@dataclass(frozen=True, slots=True, eq=False)
class Image:
    """An 8-bit RGB image with an optional alpha plane.

    ``rgb`` is always ``(H, W, 3)`` uint8, C-contiguous and read-only.
    ``alpha`` is ``None`` (opaque) or ``(H, W)`` uint8, read-only. The
    constructor takes ownership of the arrays it is given: it marks them
    read-only, so callers MUST NOT reuse a mutable buffer afterwards.
    """

    rgb: np.ndarray
    alpha: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = _normalize_plane(self.rgb, name="rgb", ndim=3, channels=3)
        object.__setattr__(self, "rgb", rgb)
        if self.alpha is not None:
            alpha = _normalize_plane(self.alpha, name="alpha", ndim=2)
            if alpha.shape != rgb.shape[:2]:
                raise ValueError(
                    f"alpha shape {alpha.shape} does not match rgb shape {rgb.shape[:2]}"
                )
            object.__setattr__(self, "alpha", alpha)

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """Return ``(width, height)``."""
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        return int(self.rgb.nbytes + (0 if self.alpha is None else self.alpha.nbytes))

    @classmethod
    def from_rgb(cls, arr: np.ndarray) -> Image:
        """Wrap an ``(H, W, 3)`` uint8 array as an opaque image."""
        array = np.asarray(arr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an (H, W, 3) array, got shape {array.shape}")
        return cls(array)

    @classmethod
    def from_rgba(cls, arr: np.ndarray) -> Image:
        """Split an ``(H, W, 4)`` uint8 array into RGB plus alpha."""
        array = np.asarray(arr)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError(f"expected an (H, W, 4) array, got shape {array.shape}")
        return cls(array[..., :3], array[..., 3])

    def to_rgba(self) -> np.ndarray:
        """Return a new ``(H, W, 4)`` uint8 array; missing alpha is 255."""
        if self.alpha is None:
            alpha = np.full((*self.rgb.shape[:2], 1), 255, dtype=np.uint8)
        else:
            alpha = self.alpha[:, :, None]
        return np.concatenate((self.rgb, alpha), axis=2)

    def with_alpha(self, alpha: np.ndarray | None) -> Image:
        """Return an image with the same RGB and a different alpha plane."""
        return Image(self.rgb, alpha)

    def copy(self) -> Image:
        """Return an independent copy of both planes."""
        alpha = None if self.alpha is None else self.alpha.copy()
        return Image(self.rgb.copy(), alpha)

    def resized(
        self,
        width: int,
        height: int,
        interpolation: int = cv2.INTER_AREA,
    ) -> Image:
        """Return a resized copy; alpha is resized with the same filter."""
        if width < 1 or height < 1:
            raise ValueError(f"resize target must be at least 1x1, got {width}x{height}")
        if width == self.width and height == self.height:
            return self
        rgb = cv2.resize(self.rgb, (width, height), interpolation=interpolation)
        alpha = (
            None
            if self.alpha is None
            else cv2.resize(self.alpha, (width, height), interpolation=interpolation)
        )
        return Image(rgb, alpha)

    def to_proxy(self, max_long_side: int) -> Image:
        """Downscale so the long side is at most ``max_long_side``.

        Aspect ratio is preserved (within rounding); alpha is resized too.
        An image already within the bound is returned unchanged.
        """
        if max_long_side < 1:
            raise ValueError(f"max_long_side must be positive, got {max_long_side}")
        long_side = max(self.width, self.height)
        if long_side <= max_long_side:
            return self
        scale = max_long_side / long_side
        width = max(1, round(self.width * scale))
        height = max(1, round(self.height * scale))
        return self.resized(width, height)


@dataclass(frozen=True, slots=True, eq=False)
class Mask:
    """A single-channel uint8 plane.

    ``soft=False`` masks are expected to contain only 0 and 255; that value
    check is the executor's output-validator job (§5.6), not a construction
    cost paid on every mask.
    """

    data: np.ndarray
    soft: bool = False

    def __post_init__(self) -> None:
        data = _normalize_plane(self.data, name="data", ndim=2)
        object.__setattr__(self, "data", data)
        if not isinstance(self.soft, bool):
            raise TypeError(f"soft must be a bool, got {type(self.soft).__name__}")


@dataclass(frozen=True, slots=True, eq=False)
class Layer:
    """One block output: an image plus an optional mask in its coordinates."""

    image: Image
    mask: Mask | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.image, Image):
            raise TypeError(f"image must be an Image, got {type(self.image).__name__}")
        if self.mask is not None:
            if not isinstance(self.mask, Mask):
                raise TypeError(f"mask must be a Mask or None, got {type(self.mask).__name__}")
            if self.mask.data.shape != self.image.rgb.shape[:2]:
                raise ValueError(
                    f"mask shape {self.mask.data.shape} does not match "
                    f"image shape {self.image.rgb.shape[:2]}"
                )
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a str, got {type(self.name).__name__}")


class BlendMode(StrEnum):
    """The six canvas blend modes required by FR-CAN-04."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    DIFFERENCE = "difference"

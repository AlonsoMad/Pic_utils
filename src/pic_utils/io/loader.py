"""Image loading: the only place that knows file formats and EXIF (§4).

``load_image`` normalizes PNG/JPEG/TIFF input to the canonical ``Image``:
8-bit RGB plus an optional alpha plane, EXIF orientation applied, and
16-bit or oversized inputs rejected with typed errors.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms, ImageOps, UnidentifiedImageError

from pic_utils.core.errors import ImageLoadError, ImageTooLargeError, UnsupportedImageError
from pic_utils.core.image import Image, Mask
from pic_utils.io import formats

__all__ = ["ImageInfo", "load_image", "load_mask", "probe_image"]

logger = logging.getLogger(__name__)

_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N"})
_HIGH_DEPTH_MODES = frozenset({"I", "F"})
_ALPHA_MODES = frozenset({"RGBA", "LA", "PA", "RGBa", "La"})
_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Cheap metadata read without decoding pixels."""

    width: int
    height: int
    mode: str
    format: str
    has_alpha: bool


def _has_alpha(mode: str, info: Mapping[Any, Any]) -> bool:
    return mode in _ALPHA_MODES or "transparency" in info


def _check_size(path: Path, width: int, height: int) -> None:
    if width * height > formats.MAX_PIXELS:
        raise ImageTooLargeError(
            path=path,
            reason="above the maximum supported size",
            width=width,
            height=height,
        )


def _check_mode(path: Path, mode: str) -> None:
    if mode in _SIXTEEN_BIT_MODES:
        raise UnsupportedImageError(
            path=path,
            reason="image.16bit: 16-bit images are not supported",
        )
    if mode in _HIGH_DEPTH_MODES:
        raise UnsupportedImageError(
            path=path,
            reason=f"unsupported pixel mode {mode!r}: only 8-bit images are supported",
        )


def _warn_on_non_srgb(path: Path, image: PILImage.Image) -> None:
    profile = image.info.get("icc_profile")
    if profile and profile != _SRGB_PROFILE:
        logger.warning(
            "image %s has a non-sRGB ICC profile; using the sRGB working space",
            path,
        )


def _to_uint8_array(image: PILImage.Image) -> np.ndarray:
    # Copy before the ``with`` block closes the Pillow image: the array must
    # own its buffer, not point into the decoder's memory.
    return np.array(image, dtype=np.uint8, copy=True)


def probe_image(path: str | Path) -> ImageInfo:
    """Read size, mode, format, and alpha presence without decoding pixels."""
    source = Path(path)
    try:
        with PILImage.open(source) as image:
            image_format = image.format
            if image_format is None or image_format not in formats.SUPPORTED_FORMATS:
                raise UnsupportedImageError(
                    path=source,
                    reason=f"unsupported image format {image_format!r}",
                )
            return ImageInfo(
                width=image.width,
                height=image.height,
                mode=image.mode,
                format=image_format,
                has_alpha=_has_alpha(image.mode, image.info),
            )
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageLoadError(path=source, reason=str(error)) from error


def load_image(path: str | Path) -> Image:
    """Load PNG/JPEG/TIFF, apply EXIF orientation, and normalize to RGB(A)."""
    source = Path(path)
    try:
        with PILImage.open(source) as image:
            image_format = image.format
            if image_format is None or image_format not in formats.SUPPORTED_FORMATS:
                raise UnsupportedImageError(
                    path=source,
                    reason=f"unsupported image format {image_format!r}",
                )
            _check_size(source, image.width, image.height)
            _check_mode(source, image.mode)
            _warn_on_non_srgb(source, image)

            transposed = ImageOps.exif_transpose(image)
            if transposed.mode == "RGB":
                rgb = _to_uint8_array(transposed)
                alpha = None
            elif transposed.mode == "RGBA":
                rgba = _to_uint8_array(transposed)
                rgb = rgba[..., :3]
                alpha = rgba[..., 3]
            else:
                target = "RGBA" if _has_alpha(transposed.mode, transposed.info) else "RGB"
                pixels = _to_uint8_array(transposed.convert(target))
                if target == "RGBA":
                    rgb = pixels[..., :3]
                    alpha = pixels[..., 3]
                else:
                    rgb = pixels
                    alpha = None
            return Image(rgb, alpha)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageLoadError(path=source, reason=str(error)) from error


def load_mask(path: str | Path) -> Mask:
    """Load an 8-bit grayscale PNG as a soft mask."""
    source = Path(path)
    try:
        with PILImage.open(source) as image:
            if image.format != "PNG":
                raise UnsupportedImageError(
                    path=source,
                    reason=f"mask must be a PNG, got {image.format!r}",
                )
            _check_size(source, image.width, image.height)
            _check_mode(source, image.mode)
            grayscale = image.convert("L") if image.mode != "L" else image
            return Mask(_to_uint8_array(grayscale), soft=True)
    except ImageLoadError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageLoadError(path=source, reason=str(error)) from error

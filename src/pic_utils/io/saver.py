"""Atomic image saving (architecture §4.2, NFR-REL-03).

Every write goes to a temporary file in the destination directory, is
fsynced, and is then moved into place with ``os.replace``. A failed save
never leaves a partial target or a stray temporary file.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageCms

from pic_utils.core.errors import ExportError
from pic_utils.core.image import Image, Mask
from pic_utils.io import formats

__all__ = ["save_image", "save_mask"]

_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _atomic_save(
    pil_image: PILImage.Image,
    path: Path,
    *,
    image_format: str,
    **params: Any,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".pic_utils-",
        suffix=path.suffix,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pil_image.save(handle, format=image_format, **params)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


def _flatten_on_white(image: Image) -> np.ndarray:
    if image.alpha is None:
        return image.rgb
    alpha = image.alpha.astype(np.uint32)[..., None]
    rgb = image.rgb.astype(np.uint32)
    flattened = (rgb * alpha + 255 * (255 - alpha) + 127) // 255
    return flattened.astype(np.uint8)


def save_image(
    image: Image,
    path: str | Path,
    *,
    jpeg_quality: int = 92,
    tiff_compression: str = "tiff_lzw",
    embed_srgb: bool = True,
) -> None:
    """Save ``image`` as PNG, JPEG, or TIFF, atomically.

    The format comes from the destination extension. JPEG flattens alpha
    onto white; TIFF uses LZW by default; all three formats embed an sRGB
    ICC profile unless ``embed_srgb`` is false.
    """
    if not isinstance(image, Image):
        raise TypeError(f"image must be an Image, got {type(image).__name__}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be in 1..100, got {jpeg_quality}")

    destination = Path(path)
    image_format = formats.format_for_path(destination)
    profile = _SRGB_PROFILE if embed_srgb else None

    if image_format == "PNG":
        pixels = image.rgb if image.alpha is None else image.to_rgba()
        pil_image = PILImage.fromarray(pixels)
        params: dict[str, Any] = {"compress_level": 3}
    elif image_format == "JPEG":
        pil_image = PILImage.fromarray(_flatten_on_white(image))
        params = {"quality": jpeg_quality}
    else:
        pixels = image.rgb if image.alpha is None else image.to_rgba()
        pil_image = PILImage.fromarray(pixels)
        params = {"compression": tiff_compression}

    if profile is not None:
        params["icc_profile"] = profile

    try:
        _atomic_save(pil_image, destination, image_format=image_format, **params)
    except (OSError, ValueError) as error:
        raise ExportError(f"cannot save {destination}: {error}") from error


def save_mask(mask: Mask, path: str | Path) -> None:
    """Save a mask as an atomic 8-bit grayscale PNG."""
    if not isinstance(mask, Mask):
        raise TypeError(f"mask must be a Mask, got {type(mask).__name__}")
    destination = Path(path)
    if formats.format_for_path(destination) != "PNG":
        raise ValueError("masks must be saved as PNG")
    pil_image = PILImage.fromarray(mask.data)
    try:
        _atomic_save(pil_image, destination, image_format="PNG", compress_level=3)
    except (OSError, ValueError) as error:
        raise ExportError(f"cannot save {destination}: {error}") from error

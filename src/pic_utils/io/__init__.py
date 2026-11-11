"""The image file boundary (architecture §4).

``io/`` is the only package that knows file formats, EXIF, and channel
order at the boundary. Loads normalize to ``core.image.Image``; saves are
atomic. Nothing here leaks OpenCV's BGR order.
"""

from pic_utils.io.formats import (
    MAX_PIXELS,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
    format_for_path,
    is_supported_path,
)
from pic_utils.io.loader import ImageInfo, load_image, load_mask, probe_image
from pic_utils.io.saver import save_image, save_mask

__all__ = [
    "MAX_PIXELS",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMATS",
    "ImageInfo",
    "format_for_path",
    "is_supported_path",
    "load_image",
    "load_mask",
    "probe_image",
    "save_image",
    "save_mask",
]

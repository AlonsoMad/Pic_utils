"""Shared canvas painting primitives (architecture §10.3-§10.5).

The scene and the export renderer use the same transform construction and
the same blend-mode mapping, so a 1:1 export matches the screen. Qt
post-multiplies transforms: the first call is the outermost transform
(applied last to a point). ``local_transform`` therefore starts with the
image-center translation and ends with its inverse.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QImage, QPainter, QTransform

from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.core.color import parse_hex_color
from pic_utils.core.image import BlendMode, Image, Mask

__all__ = [
    "BLEND_TO_QT",
    "background_color",
    "effective_opacity",
    "layer_transform",
    "layer_visible",
    "local_transform",
    "qimage_from_image",
    "qimage_to_image",
]

BLEND_TO_QT: dict[BlendMode, QPainter.CompositionMode] = {
    BlendMode.NORMAL: QPainter.CompositionMode.CompositionMode_SourceOver,
    BlendMode.MULTIPLY: QPainter.CompositionMode.CompositionMode_Multiply,
    BlendMode.SCREEN: QPainter.CompositionMode.CompositionMode_Screen,
    BlendMode.OVERLAY: QPainter.CompositionMode.CompositionMode_Overlay,
    BlendMode.ADD: QPainter.CompositionMode.CompositionMode_Plus,
    BlendMode.DIFFERENCE: QPainter.CompositionMode.CompositionMode_Difference,
}


def local_transform(layer: CanvasLayer, width: int, height: int) -> QTransform:
    """Rotation/scale/flip about the image center; no translation (§10.3)."""
    transform = QTransform()
    transform.translate(width / 2, height / 2)
    transform.rotate(layer.rotation)
    transform.scale(
        layer.scale * (-1 if layer.flip_h else 1),
        layer.scale * (-1 if layer.flip_v else 1),
    )
    transform.translate(-width / 2, -height / 2)
    return transform


def layer_transform(layer: CanvasLayer, width: int, height: int) -> QTransform:
    """Full transform: translation is outermost, then the local transform."""
    transform = QTransform()
    transform.translate(layer.x, layer.y)
    transform.translate(width / 2, height / 2)
    transform.rotate(layer.rotation)
    transform.scale(
        layer.scale * (-1 if layer.flip_h else 1),
        layer.scale * (-1 if layer.flip_v else 1),
    )
    transform.translate(-width / 2, -height / 2)
    return transform


def background_color(background: str | None) -> QColor | None:
    """Return the canvas background color, or ``None`` for transparency."""
    if background is None:
        return None
    red, green, blue = parse_hex_color(background)
    return QColor(red, green, blue)


def layer_visible(doc: CanvasDoc, layer: CanvasLayer) -> bool:
    """Whether ``layer`` is visible, including its group's visibility."""
    if not layer.visible:
        return False
    group = doc.groups.get(layer.group_id) if layer.group_id is not None else None
    return group is None or group.visible


def effective_opacity(doc: CanvasDoc, layer: CanvasLayer) -> float:
    """Layer opacity multiplied by its group's opacity (per child, §10.4)."""
    group = doc.groups.get(layer.group_id) if layer.group_id is not None else None
    return layer.opacity * (group.opacity if group is not None else 1.0)


def qimage_from_image(image: Image, mask: Mask | None = None) -> QImage:
    """Build a premultiplied ARGB image; ``mask`` multiplies the alpha plane."""
    rgba = image.to_rgba()
    if mask is not None:
        if mask.data.shape != rgba.shape[:2]:
            raise ValueError(
                f"mask shape {mask.data.shape} does not match image shape {rgba.shape[:2]}"
            )
        alpha = rgba[:, :, 3].astype(np.uint16)
        rgba[:, :, 3] = (alpha * mask.data // 255).astype(np.uint8)
    height, width = rgba.shape[:2]
    qimage = QImage(
        rgba.data,
        width,
        height,
        int(rgba.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()
    return qimage.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)


def qimage_to_image(qimage: QImage) -> Image:
    """Convert a QImage (any format) back to the core RGBA image type."""
    converted = qimage.convertToFormat(QImage.Format.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    stride = converted.bytesPerLine()
    buffer = converted.constBits()
    rows = np.frombuffer(buffer, dtype=np.uint8, count=stride * height).reshape(height, stride)
    rgba = rows[:, : width * 4].reshape(height, width, 4).copy()
    return Image.from_rgba(rgba)

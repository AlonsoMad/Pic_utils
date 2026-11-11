"""Canvas scene: layer items, transform handles, and group semantics (§10.3).

The scene is a view of :class:`~pic_utils.canvas.model.CanvasDoc`. Layer
items use ``local_transform`` plus ``setPos``; z-order is the document
index. Groups are applied per child (visibility and opacity), matching the
measured Qt behavior and the export renderer (§10.4).
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsRectItem,
    QGraphicsScene,
    QStyleOptionGraphicsItem,
    QWidget,
)

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.canvas.render import (
    BLEND_TO_QT,
    background_color,
    effective_opacity,
    layer_visible,
    local_transform,
    qimage_from_image,
)
from pic_utils.core.errors import CanvasFileError
from pic_utils.core.image import Image
from pic_utils.io.loader import load_image, load_mask

__all__ = ["CanvasScene", "LayerItem", "TransformHandle"]

_HANDLE_SIZE = 8.0
_ROTATION_OFFSET = 24.0
_OUTLINE = QColor(0, 120, 215)
_HANDLE_FILL = QColor(255, 255, 255)
_HANDLE_BORDER = QColor(0, 90, 170)
_MIN_SCALE = 0.01
_MAX_SCALE = 100.0


class TransformHandle(QGraphicsRectItem):
    """A constant-screen-size handle; drags are forwarded to the layer item."""

    def __init__(self, role: str, parent: LayerItem) -> None:
        half = _HANDLE_SIZE / 2
        super().__init__(-half, -half, _HANDLE_SIZE, _HANDLE_SIZE, parent)
        self.role = role
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setBrush(_HANDLE_FILL)
        self.setPen(QPen(_HANDLE_BORDER, 1))
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(
            Qt.CursorShape.SizeFDiagCursor
            if role in ("nw", "se")
            else Qt.CursorShape.SizeBDiagCursor
            if role in ("ne", "sw")
            else Qt.CursorShape.CrossCursor
        )

    def mousePressEvent(self, event) -> None:
        parent = self.parentItem()
        if isinstance(parent, LayerItem):
            parent.handle_press(self.role, event.scenePos())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        parent = self.parentItem()
        if isinstance(parent, LayerItem):
            parent.handle_drag(self.role, event.scenePos(), final=False)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        parent = self.parentItem()
        if isinstance(parent, LayerItem):
            parent.handle_drag(self.role, event.scenePos(), final=True)
        event.accept()


class LayerItem(QGraphicsObject):
    """One canvas layer; paints its (masked) image with the layer blend mode."""

    move_finished = Signal(str, float, float)
    transform_finished = Signal(str, float, float)

    def __init__(self, layer: CanvasLayer, image: QImage, natural_size: tuple[int, int]) -> None:
        super().__init__()
        self._layer = layer
        self._image = image
        self._natural_size = natural_size
        self._handles: list[TransformHandle] = []
        self._drag_start: QPointF | None = None
        self._start_scale = layer.scale
        self._start_rotation = layer.rotation
        self._start_distance = 1.0
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self._build_handles()
        self._position_handles()

    # --- geometry ------------------------------------------------------------

    @property
    def layer(self) -> CanvasLayer:
        return self._layer

    def boundingRect(self) -> QRectF:
        width, height = self._natural_size
        return QRectF(0, 0, width, height)

    @property
    def proxy_size(self) -> tuple[int, int]:
        """The current display proxy size (tests and diagnostics)."""
        return self._image.width(), self._image.height()

    def set_image(self, image: QImage, natural_size: tuple[int, int]) -> None:
        """Replace the displayed proxy (ref, mask, or proxy bound changed)."""
        self.prepareGeometryChange()
        self._image = image
        self._natural_size = natural_size
        self._position_handles()
        self.update()

    def sync(
        self,
        *,
        opacity: float | None = None,
        visible: bool | None = None,
    ) -> None:
        """Re-read the layer's transform and flags from the model."""
        self.setPos(self._layer.x, self._layer.y)
        self.setTransform(local_transform(self._layer, *self._natural_size))
        if opacity is not None:
            self.setOpacity(opacity)
        if visible is not None:
            self.setVisible(visible)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not self._layer.locked)
        self._position_handles()
        self.update()

    # --- painting ------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setCompositionMode(BLEND_TO_QT[self._layer.blend])
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        width, height = self._natural_size
        painter.drawImage(QRectF(0, 0, width, height), self._image)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        if self.isSelected():
            pen = QPen(_OUTLINE, 1)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())

    # --- interaction ---------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._drag_start = QPointF(self.pos())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if self._drag_start is not None:
            start = self._drag_start
            self._drag_start = None
            if self.pos() != start:
                self.move_finished.emit(self._layer.id, self.pos().x(), self.pos().y())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_handle_visibility()
        return super().itemChange(change, value)

    # --- handles -------------------------------------------------------------

    def _build_handles(self) -> None:
        self._handles = [
            TransformHandle("nw", self),
            TransformHandle("ne", self),
            TransformHandle("se", self),
            TransformHandle("sw", self),
            TransformHandle("rotate", self),
        ]

    def _position_handles(self) -> None:
        width, height = self._natural_size
        positions = {
            "nw": QPointF(0, 0),
            "ne": QPointF(width, 0),
            "se": QPointF(width, height),
            "sw": QPointF(0, height),
            "rotate": QPointF(width / 2, -_ROTATION_OFFSET),
        }
        for handle in self._handles:
            handle.setPos(positions[handle.role])
        self._update_handle_visibility()

    def _update_handle_visibility(self) -> None:
        shown = self.isSelected() and not self._layer.locked
        for handle in self._handles:
            handle.setVisible(shown)

    def handle_press(self, role: str, scene_pos: QPointF) -> None:
        center = self.mapToScene(self.boundingRect().center())
        self._start_scale = self._layer.scale
        self._start_rotation = self._layer.rotation
        self._start_distance = max(
            math.hypot(scene_pos.x() - center.x(), scene_pos.y() - center.y()), 1e-6
        )

    def handle_drag(self, role: str, scene_pos: QPointF, *, final: bool) -> None:
        center = self.mapToScene(self.boundingRect().center())
        if role == "rotate":
            angle = math.degrees(math.atan2(scene_pos.y() - center.y(), scene_pos.x() - center.x()))
            scale = self._start_scale
            rotation = angle + 90.0
        else:
            distance = math.hypot(scene_pos.x() - center.x(), scene_pos.y() - center.y())
            scale = self._start_scale * distance / self._start_distance
            scale = max(_MIN_SCALE, min(_MAX_SCALE, scale))
            rotation = self._start_rotation
        if final:
            if (scale, rotation) != (self._start_scale, self._start_rotation):
                self.transform_finished.emit(self._layer.id, scale, rotation)
            return
        self.setTransform(self._transform_for(scale, rotation))

    def _transform_for(self, scale: float, rotation: float) -> QTransform:
        width, height = self._natural_size
        transform = QTransform()
        transform.translate(width / 2, height / 2)
        transform.rotate(rotation)
        transform.scale(
            scale * (-1 if self._layer.flip_h else 1),
            scale * (-1 if self._layer.flip_v else 1),
        )
        transform.translate(-width / 2, -height / 2)
        return transform


class CanvasScene(QGraphicsScene):
    """Scene view of a :class:`CanvasDoc`; re-syncs after every command."""

    move_finished = Signal(str, float, float)
    transform_finished = Signal(str, float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc: CanvasDoc | None = None
        self._store: AssetStore | None = None
        self._items: dict[str, LayerItem] = {}
        self._refs: dict[str, tuple[str, str | None]] = {}
        self._images: dict[tuple[str, str | None], tuple[QImage, tuple[int, int]]] = {}
        self._proxy_long_side = 2048

    # --- document ------------------------------------------------------------

    def set_doc(self, doc: CanvasDoc | None) -> None:
        """Bind the scene to ``doc`` and rebuild every item."""
        self._doc = doc
        self._store = AssetStore(doc.assets_dir) if doc is not None and doc.assets_dir else None
        self._images.clear()
        self._refs.clear()
        self.sync()

    def set_proxy_long_side(self, value: int) -> None:
        """Set the screen proxy bound (long side, §10.7)."""
        if value < 1:
            raise ValueError(f"proxy long side must be positive, got {value}")
        if value != self._proxy_long_side:
            self._proxy_long_side = value
            self._images.clear()
            self._refs.clear()
            self.sync()

    def sync(self) -> None:
        """Create, update, and remove items so the scene matches the model."""
        doc = self._doc
        if doc is None:
            for layer_id in list(self._items):
                self.removeItem(self._items.pop(layer_id))
            self._refs.clear()
            self.update()
            return
        if doc.assets_dir is not None and (
            self._store is None or self._store.assets_dir != doc.assets_dir
        ):
            self._store = AssetStore(doc.assets_dir)
        self.setSceneRect(0, 0, doc.width, doc.height)
        present = {layer.id for layer in doc.layers}
        for layer_id in list(self._items):
            if layer_id not in present:
                removed = self._items.pop(layer_id)
                self.removeItem(removed)
                self._refs.pop(layer_id, None)

        for index, layer in enumerate(doc.layers):
            refs = (layer.image_ref, layer.mask_ref)
            item = self._items.get(layer.id)
            if item is None:
                qimage, natural_size = self._load_qimage(layer)
                item = LayerItem(layer, qimage, natural_size)
                item.move_finished.connect(self.move_finished.emit)
                item.transform_finished.connect(self.transform_finished.emit)
                self.addItem(item)
                self._items[layer.id] = item
                self._refs[layer.id] = refs
            elif self._refs.get(layer.id) != refs:
                qimage, natural_size = self._load_qimage(layer)
                item.set_image(qimage, natural_size)
                self._refs[layer.id] = refs
            item.setZValue(index)
            item.sync(
                opacity=effective_opacity(doc, layer),
                visible=layer_visible(doc, layer),
            )
        self.update()

    # --- painting ------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        color = background_color(self._doc.background) if self._doc is not None else None
        if color is not None:
            painter.fillRect(QRectF(rect).intersected(self.sceneRect()), color)

    # --- queries -------------------------------------------------------------

    def layer_item(self, layer_id: str) -> LayerItem | None:
        return self._items.get(layer_id)

    def selected_layer_ids(self) -> list[str]:
        return [item.layer.id for item in self.selectedItems() if isinstance(item, LayerItem)]

    def select_layer(self, layer_id: str | None) -> None:
        """Select exactly one layer (or clear the selection)."""
        self.clearSelection()
        if layer_id is None:
            return
        item = self._items.get(layer_id)
        if item is not None:
            item.setSelected(True)

    # --- loading -------------------------------------------------------------

    def _load_qimage(self, layer: CanvasLayer) -> tuple[QImage, tuple[int, int]]:
        key = (layer.image_ref, layer.mask_ref)
        cached = self._images.get(key)
        if cached is not None:
            return cached
        if self._store is None:
            raise CanvasFileError("canvas has no assets folder")
        image = load_image(self._store.resolve(layer.image_ref))
        natural_size = image.size
        mask = (
            load_mask(self._store.resolve(layer.mask_ref)) if layer.mask_ref is not None else None
        )
        if mask is not None:
            rgba = image.to_rgba()
            alpha = rgba[:, :, 3].astype(np.uint16)
            rgba[:, :, 3] = (alpha * mask.data // 255).astype(np.uint8)
            image = Image.from_rgba(rgba)
        proxy = image.to_proxy(self._proxy_long_side)
        result = (qimage_from_image(proxy), natural_size)
        self._images[key] = result
        return result

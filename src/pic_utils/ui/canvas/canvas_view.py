"""Canvas viewport: zoom, pan, crop rubber band, and keyboard shortcuts.

The view owns a :class:`~pic_utils.canvas.scene.CanvasScene`; the page wires
the scene's move/transform signals to undo commands. A checkerboard is
painted behind a transparent canvas background.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView, QWidget

from pic_utils.canvas.model import CanvasDoc
from pic_utils.canvas.scene import CanvasScene

__all__ = ["CanvasView"]

_CHECKER_LIGHT = QColor(235, 235, 235)
_CHECKER_DARK = QColor(205, 205, 205)
_CHECKER_SIZE = 16


class CanvasView(QGraphicsView):
    """A zoomable canvas viewport with crop and nudge signals."""

    crop_requested = Signal(QRectF)
    delete_requested = Signal()
    nudge_requested = Signal(float, float)
    zoom_changed = Signal(float)
    selection_changed = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = CanvasScene(self)
        self.setScene(self._scene)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(_CHECKER_DARK)
        self._zoom = 1.0
        self._crop_mode = False
        self._crop_rect: QRectF | None = None
        self.rubberBandChanged.connect(self._on_rubber_band)
        self._scene.selectionChanged.connect(self._emit_selection)

    # --- document ------------------------------------------------------------

    def canvas_scene(self) -> CanvasScene:
        return self._scene

    def set_doc(self, doc: CanvasDoc | None) -> None:
        self._scene.set_doc(doc)
        self.fit_canvas()

    def sync(self) -> None:
        self._scene.sync()

    def select_layer(self, layer_id: str | None) -> None:
        self._scene.select_layer(layer_id)

    def selected_layer_ids(self) -> list[str]:
        return self._scene.selected_layer_ids()

    # --- zoom ----------------------------------------------------------------

    def zoom(self) -> float:
        return self._zoom

    def fit_canvas(self) -> None:
        """Fit the canvas rectangle in the viewport."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self._apply_zoom(1.25)

    def zoom_out(self) -> None:
        self._apply_zoom(0.8)

    def _apply_zoom(self, factor: float) -> None:
        self.scale(factor, factor)
        self._zoom *= factor
        self.zoom_changed.emit(self._zoom)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._apply_zoom(1.25 if event.angleDelta().y() > 0 else 0.8)
            event.accept()
            return
        super().wheelEvent(event)

    # --- crop ----------------------------------------------------------------

    def begin_crop(self) -> None:
        """Enter rubber-band crop mode; the next release emits a scene rect."""
        self._crop_mode = True
        self._crop_rect = None
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def cancel_crop(self) -> None:
        self._crop_mode = False
        self._crop_rect = None
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.unsetCursor()

    def _on_rubber_band(self, rect, from_scene: QPointF, to_scene: QPointF) -> None:
        if self._crop_mode and not rect.isNull():
            self._crop_rect = QRectF(from_scene, to_scene).normalized()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._crop_mode:
            rect = self._crop_rect
            self.cancel_crop()
            if rect is not None and rect.width() >= 1 and rect.height() >= 1:
                self.crop_requested.emit(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # --- keyboard ------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit()
            event.accept()
            return
        step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        if key == Qt.Key.Key_Left:
            self.nudge_requested.emit(-step, 0.0)
        elif key == Qt.Key.Key_Right:
            self.nudge_requested.emit(step, 0.0)
        elif key == Qt.Key.Key_Up:
            self.nudge_requested.emit(0.0, -step)
        elif key == Qt.Key.Key_Down:
            self.nudge_requested.emit(0.0, step)
        else:
            super().keyPressEvent(event)
            return
        event.accept()
        return
        super().keyPressEvent(event)

    # --- painting ------------------------------------------------------------

    def drawBackground(self, painter: QPainter, rect: QRectF | QRect) -> None:
        left = int(rect.left()) - (int(rect.left()) % _CHECKER_SIZE)
        top = int(rect.top()) - (int(rect.top()) % _CHECKER_SIZE)
        painter.fillRect(rect, _CHECKER_LIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_CHECKER_DARK)
        y = top
        while y < rect.bottom():
            x = left + (_CHECKER_SIZE if (y // _CHECKER_SIZE) % 2 else 0)
            while x < rect.right():
                painter.drawRect(QRectF(x, y, _CHECKER_SIZE, _CHECKER_SIZE))
                x += _CHECKER_SIZE * 2
            y += _CHECKER_SIZE
        super().drawBackground(painter, rect)

    # --- selection -----------------------------------------------------------

    def _emit_selection(self) -> None:
        self.selection_changed.emit(self._scene.selected_layer_ids())

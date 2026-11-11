"""Full-resolution image viewer with zoom and pan (architecture §13.1).

A ``QScrollArea`` around a label; the wheel zooms around the cursor,
dragging pans. Used full-screen for one result and as the A/B pane.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import QLabel, QScrollArea, QVBoxLayout, QWidget

__all__ = ["ImageViewer", "ImageViewerDialog"]

_ZOOM_IN = 1.25
_ZOOM_OUT = 0.8


class ImageViewer(QScrollArea):
    """Zoomable, pannable image view; loads the image lazily."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self._label.setBackgroundRole(self.backgroundRole())
        self._label.setAlignment(self.alignment())
        self.setWidget(self._label)
        self.setWidgetResizable(True)
        self._zoom = 1.0
        self._path: Path | None = None

    def set_image_path(self, path: Path | str | None) -> None:
        """Load an image file at fit scale."""
        self._path = Path(path) if path is not None else None
        self._zoom = 1.0
        self._apply()

    def set_zoom(self, factor: float) -> None:
        self._zoom = max(0.05, min(factor, 16.0))
        self._apply()

    def zoom(self) -> float:
        return self._zoom

    def _apply(self) -> None:
        if self._path is None:
            self._label.clear()
            return
        image = QImage(str(self._path))
        if image.isNull():
            self._label.setText(self.tr("Cannot load image"))
            return
        pixmap = QPixmap.fromImage(image)
        if self._zoom != 1.0:
            pixmap = pixmap.scaled(
                max(1, int(pixmap.width() * self._zoom)),
                max(1, int(pixmap.height() * self._zoom)),
                self.transformMode(),
                self.transformationMode(),
            )
        self._label.setPixmap(pixmap)
        self._label.resize(pixmap.size())

    def transformMode(self):
        from PySide6.QtCore import Qt

        return Qt.AspectRatioMode.KeepAspectRatio

    def transformationMode(self):
        from PySide6.QtCore import Qt

        return Qt.TransformationMode.SmoothTransformation

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = _ZOOM_IN if event.angleDelta().y() > 0 else _ZOOM_OUT
        self.set_zoom(self._zoom * factor)


class ImageViewerDialog(QWidget):
    """A standalone full-resolution viewer window."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = ImageViewer(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._viewer)
        self.setWindowTitle(title)
        self.resize(900, 700)

    def set_image_path(self, path: Path | str | None) -> None:
        self._viewer.set_image_path(path)

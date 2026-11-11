"""A/B compare view: two images side by side with a swap (architecture §13.1).

Used by the results grid to compare two results. Images load lazily on the
global thread pool and are converted to ``QPixmap`` on the GUI thread
(§13.3); while loading, a placeholder is shown.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

__all__ = ["CompareView"]

_PREVIEW_SIZE = 320


class _LoaderSignals(QObject):
    loaded = Signal(str, QPixmap)


class _ImageLoader(QRunnable):
    """Loads one image scaled to the preview size, off the GUI thread."""

    _signals = _LoaderSignals()

    def __init__(self, path: str, size: int = _PREVIEW_SIZE) -> None:
        super().__init__()
        self._path = path
        self._size = size

    def run(self) -> None:
        image = QImage(self._path)
        if image.isNull():
            return
        pixmap = QPixmap.fromImage(
            image.scaled(
                self._size,
                self._size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._signals.loaded.emit(self._path, pixmap)


class CompareView(QWidget):
    """Two labelled images side by side with a swap button (FR-UI-06)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._left = QLabel(self)
        self._right = QLabel(self)
        self._left_label = QLabel(self)
        self._right_label = QLabel(self)
        self._swap = QPushButton(self)
        self._swap.clicked.connect(self._swap_images)

        for label in (self._left, self._right):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)

        left_box = QVBoxLayout()
        left_box.addWidget(self._left_label)
        left_box.addWidget(self._left)
        right_box = QVBoxLayout()
        right_box.addWidget(self._right_label)
        right_box.addWidget(self._right)

        row = QHBoxLayout()
        row.addLayout(left_box, 1)
        row.addLayout(right_box, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self._swap, 0, Qt.AlignmentFlag.AlignCenter)

        self._paths: tuple[str, str] = ("", "")
        self._swapped = False
        self.retranslate()

    # --- public API ---------------------------------------------------------

    def set_images(self, left_path: str, right_path: str) -> None:
        """Show two images; loads them lazily off the GUI thread."""
        self._paths = (left_path, right_path)
        self._swapped = False
        self._load(self._left, left_path)
        self._load(self._right, right_path)
        self._update_labels()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._swap.setText(self.tr("Swap"))
        self._update_labels()

    # --- internals ------------------------------------------------------------

    def _load(self, label: QLabel, path: str) -> None:
        if not path:
            label.setText(self.tr("No image"))
            return
        loader = _ImageLoader(path)
        loader._signals.loaded.connect(self._on_loaded)
        QThreadPool.globalInstance().start(loader)

    def _on_loaded(self, path: str, pixmap: QPixmap) -> None:
        target = self._right if path == self._paths[1] else self._left
        if self._swapped:
            target = self._left if target is self._right else self._right
        target.setPixmap(pixmap)

    def _swap_images(self) -> None:
        self._swapped = not self._swapped
        left_pixmap = self._left.pixmap()
        right_pixmap = self._right.pixmap()
        self._left.setPixmap(right_pixmap)
        self._right.setPixmap(left_pixmap)
        self._update_labels()

    def _update_labels(self) -> None:
        left, right = self._paths
        if self._swapped:
            left, right = right, left
        self._left_label.setText(Path(left).name if left else self.tr("A"))
        self._right_label.setText(Path(right).name if right else self.tr("B"))

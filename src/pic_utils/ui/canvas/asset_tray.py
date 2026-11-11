"""Asset tray: accepted results and image files available as layers (§13.1).

The tray lists accepted results from every run (via ``AcceptedIndex``) and
lets the user add an arbitrary image file. Double-click or the Add button
emits the source path; the page copies it into the canvas asset store and
pushes an ``AddLayerCommand``. Thumbnails decode on ``QThreadPool`` and are
converted to ``QPixmap`` on the GUI thread (§13.3).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pic_utils.io import formats
from pic_utils.results.store import AcceptedIndex

__all__ = ["AssetTray"]

_ROLE_PATH = Qt.ItemDataRole.UserRole
_ROLE_THUMB = Qt.ItemDataRole.UserRole + 1
_ICON_SIZE = 96


class _ThumbnailSignals(QObject):
    loaded = Signal(str, QImage)
    failed = Signal(str)


class _ThumbnailLoader(QRunnable):
    """Decode one thumbnail off the GUI thread and emit a ``QImage``."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        self._signals = _ThumbnailSignals()

    def signals(self) -> _ThumbnailSignals:
        return self._signals

    def run(self) -> None:
        image = QImage(self._path)
        if image.isNull():
            self._signals.failed.emit(self._path)
            return
        scaled = image.scaled(
            _ICON_SIZE,
            _ICON_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._signals.loaded.emit(self._path, scaled)


class AssetTray(QWidget):
    """Accepted-result thumbnails plus an "Add file…" action."""

    add_requested = Signal(object)  # Path

    def __init__(self, index: AcceptedIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
        self._pixmaps: dict[str, QPixmap] = {}
        self._pending: set[str] = set()
        self._list = QListWidget(self)
        self._list.setViewMode(QListView.ViewMode.IconMode)
        self._list.setResizeMode(QListView.ResizeMode.Adjust)
        self._list.setIconSize(QSize(_ICON_SIZE, _ICON_SIZE))
        self._list.setSpacing(6)
        self._list.setUniformItemSizes(True)
        self._list.itemDoubleClicked.connect(self._add_item)

        self._add = QPushButton(self)
        self._add.clicked.connect(self._add_current)
        self._add_file = QPushButton(self)
        self._add_file.clicked.connect(self._add_file_dialog)
        self._refresh = QPushButton(self)
        self._refresh.clicked.connect(lambda: self.refresh())

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self._add)
        buttons.addWidget(self._add_file)
        buttons.addWidget(self._refresh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list, 1)
        layout.addLayout(buttons)

        self.refresh()
        self.retranslate()

    # --- public API ----------------------------------------------------------

    def index(self) -> AcceptedIndex:
        return self._index

    def refresh(self, *, rescan: bool = True) -> None:
        """Rebuild the thumbnails; ``rescan`` re-reads the results root."""
        if rescan:
            self._index.refresh()
        self._list.clear()
        for accepted in self._index.results():
            record = accepted.record
            path = self._index.results_root / accepted.run_id / record.path
            if not path.is_file():
                continue
            item = QListWidgetItem(f"{accepted.run_id}\n{record.id}")
            item.setData(_ROLE_PATH, str(path))
            thumb = self._index.results_root / accepted.run_id / record.thumb
            if thumb.is_file():
                thumb_path = str(thumb)
                item.setData(_ROLE_THUMB, thumb_path)
                pixmap = self._pixmaps.get(thumb_path)
                if pixmap is not None:
                    item.setIcon(QIcon(pixmap))
                elif thumb_path not in self._pending:
                    self._pending.add(thumb_path)
                    loader = _ThumbnailLoader(thumb_path)
                    loader.signals().loaded.connect(self._on_thumb_loaded)
                    loader.signals().failed.connect(self._on_thumb_failed)
                    QThreadPool.globalInstance().start(loader)
            self._list.addItem(item)

    def current_path(self) -> Path | None:
        item = self._list.currentItem()
        if item is None:
            return None
        raw = item.data(_ROLE_PATH)
        return Path(str(raw)) if raw else None

    # --- thumbnails ----------------------------------------------------------

    def _on_thumb_loaded(self, thumb_path: str, image: QImage) -> None:
        self._pending.discard(thumb_path)
        pixmap = QPixmap.fromImage(image)
        self._pixmaps[thumb_path] = pixmap
        icon = QIcon(pixmap)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.data(_ROLE_THUMB) == thumb_path:
                item.setIcon(icon)

    def _on_thumb_failed(self, thumb_path: str) -> None:
        self._pending.discard(thumb_path)

    # --- actions -------------------------------------------------------------

    def _add_item(self, item: QListWidgetItem) -> None:
        raw = item.data(_ROLE_PATH)
        if raw:
            self.add_requested.emit(Path(str(raw)))

    def _add_current(self) -> None:
        path = self.current_path()
        if path is not None:
            self.add_requested.emit(path)

    @staticmethod
    def image_pattern() -> str:
        """The file-dialog pattern for formats the loader actually supports."""
        return " ".join(f"*{suffix}" for suffix in sorted(formats.SUPPORTED_EXTENSIONS))

    def _add_file_dialog(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Add image"),
            "",
            self.tr("Images ({patterns})").replace("{patterns}", self.image_pattern()),
        )
        if chosen:
            self.add_requested.emit(Path(chosen))

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._add.setText(self.tr("Add layer"))
        self._add_file.setText(self.tr("Add file…"))
        self._refresh.setText(self.tr("Refresh"))

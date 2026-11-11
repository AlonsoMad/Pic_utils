"""Asset tray: accepted results and image files available as layers (§13.1).

The tray lists accepted results from every run (via ``AcceptedIndex``) and
lets the user add an arbitrary image file. Double-click or the Add button
emits the source path; the page copies it into the canvas asset store and
pushes an ``AddLayerCommand``.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
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

from pic_utils.results.store import AcceptedIndex
from pic_utils.ui.lab.project_actions import IMAGE_SUFFIXES

__all__ = ["AssetTray"]

_ROLE_PATH = Qt.ItemDataRole.UserRole
_ICON_SIZE = 96


class AssetTray(QWidget):
    """Accepted-result thumbnails plus an "Add file…" action."""

    add_requested = Signal(object)  # Path

    def __init__(self, index: AcceptedIndex, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._index = index
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
        self._refresh.clicked.connect(self.refresh)

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

    def refresh(self) -> None:
        """Rescan accepted results and rebuild the thumbnails."""
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
                pixmap = QPixmap(str(thumb))
                if not pixmap.isNull():
                    item.setIcon(QIcon(pixmap))
            self._list.addItem(item)

    def current_path(self) -> Path | None:
        item = self._list.currentItem()
        if item is None:
            return None
        raw = item.data(_ROLE_PATH)
        return Path(str(raw)) if raw else None

    # --- actions -------------------------------------------------------------

    def _add_item(self, item: QListWidgetItem) -> None:
        raw = item.data(_ROLE_PATH)
        if raw:
            self.add_requested.emit(Path(str(raw)))

    def _add_current(self) -> None:
        path = self.current_path()
        if path is not None:
            self.add_requested.emit(path)

    def _add_file_dialog(self) -> None:
        pattern = " ".join(f"*{suffix}" for suffix in sorted(IMAGE_SUFFIXES))
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Add image"),
            "",
            self.tr("Images ({patterns})").replace("{patterns}", pattern),
        )
        if chosen:
            self.add_requested.emit(Path(chosen))

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._add.setText(self.tr("Add layer"))
        self._add_file.setText(self.tr("Add file…"))
        self._refresh.setText(self.tr("Refresh"))

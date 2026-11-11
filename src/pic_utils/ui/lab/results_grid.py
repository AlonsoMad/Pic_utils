"""Results grid: curation over one run's results (architecture §9.4, §13.1-§13.2).

``ResultsGridModel`` lists one run's ``ResultRecord`` rows with their
curation state; thumbnails load lazily on a ``QThreadPool`` (§13.3) and are
converted to ``QPixmap`` on the GUI thread. Accept/discard/restore move files
through the session trash and record the decision atomically (§9.3: move
first, then record). Not undoable (FR-UNDO-04).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractListModel,
    QModelIndex,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QListView, QVBoxLayout, QWidget

from pic_utils.results.trash import SessionTrash
from pic_utils.runner.manifest import ResultRecord, RunStore

if TYPE_CHECKING:
    pass

__all__ = ["ResultsGrid", "ResultsGridModel", "ThumbnailLoader"]

ROLE_RECORD = Qt.ItemDataRole.UserRole
ROLE_CURATION = Qt.ItemDataRole.UserRole + 1
ROLE_THUMB = Qt.ItemDataRole.UserRole + 2

_THUMB_SIZE = 128


class ThumbnailLoader(QRunnable):
    """Loads one thumbnail off the GUI thread (§13.3)."""

    class Signals(QObject):
        loaded = Signal(str, QPixmap)  # thumb path, pixmap

    _signals = Signals()

    @classmethod
    def loaded_signal(cls) -> Signal:
        return cls._signals.loaded

    def __init__(self, thumb_path: str, size: int = _THUMB_SIZE) -> None:
        super().__init__()
        self._thumb_path = thumb_path
        self._size = size

    def run(self) -> None:
        image = QImage(self._thumb_path)
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
        self._signals.loaded.emit(self._thumb_path, pixmap)


class ResultsGridModel(QAbstractListModel):
    """One run's results with curation state; thumbnails load lazily."""

    curation_changed = Signal()

    def __init__(
        self,
        store: RunStore,
        trash: SessionTrash,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._trash = trash
        self._records: list[ResultRecord] = []
        self._curation: dict[str, str] = {}
        self._pixmaps: dict[str, QPixmap] = {}
        self._pending: set[str] = set()
        self._show_discarded = False
        self.reload()

    # --- data ----------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the run from disk."""
        self.beginResetModel()
        data = self._store.read()
        self._records = list(data.results)
        self._curation = dict(data.curation)
        self.endResetModel()
        self.curation_changed.emit()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        if not self._show_discarded:
            return sum(1 for record in self._records if self._state(record.id) != "discarded")
        return len(self._records)

    def _visible_records(self) -> list[ResultRecord]:
        if self._show_discarded:
            return self._records
        return [r for r in self._records if self._state(r.id) != "discarded"]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        records = self._visible_records()
        if not 0 <= index.row() < len(records):
            return None
        record = records[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return record.id
        if role == ROLE_RECORD:
            return record
        if role == ROLE_CURATION:
            return self._state(record.id)
        if role == ROLE_THUMB:
            return self._pixmap(record.thumb)
        if role == Qt.ItemDataRole.ToolTipRole:
            return f"seed {record.seed}"
        return None

    def _state(self, result_id: str) -> str | None:
        return self._curation.get(result_id)

    def _pixmap(self, thumb_path: str) -> QPixmap | None:
        pixmap = self._pixmaps.get(thumb_path)
        if pixmap is not None:
            return pixmap
        if thumb_path not in self._pending:
            self._pending.add(thumb_path)
            loader = ThumbnailLoader(str(self._store.run_dir / thumb_path))
            loader.loaded_signal().connect(self._on_thumb_loaded)
            QThreadPool.globalInstance().start(loader)
        return None

    def _on_thumb_loaded(self, thumb_path: str, pixmap: QPixmap) -> None:
        self._pixmaps[thumb_path] = pixmap
        for row, record in enumerate(self._visible_records()):
            if record.thumb == thumb_path:
                index = self.index(row, 0)
                self.dataChanged.emit(index, index)

    # --- curation (§9.3: move first, then record) ------------------------------

    def accept(self, row: int) -> None:
        record = self._record_at(row)
        if record is None:
            return
        if self._state(record.id) == "discarded":
            self._trash.restore(self._store.run_id, [record.path, record.thumb])
        self._store.set_curation(record.id, "accepted")
        self._curation[record.id] = "accepted"
        self._refresh_row(row)
        self.curation_changed.emit()

    def discard(self, row: int) -> None:
        record = self._record_at(row)
        if record is None:
            return
        self._trash.discard(self._store.run_id, [record.path, record.thumb])
        self._store.set_curation(record.id, "discarded")
        self._curation[record.id] = "discarded"
        self._refresh_row(row)
        self.curation_changed.emit()

    def restore(self, row: int) -> None:
        record = self._record_at(row)
        if record is None:
            return
        self._trash.restore(self._store.run_id, [record.path, record.thumb])
        self._store.clear_curation(record.id)
        self._curation.pop(record.id, None)
        self._refresh_row(row)
        self.curation_changed.emit()

    def _record_at(self, row: int) -> ResultRecord | None:
        records = self._visible_records()
        return records[row] if 0 <= row < len(records) else None

    def _refresh_row(self, row: int) -> None:
        self.beginResetModel()
        self.endResetModel()

    # --- view options -----------------------------------------------------------

    def set_show_discarded(self, show: bool) -> None:
        self.beginResetModel()
        self._show_discarded = show
        self.endResetModel()

    @property
    def show_discarded(self) -> bool:
        return self._show_discarded


class ResultsGrid(QWidget):
    """Thumbnail grid with curation shortcuts and a discarded toggle (§13.1)."""

    #: (result_id, record) — the user wants the full-resolution view.
    view_requested = Signal(str, object)
    #: (left_id, right_id) — the user wants an A/B comparison.
    compare_requested = Signal(str, str)

    def __init__(
        self,
        store: RunStore,
        trash: SessionTrash,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = ResultsGridModel(store, trash, self)

        self._view = QListView(self)
        self._view.setModel(self._model)
        self._view.setViewMode(QListView.ViewMode.IconMode)
        self._view.setResizeMode(QListView.ResizeMode.Adjust)
        self._view.setUniformItemSizes(True)
        self._view.setSpacing(6)
        self._view.doubleClicked.connect(self._on_double_clicked)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._view.activated.connect(self._on_double_clicked)

    # --- public API ---------------------------------------------------------

    def model(self) -> ResultsGridModel:
        return self._model

    def current_record(self) -> ResultRecord | None:
        index = self._view.currentIndex()
        if not index.isValid():
            return None
        return self._model.data(index, ROLE_RECORD)  # type: ignore[return-value]

    # --- curation shortcuts (FR-UI-06) ---------------------------------------

    def accept_current(self) -> None:
        self._model.accept(self._view.currentIndex().row())

    def discard_current(self) -> None:
        self._model.discard(self._view.currentIndex().row())

    def restore_current(self) -> None:
        self._model.restore(self._view.currentIndex().row())

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_A:
            self.accept_current()
        elif key == Qt.Key.Key_D:
            self.discard_current()
        elif key == Qt.Key.Key_R:
            self.restore_current()
        else:
            super().keyPressEvent(event)

    # --- internals ------------------------------------------------------------

    def _on_double_clicked(self, index: QModelIndex) -> None:
        record = self._model.data(index, ROLE_RECORD)
        if record is not None:
            self.view_requested.emit(record.id, record)

"""Block library panel: search, categories, reload, auto-reload (§5.5, §13.2).

The model lists every block in the registry snapshot, grouped by category
(built-in categories are translated through a fixed mapping; user categories
are shown verbatim, §15). A ``QSortFilterProxyModel`` applies the search text
and the category filter. Reload errors are reported non-modally and
deduplicated per file+reason until a reload succeeds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import (
    QAbstractListModel,
    QFileSystemWatcher,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    QTimer,
    Qt,
    Signal,
)
from PySide6.QtWidgets import QComboBox, QListView, QLineEdit, QPushButton, QVBoxLayout, QWidget

from pic_utils.blocks.api import BlockDef, BlockLoadError
from pic_utils.blocks.registry import BlockRegistry, ReloadReport

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["CATEGORY_LABELS", "BlockLibraryModel", "BlockLibraryPanel", "BlockLibraryProxy"]

ROLE_BLOCK_DEF = Qt.ItemDataRole.UserRole
ROLE_CATEGORY = Qt.ItemDataRole.UserRole + 1
ROLE_DESCRIPTION = Qt.ItemDataRole.UserRole + 2
ROLE_OUTPUT = Qt.ItemDataRole.UserRole + 3

_ALL_CATEGORIES = "—all—"
_AUTO_RELOAD_DEBOUNCE_MS = 500

#: Built-in category keys → English source strings, translated at call time
#: (§15). Categories outside this mapping are shown verbatim.
CATEGORY_LABELS: dict[str, Callable[[], str]] = {
    "channels": lambda: _translate("Channels"),
    "compose": lambda: _translate("Compose"),
    "masks": lambda: _translate("Masks"),
    "segment": lambda: _translate("Segment"),
    "examples": lambda: _translate("Examples"),
}


def _translate(source: str) -> str:
    from PySide6.QtCore import QCoreApplication

    return QCoreApplication.translate("BlockLibraryPanel", source)


def _category_label(category: str) -> str:
    factory = CATEGORY_LABELS.get(category)
    return factory() if factory is not None else category


class _Row:
    __slots__ = ("definition",)

    def __init__(self, definition: BlockDef) -> None:
        self.definition = definition


class BlockLibraryModel(QAbstractListModel):
    """All blocks in the current snapshot; one row per block."""

    def __init__(self, registry: BlockRegistry, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._rows: list[_Row] = []
        self.rebuild()

    # --- QAbstractListModel -------------------------------------------------

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not 0 <= index.row() < len(self._rows):
            return None
        definition = self._rows[index.row()].definition
        if role == Qt.ItemDataRole.DisplayRole:
            return definition.name
        if role == ROLE_BLOCK_DEF:
            return definition
        if role == ROLE_CATEGORY:
            return _category_label(definition.category)
        if role == ROLE_DESCRIPTION:
            return definition.description
        if role == ROLE_OUTPUT:
            return definition.output
        return None

    # --- snapshot sync ---------------------------------------------------------

    def rebuild(self) -> None:
        """Re-read the registry snapshot and refresh the list."""
        self.beginResetModel()
        snapshot = self._registry.snapshot()
        self._rows = [_Row(snapshot.get(name)) for name in sorted(snapshot)]
        self.endResetModel()

    def categories(self) -> list[str]:
        """Distinct display labels, sorted."""
        seen: dict[str, str] = {}
        for row in self._rows:
            seen.setdefault(row.definition.category, _category_label(row.definition.category))
        return sorted(seen.values())


class BlockLibraryProxy(QSortFilterProxyModel):
    """Filters by case-insensitive search text and category."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._search = ""
        self._category: str | None = None

    def set_search(self, text: str) -> None:
        self.beginFilterChange()
        self._search = text.casefold()
        self.endFilterChange()

    def set_category_label(self, label: str | None) -> None:
        self.beginFilterChange()
        self._category = label
        self.endFilterChange()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        assert isinstance(model, BlockLibraryModel)
        index = model.index(source_row, 0, source_parent)
        definition = model.data(index, ROLE_BLOCK_DEF)
        assert isinstance(definition, BlockDef)
        if self._category is not None and _category_label(definition.category) != self._category:
            return False
        if not self._search:
            return True
        haystack = f"{definition.name} {definition.description}".casefold()
        return self._search in haystack


class BlockLibraryPanel(QWidget):
    """Search box, category filter, block list, reload action, auto-reload."""

    #: A block was chosen (double-click or Enter) for the pipeline editor.
    block_activated = Signal(str)
    #: Non-modal reload errors, deduplicated (§13.6).
    reload_errors = Signal(list)  # list[BlockLoadError]
    #: A reload finished without new errors.
    reload_succeeded = Signal()

    def __init__(self, registry: BlockRegistry, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry = registry
        self._model = BlockLibraryModel(registry, self)
        self._proxy = BlockLibraryProxy(self)
        self._proxy.setSourceModel(self._model)

        self._search = QLineEdit(self)
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._proxy.set_search)

        self._category = QComboBox(self)
        self._category.currentTextChanged.connect(self._proxy.set_category_label)

        self._view = QListView(self)
        self._view.setModel(self._proxy)
        self._view.setUniformItemSizes(True)
        self._view.activated.connect(self._on_activated)
        self._view.doubleClicked.connect(self._on_activated)

        self._reload = QPushButton(self)
        self._reload.clicked.connect(self._manual_reload)

        self._watcher = QFileSystemWatcher(self)
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_AUTO_RELOAD_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._auto_reload)
        self._watcher.directoryChanged.connect(lambda _path: self._debounce.start())

        layout = QVBoxLayout(self)
        layout.addWidget(self._search)
        layout.addWidget(self._category)
        layout.addWidget(self._view)
        layout.addWidget(self._reload)

        self._reported_errors: set[tuple[str, str]] = set()
        registry.changed.connect(self._on_registry_changed)
        self._refresh_categories()
        self._watch_user_dir()
        self.retranslate()

    # --- public API ----------------------------------------------------------

    def block_def(self, proxy_index: QModelIndex) -> BlockDef | None:
        source_index = self._proxy.mapToSource(proxy_index)
        definition = self._model.data(source_index, ROLE_BLOCK_DEF)
        return definition if isinstance(definition, BlockDef) else None

    def reload_now(self) -> ReloadReport:
        """Manual reload; returns the report for callers that log it."""
        report = self._registry.reload()
        self._finish_reload(report)
        return report

    def set_auto_reload(self, enabled: bool) -> None:
        """Watch (or stop watching) the user blocks folder (§16)."""
        folders = self._watcher.directories()
        if folders:
            self._watcher.removePaths(folders)
        if enabled:
            self._watch_user_dir()

    # --- internals ------------------------------------------------------------

    def _on_registry_changed(self, _snapshot: object) -> None:
        self._model.rebuild()
        self._refresh_categories()

    def _refresh_categories(self) -> None:
        current = self._category.currentText()
        self._category.blockSignals(True)
        self._category.clear()
        self._category.addItem(_ALL_CATEGORIES)
        self._category.addItems(self._model.categories())
        index = self._category.findText(current)
        self._category.setCurrentIndex(max(index, 0))
        self._category.blockSignals(False)

    def _on_activated(self, index: QModelIndex) -> None:
        definition = self.block_def(index)
        if definition is not None:
            self.block_activated.emit(definition.name)

    def _manual_reload(self) -> None:
        self._finish_reload(self._registry.reload())

    def _finish_reload(self, report: ReloadReport) -> None:
        """Report only the errors not already shown (§13.6 deduplication)."""
        if not report.errors:
            self._reported_errors.clear()
            self.reload_succeeded.emit()
            return
        fresh = [error for error in report.errors if self._error_key(error) not in self._reported_errors]
        for error in fresh:
            self._reported_errors.add(self._error_key(error))
        if fresh:
            self.reload_errors.emit(fresh)

    @staticmethod
    def _error_key(error: BlockLoadError) -> tuple[str, str]:
        return (str(getattr(error, "path", "")), getattr(error, "reason", str(error)))

    def _watch_user_dir(self) -> None:
        folder = self._registry.user_blocks_dir
        if folder is not None and folder.is_dir():
            self._watcher.addPath(str(folder))

    def _auto_reload(self) -> None:
        """Debounced reload after a file change (§5.5: within ~0.5 s)."""
        self._finish_reload(self._registry.reload())

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._search.setPlaceholderText(self.tr("Search blocks…"))
        self._reload.setText(self.tr("Reload blocks"))

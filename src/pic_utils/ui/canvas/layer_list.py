"""Layer list: reverse z-order display, visibility, drag reorder, groups (§13.2).

The model shows the top layer first, like every compositor. Drag-and-drop
emits a document-index move request; the page pushes ``MoveLayerCommand``.
Grouping and ungrouping are buttons over the current selection.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import (
    QAbstractListModel,
    QMimeData,
    QModelIndex,
    QPersistentModelIndex,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pic_utils.canvas.model import CanvasDoc, CanvasLayer

__all__ = ["LayerListModel", "LayerListPanel"]

ROLE_LAYER_ID = Qt.ItemDataRole.UserRole
_MIME_TYPE = "application/x-pic-utils-layers"


class LayerListModel(QAbstractListModel):
    """Rows = layers in reverse z-order; the document list stays the truth."""

    visibility_toggled = Signal(str, bool)
    move_requested = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._doc: CanvasDoc | None = None

    @property
    def doc(self) -> CanvasDoc | None:
        return self._doc

    def set_doc(self, doc: CanvasDoc | None) -> None:
        self.beginResetModel()
        self._doc = doc
        self.endResetModel()

    def reload(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _layer_at(self, row: int) -> CanvasLayer | None:
        if self._doc is None or not 0 <= row < len(self._doc.layers):
            return None
        return self._doc.layers[len(self._doc.layers) - 1 - row]

    def layer_id_at(self, row: int) -> str | None:
        layer = self._layer_at(row)
        return layer.id if layer is not None else None

    def row_for(self, layer_id: str) -> int:
        if self._doc is None:
            return -1
        index = self._doc.layer_index(layer_id)
        return -1 if index < 0 else len(self._doc.layers) - 1 - index

    # --- QAbstractListModel --------------------------------------------------

    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._doc.layers) if self._doc is not None else 0

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        layer = self._layer_at(index.row())
        if layer is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return layer.name
        if role == ROLE_LAYER_ID:
            return layer.id
        if role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked
        if role == Qt.ItemDataRole.ToolTipRole:
            group = self._doc.groups.get(layer.group_id) if self._doc and layer.group_id else None
            return group.name if group is not None else ""
        return None

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        if not index.isValid():
            return base | Qt.ItemFlag.ItemIsDropEnabled
        return (
            base
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )

    def setData(
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: object,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if role != Qt.ItemDataRole.CheckStateRole:
            return False
        layer = self._layer_at(index.row())
        if layer is None:
            return False
        visible = Qt.CheckState(value) == Qt.CheckState.Checked
        self.visibility_toggled.emit(layer.id, visible)
        return True

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [_MIME_TYPE]

    def mimeData(self, indexes: Sequence[QModelIndex]) -> QMimeData | None:  # type: ignore[override]
        rows = sorted(index.row() for index in indexes if index.isValid())
        if not rows:
            return None
        mime = QMimeData()
        mime.setData(_MIME_TYPE, ",".join(str(row) for row in rows).encode("ascii"))
        return mime

    def dropMimeData(
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        _column: int,
        _parent: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        if action != Qt.DropAction.MoveAction or self._doc is None:
            return False
        raw = bytes(data.data(_MIME_TYPE).data()).decode("ascii", errors="ignore")
        if not raw:
            return False
        try:
            source_rows = [int(part) for part in raw.split(",")]
        except ValueError:
            return False
        if len(source_rows) != 1:
            return False
        source_display = source_rows[0]
        count = self.rowCount()
        if not 0 <= source_display < count:
            return False
        target_display = count if row < 0 else row
        display = list(range(count))
        item = display.pop(source_display)
        if target_display > source_display:
            target_display -= 1
        display.insert(target_display, item)
        source_doc = count - 1 - source_display
        target_doc = list(reversed(display)).index(item)
        if source_doc != target_doc:
            self.move_requested.emit(source_doc, target_doc)
        return False


class LayerListPanel(QWidget):
    """List view plus z-order and grouping buttons."""

    layer_selected = Signal(object)  # str | None
    visibility_toggled = Signal(str, bool)
    move_requested = Signal(int, int)
    group_requested = Signal(list)
    ungroup_requested = Signal(str)
    delete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = LayerListModel(self)
        self._view = QListView(self)
        self._view.setModel(self._model)
        self._view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self._view.setDragDropMode(QListView.DragDropMode.InternalMove)
        self._view.setDragEnabled(True)
        self._view.setAcceptDrops(True)
        self._view.setDropIndicatorShown(True)
        self._view.selectionModel().currentChanged.connect(self._on_current_changed)

        self._up = QPushButton(self)
        self._up.clicked.connect(self._move_up)
        self._down = QPushButton(self)
        self._down.clicked.connect(self._move_down)
        self._group = QPushButton(self)
        self._group.clicked.connect(self._group_selection)
        self._ungroup = QPushButton(self)
        self._ungroup.clicked.connect(self._ungroup_selection)
        self._delete = QPushButton(self)
        self._delete.clicked.connect(self.delete_requested.emit)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        for button in (self._up, self._down, self._group, self._ungroup, self._delete):
            buttons.addWidget(button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view, 1)
        layout.addLayout(buttons)

        self._model.visibility_toggled.connect(self.visibility_toggled.emit)
        self._model.move_requested.connect(self.move_requested.emit)
        self.retranslate()

    # --- document ------------------------------------------------------------

    def model(self) -> LayerListModel:
        return self._model

    def set_doc(self, doc: CanvasDoc | None) -> None:
        self._model.set_doc(doc)

    def reload(self) -> None:
        self._model.reload()

    def layer_id_at(self, row: int) -> str | None:
        return self._model.layer_id_at(row)

    def selected_layer_ids(self) -> list[str]:
        rows = self._view.selectionModel().selectedRows()
        return [
            layer_id for row in rows if (layer_id := self._model.layer_id_at(row.row())) is not None
        ]

    def selected_group_id(self) -> str | None:
        if self._model.doc is None:
            return None
        selected = self.selected_layer_ids()
        if not selected:
            return None
        groups = {
            layer.group_id
            for layer_id in selected
            if (layer := self._model.doc.find_layer(layer_id)) is not None
        }
        if len(groups) == 1:
            group_id = next(iter(groups))
            return group_id if isinstance(group_id, str) else None
        return None

    def select_layer(self, layer_id: str | None) -> None:
        if layer_id is None:
            self._view.clearSelection()
            self._view.setCurrentIndex(QModelIndex())
            return
        row = self._model.row_for(layer_id)
        if row < 0:
            return
        index = self._model.index(row, 0)
        self._view.setCurrentIndex(index)
        self._view.scrollTo(index)

    # --- actions -------------------------------------------------------------

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        layer_id = self._model.layer_id_at(current.row()) if current.isValid() else None
        self.layer_selected.emit(layer_id)

    def _move_up(self) -> None:
        selected = self.selected_layer_ids()
        if len(selected) != 1 or self._model.doc is None:
            return
        index = self._model.doc.layer_index(selected[0])
        if 0 <= index < len(self._model.doc.layers) - 1:
            self.move_requested.emit(index, index + 1)

    def _move_down(self) -> None:
        selected = self.selected_layer_ids()
        if len(selected) != 1 or self._model.doc is None:
            return
        index = self._model.doc.layer_index(selected[0])
        if index > 0:
            self.move_requested.emit(index, index - 1)

    def _group_selection(self) -> None:
        selected = self.selected_layer_ids()
        if selected:
            self.group_requested.emit(selected)

    def _ungroup_selection(self) -> None:
        group_id = self.selected_group_id()
        if group_id is not None:
            self.ungroup_requested.emit(group_id)

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._up.setText(self.tr("Up"))
        self._down.setText(self.tr("Down"))
        self._group.setText(self.tr("Group"))
        self._ungroup.setText(self.tr("Ungroup"))
        self._delete.setText(self.tr("Delete"))

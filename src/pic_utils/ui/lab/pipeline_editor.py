"""Pipeline editor: step list, reorder, bypass, inline validation (§6.3, §13.2).

``PipelineModel`` owns the ``Pipeline`` value and is the single source of
truth; undo commands (``ui/lab/commands.py``) call its raw mutators. Every
mutation re-validates against the live block snapshot and repaints the rows
with validity icons; the panel shows the full issue list inline and disables
launch-relevant actions while any error exists.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import (
    QAbstractListModel,
    QMimeData,
    QModelIndex,
    QObject,
    Qt,
    Signal,
)
from PySide6.QtGui import QBrush, QColor, QUndoStack
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from pic_utils.pipeline.model import Pipeline, PipelineStep
from pic_utils.pipeline.validate import Issue, validate_pipeline
from pic_utils.ui.lab.commands import (
    AddStepCommand,
    BypassStepCommand,
    LoadPipelineCommand,
    MoveStepCommand,
    RemoveStepCommand,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pic_utils.blocks.registry import BlockSnapshot

__all__ = ["PipelineEditorPanel", "PipelineModel"]

ROLE_STEP_ID = Qt.ItemDataRole.UserRole
ROLE_BLOCK_NAME = Qt.ItemDataRole.UserRole + 1
ROLE_BYPASS = Qt.ItemDataRole.UserRole + 2
ROLE_ISSUES = Qt.ItemDataRole.UserRole + 3

_MIME_TYPE = "application/x-pic-utils-steps"
_ERROR_COLOR = QColor(200, 60, 60)
_WARNING_COLOR = QColor(190, 140, 0)


class PipelineModel(QAbstractListModel):
    """Rows = pipeline steps; every mutation re-validates (§13.2)."""

    pipeline_changed = Signal(object)
    blocking_changed = Signal(bool)

    def __init__(
        self,
        snapshot_provider: Callable[[], BlockSnapshot],
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._snapshot_provider = snapshot_provider
        self._steps: tuple[PipelineStep, ...] = ()
        self._issues: list[Issue] = []
        self.second_image_available = False
        self.sources_available = False
        self._validate()

    # --- state ---------------------------------------------------------------

    @property
    def pipeline(self) -> Pipeline:
        return Pipeline(self._steps)

    @property
    def issues(self) -> list[Issue]:
        return list(self._issues)

    def has_blocking_issues(self) -> bool:
        return any(issue.severity == "error" for issue in self._issues)

    def snapshot(self) -> BlockSnapshot:
        return self._snapshot_provider()

    def revalidate(self) -> None:
        """Re-run validation (e.g. after a block reload) and repaint."""
        self._validate()

    # --- QAbstractListModel ------------------------------------------------------

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._steps)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not 0 <= index.row() < len(self._steps):
            return None
        step = self._steps[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return step.block
        if role == ROLE_STEP_ID:
            return step.id
        if role == ROLE_BLOCK_NAME:
            return step.block
        if role == ROLE_BYPASS:
            return step.bypass
        if role == ROLE_ISSUES:
            return [issue for issue in self._issues if issue.step_id == step.id]
        if role == Qt.ItemDataRole.ForegroundRole:
            worst = self._worst_severity(step.id)
            if worst == "error":
                return QBrush(_ERROR_COLOR)
            if worst == "warning":
                return QBrush(_WARNING_COLOR)
            return None
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        base = super().flags(index)
        if index.isValid():
            return base | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        return base | Qt.ItemFlag.ItemIsDropEnabled

    def supportedDropActions(self) -> Qt.DropAction:
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:
        return [_MIME_TYPE]

    def mimeData(self, indexes: list) -> QMimeData | None:
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
        _parent: QModelIndex,
    ) -> bool:
        """Reorder on drop; returns False so the view skips its own removal.

        The move is already applied here; returning False prevents Qt's
        InternalMove from additionally deleting the source rows.
        """
        if action != Qt.DropAction.MoveAction:
            return False
        raw = bytes(data.data(_MIME_TYPE)).decode("ascii", errors="ignore")
        if not raw:
            return False
        try:
            source_rows = [int(part) for part in raw.split(",")]
        except ValueError:
            return False
        if len(source_rows) != 1:
            return False
        source = source_rows[0]
        target = self.rowCount() if row < 0 else row
        if target > source:
            target -= 1
        if target == source or not 0 <= source < self.rowCount():
            return False
        self.move_step(source, target)
        return False

    # --- raw mutators (commands call these) --------------------------------------

    def add_step(self, block_name: str, index: int | None = None) -> str:
        """Append (or insert) a step for ``block_name``; returns the step id."""
        step = PipelineStep(id=uuid.uuid4().hex, block=block_name)
        position = len(self._steps) if index is None else max(0, min(index, len(self._steps)))
        self._insert(position, step)
        return step.id

    def insert_step(self, step: PipelineStep, index: int) -> None:
        position = max(0, min(index, len(self._steps)))
        self._insert(position, step)

    def remove_step(self, step_id: str) -> int:
        index = self.step_index(step_id)
        if index < 0:
            return index
        self.beginRemoveRows(QModelIndex(), index, index)
        steps = list(self._steps)
        del steps[index]
        self._steps = tuple(steps)
        self.endRemoveRows()
        self._validate()
        self.pipeline_changed.emit(self.pipeline)
        return index

    def move_step(self, from_index: int, to_index: int) -> None:
        """Move one step; ``to_index`` is the target position after removal."""
        count = len(self._steps)
        if not (0 <= from_index < count) or not (0 <= to_index < count) or from_index == to_index:
            return
        target = to_index if to_index < from_index else to_index + 1
        self.beginMoveRows(QModelIndex(), from_index, from_index, QModelIndex(), target)
        steps = list(self._steps)
        step = steps.pop(from_index)
        steps.insert(to_index if to_index < from_index else to_index, step)
        self._steps = tuple(steps)
        self.endMoveRows()
        self._validate()
        self.pipeline_changed.emit(self.pipeline)

    def set_bypass(self, step_id: str, bypass: bool) -> None:
        index = self.step_index(step_id)
        if index < 0 or self._steps[index].bypass == bypass:
            return
        replacement = PipelineStep(
            id=step_id,
            block=self._steps[index].block,
            params=dict(self._steps[index].params),
            bypass=bypass,
            mask_bindings=dict(self._steps[index].mask_bindings),
        )
        self._replace(index, replacement)

    def set_param(self, step_id: str, param: str, value: Any) -> None:
        index = self.step_index(step_id)
        if index < 0:
            return
        step = self._steps[index]
        params = dict(step.params)
        if params.get(param) == value and param in params:
            return
        params[param] = value
        self._replace(
            index,
            PipelineStep(
                id=step_id, block=step.block, params=params, bypass=step.bypass,
                mask_bindings=dict(step.mask_bindings),
            ),
        )

    def delete_param(self, step_id: str, param: str) -> None:
        """Remove an explicitly-set parameter (restores the block default)."""
        index = self.step_index(step_id)
        if index < 0:
            return
        step = self._steps[index]
        if param not in step.params:
            return
        params = dict(step.params)
        del params[param]
        self._replace(
            index,
            PipelineStep(
                id=step_id, block=step.block, params=params, bypass=step.bypass,
                mask_bindings=dict(step.mask_bindings),
            ),
        )

    def set_mask_binding(self, step_id: str, param: str, upstream_step_id: str | None) -> None:
        index = self.step_index(step_id)
        if index < 0:
            return
        step = self._steps[index]
        bindings = dict(step.mask_bindings)
        if upstream_step_id is None:
            bindings.pop(param, None)
        else:
            bindings[param] = upstream_step_id
        if bindings == step.mask_bindings:
            return
        self._replace(
            index,
            PipelineStep(
                id=step_id, block=step.block, params=dict(step.params), bypass=step.bypass,
                mask_bindings=bindings,
            ),
        )

    def set_pipeline(self, pipeline: Pipeline) -> None:
        """Replace the whole pipeline (project load; the undo stack clears)."""
        self.beginResetModel()
        self._steps = tuple(pipeline.steps)
        self.endResetModel()
        self._validate()
        self.pipeline_changed.emit(self.pipeline)

    # --- helpers -----------------------------------------------------------------

    def _insert(self, position: int, step: PipelineStep) -> None:
        self.beginInsertRows(QModelIndex(), position, position)
        steps = list(self._steps)
        steps.insert(position, step)
        self._steps = tuple(steps)
        self.endInsertRows()
        self._validate()
        self.pipeline_changed.emit(self.pipeline)

    def _replace(self, index: int, step: PipelineStep) -> None:
        steps = list(self._steps)
        steps[index] = step
        self._steps = tuple(steps)
        model_index = self.index(index, 0)
        self.dataChanged.emit(model_index, model_index)
        self._validate()
        self.pipeline_changed.emit(self.pipeline)

    def step_index(self, step_id: str) -> int:
        for index, step in enumerate(self._steps):
            if step.id == step_id:
                return index
        return -1

    def step(self, step_id: str) -> PipelineStep | None:
        index = self.step_index(step_id)
        return self._steps[index] if 0 <= index < len(self._steps) else None

    def step_id_at(self, row: int) -> str | None:
        """The step id at ``row``, or ``None`` when out of range."""
        if 0 <= row < len(self._steps):
            return self._steps[row].id
        return None

    def _worst_severity(self, step_id: str) -> str | None:
        severities = [issue.severity for issue in self._issues if issue.step_id == step_id]
        if "error" in severities:
            return "error"
        if "warning" in severities:
            return "warning"
        return None

    def _validate(self) -> None:
        self._issues = validate_pipeline(
            self.pipeline,
            self._snapshot_provider(),
            second_image_available=self.second_image_available,
            sources_available=self.sources_available,
        )
        if self._issues and self._steps:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._steps) - 1, 0))
        self.blocking_changed.emit(self.has_blocking_issues())


class PipelineEditorPanel(QWidget):
    """Step list with add/remove/reorder/bypass and inline validation."""

    #: The selected step changed (drives the parameter panel).
    step_selected = Signal(str)  # step id, empty string when none

    def __init__(
        self,
        model: PipelineModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._stack = QUndoStack(self)
        self._stack.setUndoLimit(100)

        self._view = QListView(self)
        self._view.setModel(model)
        self._view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self._view.setDragDropMode(QListView.DragDropMode.InternalMove)
        self._view.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._view.selectionModel().currentChanged.connect(self._on_current_changed)

        self._remove_button = QToolButton(self)
        self._remove_button.clicked.connect(self._remove_selected)
        self._bypass_button = QToolButton(self)
        self._bypass_button.setCheckable(True)
        self._bypass_button.clicked.connect(self._toggle_bypass)
        self._move_up = QToolButton(self)
        self._move_up.clicked.connect(lambda: self._move(-1))
        self._move_down = QToolButton(self)
        self._move_down.clicked.connect(lambda: self._move(1))

        self._issues_label = QLabel(self)
        self._issues_label.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self._move_up)
        buttons.addWidget(self._move_down)
        buttons.addWidget(self._bypass_button)
        buttons.addWidget(self._remove_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        layout.addLayout(buttons)
        layout.addWidget(self._issues_label)

        model.blocking_changed.connect(self._on_blocking_changed)
        model.pipeline_changed.connect(lambda _p: self._refresh_issues())
        self.retranslate()

    # --- public API ------------------------------------------------------------

    def undo_stack(self) -> QUndoStack:
        return self._stack

    def model(self) -> PipelineModel:
        return self._model

    def add_block(self, block_name: str) -> None:
        """Add a block (library activation); undoable."""
        self._stack.push(AddStepCommand(self._model, block_name))

    def load_pipeline(self, pipeline: Pipeline) -> None:
        """Replace the pipeline; clears the undo stack (FR-UNDO-04)."""
        self._stack.push(LoadPipelineCommand(self._model, pipeline))
        self._stack.clear()

    def current_step_id(self) -> str | None:
        index = self._view.currentIndex()
        if not index.isValid():
            return None
        return self._model.data(index, ROLE_STEP_ID)  # type: ignore[return-value]

    # --- internals ---------------------------------------------------------------

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        step_id = self._model.data(current, ROLE_STEP_ID) if current.isValid() else None
        self.step_selected.emit(str(step_id) if step_id else "")

    def _remove_selected(self) -> None:
        step_id = self.current_step_id()
        if step_id:
            self._stack.push(RemoveStepCommand(self._model, step_id))

    def _toggle_bypass(self) -> None:
        step_id = self.current_step_id()
        if step_id:
            self._stack.push(BypassStepCommand(self._model, step_id, self._bypass_button.isChecked()))

    def _move(self, delta: int) -> None:
        index = self._view.currentIndex()
        if not index.isValid():
            return
        source = index.row()
        target = source + delta
        if not 0 <= target < self._model.rowCount():
            return
        self._stack.push(MoveStepCommand(self._model, source, target))
        self._view.setCurrentIndex(self._model.index(target, 0))

    def _refresh_issues(self) -> None:
        issues = self._model.issues
        if not issues:
            self._issues_label.setText("")
            return
        lines = [f"{'✗' if issue.severity == 'error' else '⚠'} {issue.message}" for issue in issues]
        self._issues_label.setText("\n".join(lines))

    def _on_blocking_changed(self, blocking: bool) -> None:
        self._issues_label.setStyleSheet(
            f"color: {_ERROR_COLOR.name()}" if blocking else f"color: {_WARNING_COLOR.name()}"
        )

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._remove_button.setText(self.tr("Remove"))
        self._bypass_button.setText(self.tr("Bypass"))
        self._move_up.setText(self.tr("↑"))
        self._move_down.setText(self.tr("↓"))

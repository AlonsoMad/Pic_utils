"""Lab undo commands (architecture §12, AD-14).

Commands mutate the :class:`~pic_utils.ui.lab.pipeline_editor.PipelineModel`
through its raw mutators; the panel pushes them onto a ``QUndoStack`` with
``setUndoLimit`` from settings. ``SetParamCommand`` implements ``id()`` and
``mergeWith()`` so slider/spinbox drags merge into one undo step. Loading a
pipeline clears the stack (FR-UNDO-04: project load is not undoable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtGui import QUndoCommand

from pic_utils.pipeline.model import Pipeline, PipelineStep

if TYPE_CHECKING:
    from pic_utils.ui.lab.pipeline_editor import PipelineModel

__all__ = [
    "AddStepCommand",
    "BypassStepCommand",
    "LoadPipelineCommand",
    "MoveStepCommand",
    "RemoveStepCommand",
    "SetMaskBindingCommand",
    "SetParamCommand",
]


class AddStepCommand(QUndoCommand):
    """Add one step for ``block_name`` at ``index`` (default: append)."""

    def __init__(self, model: PipelineModel, block_name: str, index: int | None = None) -> None:
        super().__init__()
        self._model = model
        self._block_name = block_name
        self._index = index
        self._step_id: str | None = None

    def redo(self) -> None:
        self._step_id = self._model.add_step(self._block_name, self._index)

    def undo(self) -> None:
        if self._step_id is not None:
            self._model.remove_step(self._step_id)

    def text(self) -> str:
        return f"add {self._block_name}"


class RemoveStepCommand(QUndoCommand):
    """Remove one step by id; undo reinserts it at its old index."""

    def __init__(self, model: PipelineModel, step_id: str) -> None:
        super().__init__()
        self._model = model
        self._step_id = step_id
        self._index: int | None = None
        self._step: PipelineStep | None = None

    def redo(self) -> None:
        self._index = self._model.step_index(self._step_id)
        self._step = self._model.step(self._step_id)
        self._model.remove_step(self._step_id)

    def undo(self) -> None:
        if self._step is not None and self._index is not None:
            self._model.insert_step(self._step, self._index)

    def text(self) -> str:
        return f"remove step {self._step_id}"


class MoveStepCommand(QUndoCommand):
    """Move a step from one row to another (drag-release pushes once, §12)."""

    def __init__(self, model: PipelineModel, from_index: int, to_index: int) -> None:
        super().__init__()
        self._model = model
        self._from = from_index
        self._to = to_index

    def redo(self) -> None:
        self._model.move_step(self._from, self._to)

    def undo(self) -> None:
        self._model.move_step(self._to, self._from)

    def text(self) -> str:
        return f"move step {self._from} → {self._to}"


class BypassStepCommand(QUndoCommand):
    """Toggle a step's bypass flag."""

    def __init__(self, model: PipelineModel, step_id: str, bypass: bool) -> None:
        super().__init__()
        self._model = model
        self._step_id = step_id
        self._bypass = bypass

    def redo(self) -> None:
        self._model.set_bypass(self._step_id, self._bypass)

    def undo(self) -> None:
        self._model.set_bypass(self._step_id, not self._bypass)

    def text(self) -> str:
        return f"{'bypass' if self._bypass else 'enable'} step {self._step_id}"


class SetParamCommand(QUndoCommand):
    """Set one parameter; consecutive edits of the same cell merge (§12)."""

    def __init__(self, model: PipelineModel, step_id: str, param: str, value: Any) -> None:
        super().__init__()
        self._model = model
        self._step_id = step_id
        self._param = param
        self._new = value
        self._old: Any = None
        self._had_key = False
        self._captured = False

    def id(self) -> int:
        return hash((SetParamCommand, self._step_id, self._param)) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, SetParamCommand):
            return False
        if (other._step_id, other._param) != (self._step_id, self._param):
            return False
        self._new = other._new
        return True

    def redo(self) -> None:
        if not self._captured:
            step = self._model.step(self._step_id)
            self._had_key = step is not None and self._param in step.params
            self._old = step.params.get(self._param) if step is not None else None
            self._captured = True
        self._model.set_param(self._step_id, self._param, self._new)

    def undo(self) -> None:
        if self._had_key:
            self._model.set_param(self._step_id, self._param, self._old)
        else:
            self._model.delete_param(self._step_id, self._param)

    def text(self) -> str:
        return f"set {self._param} in {self._step_id}"


class SetMaskBindingCommand(QUndoCommand):
    """Bind (or unbind with ``None``) a mask parameter to an upstream step."""

    def __init__(
        self, model: PipelineModel, step_id: str, param: str, upstream_step_id: str | None
    ) -> None:
        super().__init__()
        self._model = model
        self._step_id = step_id
        self._param = param
        self._new = upstream_step_id
        self._old: str | None = None
        self._captured = False

    def redo(self) -> None:
        if not self._captured:
            step = self._model.step(self._step_id)
            self._old = step.mask_bindings.get(self._param) if step is not None else None
            self._captured = True
        self._model.set_mask_binding(self._step_id, self._param, self._new)

    def undo(self) -> None:
        self._model.set_mask_binding(self._step_id, self._param, self._old)

    def text(self) -> str:
        return f"bind {self._param} in {self._step_id}"


class LoadPipelineCommand(QUndoCommand):
    """Replace the whole pipeline. The caller clears the stack after push."""

    def __init__(self, model: PipelineModel, pipeline: Pipeline) -> None:
        super().__init__()
        self._model = model
        self._pipeline = pipeline
        self._previous: Pipeline | None = None

    def redo(self) -> None:
        self._previous = self._model.pipeline
        self._model.set_pipeline(self._pipeline)

    def undo(self) -> None:
        if self._previous is not None:
            self._model.set_pipeline(self._previous)

    def text(self) -> str:
        return "load pipeline"

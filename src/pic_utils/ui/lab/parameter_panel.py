"""Parameter panel: dynamic form for the selected step (§13.2, §13.4).

The panel rebuilds from the selected step's ``BlockDef.params``: one row per
parameter with its widget (§13.4 table), a mask-binding combo for ``Mask``
parameters, and a per-parameter vary control (fixed / sweep / sample) that
edits the ``RunConfig``, not the pipeline. Every edit pushes a command onto
the mode's shared ``QUndoStack``; slider/spinbox drags merge (§12).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pic_utils.blocks.api import (
    Choice,
    FloatRange,
    IntRange,
    OutputKind,
    ParamDef,
    ParamKind,
)
from pic_utils.pipeline.model import PipelineStep
from pic_utils.ui.lab.commands import SetMaskBindingCommand, SetParamCommand
from pic_utils.ui.widgets.param_widgets import make_param_widget

if TYPE_CHECKING:
    from PySide6.QtGui import QUndoStack

    from pic_utils.ui.lab.pipeline_editor import PipelineModel

__all__ = ["ParameterPanel", "VaryMode"]

VaryMode = str  # "fixed" | "sweep" | "sample"

_VARY_MODES = ("fixed", "sweep", "sample")
_SAMPLEABLE = frozenset(
    {ParamKind.INT, ParamKind.FLOAT, ParamKind.CHOICE, ParamKind.BOOL, ParamKind.COLOR}
)


class ParameterPanel(QWidget):
    """Edits one step's parameters; drives the shared undo stack."""

    #: (step_id, param, value) — a parameter edit (undoable, merged per drag).
    param_edited = Signal(str, str, object)
    #: (step_id, param, upstream_step_id | None) — a mask binding change.
    mask_binding_changed = Signal(str, str, object)
    #: (step_id, param, mode, payload) — a vary control change (RunConfig).
    vary_changed = Signal(str, str, str, object)

    def __init__(
        self,
        model: PipelineModel,
        undo_stack: QUndoStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._undo_stack = undo_stack
        self._step_id: str | None = None
        self._vary_state: dict[tuple[str, str], tuple[VaryMode, Any]] = {}
        self._widgets: dict[str, QWidget] = {}

        self._form = QFormLayout()
        self._form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._empty = QLabel(self)
        layout = QVBoxLayout(self)
        layout.addLayout(self._form)
        layout.addWidget(self._empty)
        self.retranslate()

    # --- public API ---------------------------------------------------------

    def show_step(self, step_id: str | None) -> None:
        """Rebuild the form for ``step_id`` (empty form when ``None``)."""
        self._step_id = step_id
        self._rebuild()

    def selected_step_id(self) -> str | None:
        return self._step_id

    def widget(self, param_name: str) -> QWidget | None:
        """The live widget for one parameter of the shown step."""
        return self._widgets.get(param_name)

    def vary_state(self) -> dict[tuple[str, str], tuple[VaryMode, Any]]:
        """Per-parameter vary selections: ``(step_id, param) -> (mode, payload)``.

        ``payload`` is the sweep values tuple for ``"sweep"``, ``None``
        otherwise. Absent entries are fixed.
        """
        return dict(self._vary_state)

    def apply_vary_to(self, axes: list, sampled: list) -> None:
        """Append the vary selections to planner inputs (§7.1).

        ``axes`` collects ``(step_id, param, values)`` sweep triples;
        ``sampled`` collects ``(step_id, param)`` pairs.
        """
        for (step_id, param), (mode, payload) in sorted(self._vary_state.items()):
            if mode == "sweep" and payload:
                axes.append((step_id, param, tuple(payload)))
            elif mode == "sample":
                sampled.append((step_id, param))

    def set_vary_state(self, state: dict[tuple[str, str], tuple[VaryMode, Any]]) -> None:
        """Replace the vary selections (project load, §17.1)."""
        self._vary_state = dict(state)

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._empty.setText(self.tr("Select a step to edit its parameters"))

    # --- internals ------------------------------------------------------------

    def _rebuild(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()
        self._empty.setVisible(self._step_id is None)
        if self._step_id is None:
            return
        step = self._model.step(self._step_id)
        if step is None:
            self._empty.setVisible(True)
            return
        definition = self._model.snapshot().get(step.block)
        for param in definition.params:
            if param.kind is ParamKind.IMAGE:
                continue  # chain input / second image: injected by the executor
            self._form.addRow(self._label(param), self._field(param, step))

    def _label(self, param: ParamDef) -> QLabel:
        label = QLabel(str(param.name))
        label.setToolTip(str(param.name))
        return label

    def _field(self, param: ParamDef, step: PipelineStep) -> QWidget:
        container = QWidget(self)
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)

        if param.kind is ParamKind.MASK:
            row.addWidget(QLabel(self.tr("mask:"), self))
            binding = QComboBox(self)
            binding.addItem(self.tr("unbound"))
            binding.addItems(self._upstream_mask_steps())
            current = step.mask_bindings.get(param.name)
            if current is not None:
                binding.setCurrentText(current)
            binding.currentTextChanged.connect(
                lambda text, param_name=param.name: self._on_binding_changed(param_name, text)
            )
            row.addWidget(binding)
            return container

        widget = make_param_widget(param)
        widget.set_value(step.params.get(param.name, param.default))
        widget.edited.connect(
            lambda value, param_name=param.name: self._on_param_edited(param_name, value)
        )
        row.addWidget(widget, 1)
        self._widgets[param.name] = widget

        if param.kind in _SAMPLEABLE:
            vary = QComboBox(self)
            vary.addItems(list(_VARY_MODES))
            mode = self._vary_state.get((self._step_id or "", param.name), ("fixed", None))[0]
            vary.setCurrentText(mode)
            vary.currentTextChanged.connect(
                lambda mode_, param_name=param.name: self._on_vary_changed(param_name, mode_)
            )
            row.addWidget(vary)

        return container

    def _on_param_edited(self, param: str, value: object) -> None:
        if self._step_id is None:
            return
        self._undo_stack.push(SetParamCommand(self._model, self._step_id, param, value))
        self.param_edited.emit(self._step_id, param, value)

    def _on_binding_changed(self, param: str, text: str) -> None:
        if self._step_id is None:
            return
        upstream = None if text == self.tr("unbound") else text
        self._undo_stack.push(SetMaskBindingCommand(self._model, self._step_id, param, upstream))
        self.mask_binding_changed.emit(self._step_id, param, upstream)

    def _on_vary_changed(self, param: str, mode: VaryMode) -> None:
        if self._step_id is None:
            return
        payload = self._sweep_payload(param) if mode == "sweep" else None
        if mode == "fixed":
            self._vary_state.pop((self._step_id, param), None)
        else:
            self._vary_state[(self._step_id, param)] = (mode, payload)
        self.vary_changed.emit(self._step_id, param, mode, payload)

    def _sweep_payload(self, param: str) -> tuple:
        """Sweep values from the parameter's spec: endpoints and options."""
        step = self._model.step(self._step_id) if self._step_id else None
        if step is None:
            return ()
        definition = self._model.snapshot().get(step.block)
        param_def = next((p for p in definition.params if p.name == param), None)
        if param_def is None:
            return ()
        spec = param_def.spec
        if isinstance(spec, IntRange | FloatRange):
            return (spec.min, spec.max)
        if isinstance(spec, Choice):
            return tuple(spec.options)
        if param_def.kind is ParamKind.BOOL:
            return (True, False)
        return ()

    def _upstream_mask_steps(self) -> list[str]:
        """Step ids before the selection whose block outputs a mask (§6.3)."""
        upstream: list[str] = []
        for step in self._model.pipeline.steps:
            if step.id == self._step_id:
                break
            if self._model.snapshot().get(step.block).output is OutputKind.MASK:
                upstream.append(step.id)
        return upstream

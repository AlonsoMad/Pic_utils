"""Pipeline editor and undo command tests (P5-03)."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QMimeData, QModelIndex, Qt
from PySide6.QtGui import QUndoStack

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.pipeline.model import Pipeline
from pic_utils.ui.lab.commands import (
    AddStepCommand,
    BypassStepCommand,
    MoveStepCommand,
    RemoveStepCommand,
    SetMaskBindingCommand,
    SetParamCommand,
)
from pic_utils.ui.lab.pipeline_editor import PipelineEditorPanel, PipelineModel


@pytest.fixture
def registry() -> BlockRegistry:
    return BlockRegistry(None)


@pytest.fixture
def model(registry: BlockRegistry) -> PipelineModel:
    pipeline_model = PipelineModel(registry.snapshot)
    pipeline_model.sources_available = True
    return pipeline_model


def _names(model: PipelineModel) -> list[str]:
    return [
        model.data(model.index(row, 0), Qt.ItemDataRole.DisplayRole)
        for row in range(model.rowCount())
    ]


class TestModelMutations:
    def test_add_step_appends(self, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        assert _names(model) == ["channel_mask"]
        assert model.step(step_id) is not None

    def test_add_step_at_index(self, model: PipelineModel) -> None:
        model.add_step("channel_mask")
        model.add_step("overlay")
        model.add_step("segment", 1)
        assert _names(model) == ["channel_mask", "segment", "overlay"]

    def test_remove_step(self, model: PipelineModel) -> None:
        first = model.add_step("channel_mask")
        model.add_step("overlay")
        model.remove_step(first)
        assert _names(model) == ["overlay"]

    def test_move_step_down(self, model: PipelineModel) -> None:
        model.add_step("channel_mask")
        model.add_step("overlay")
        model.move_step(0, 1)
        assert _names(model) == ["overlay", "channel_mask"]

    def test_move_step_up(self, model: PipelineModel) -> None:
        model.add_step("channel_mask")
        model.add_step("overlay")
        model.move_step(1, 0)
        assert _names(model) == ["overlay", "channel_mask"]

    def test_set_param(self, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 42)
        assert model.step(step_id).params["low"] == 42

    def test_set_bypass(self, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        model.set_bypass(step_id, True)
        assert model.step(step_id).bypass is True

    def test_set_mask_binding(self, model: PipelineModel) -> None:
        mask_id = model.add_step("channel_mask")
        overlay_id = model.add_step("conditional_overlay")
        model.set_mask_binding(overlay_id, "mask", mask_id)
        assert model.step(overlay_id).mask_bindings == {"mask": mask_id}

    def test_set_mask_binding_unbinds(self, model: PipelineModel) -> None:
        mask_id = model.add_step("channel_mask")
        overlay_id = model.add_step("conditional_overlay")
        model.set_mask_binding(overlay_id, "mask", mask_id)
        model.set_mask_binding(overlay_id, "mask", None)
        assert model.step(overlay_id).mask_bindings == {}

    def test_pipeline_changed_emitted(self, model: PipelineModel) -> None:
        seen: list[Pipeline] = []
        model.pipeline_changed.connect(seen.append)
        model.add_step("channel_mask")
        assert len(seen) == 1
        assert isinstance(seen[0], Pipeline)

    def test_set_pipeline_round_trip(self, model: PipelineModel) -> None:
        pipeline = Pipeline.from_dict(
            {"steps": [{"id": "a", "block": "channel_mask", "params": {"low": 10}}]}
        )
        model.set_pipeline(pipeline)
        assert _names(model) == ["channel_mask"]
        assert model.step("a").params["low"] == 10


class TestValidation:
    def test_empty_pipeline_blocks(self, model: PipelineModel) -> None:
        assert model.has_blocking_issues()
        codes = {issue.code for issue in model.issues}
        assert "pipeline.empty" in codes

    def test_unknown_block_flagged(self, model: PipelineModel) -> None:
        model.set_pipeline(Pipeline.from_dict({"steps": [{"id": "a", "block": "nope"}]}))
        codes = {issue.code for issue in model.issues}
        assert "step.unknown_block" in codes

    def test_valid_pipeline_not_blocking(self, model: PipelineModel) -> None:
        model.add_step("channel_permute")
        assert not model.has_blocking_issues()

    def test_issues_attach_to_step(self, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 10_000)
        issues = [issue for issue in model.issues if issue.param == "low"]
        assert issues

    def test_revalidate_after_snapshot_change(self, registry: BlockRegistry) -> None:
        model = PipelineModel(registry.snapshot)
        model.sources_available = True
        model.add_step("channel_permute")
        assert not model.has_blocking_issues()
        model.revalidate()
        assert not model.has_blocking_issues()


class TestUndo:
    def test_add_undo_removes(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        stack.push(AddStepCommand(model, "channel_mask"))
        assert model.rowCount() == 1
        stack.undo()
        assert model.rowCount() == 0
        stack.redo()
        assert model.rowCount() == 1

    def test_remove_undo_reinserts_at_index(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        first = model.add_step("channel_mask")
        model.add_step("overlay")
        stack.push(RemoveStepCommand(model, first))
        assert _names(model) == ["overlay"]
        stack.undo()
        assert _names(model) == ["channel_mask", "overlay"]

    def test_move_undo_restores_order(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        model.add_step("channel_mask")
        model.add_step("overlay")
        stack.push(MoveStepCommand(model, 0, 1))
        assert _names(model) == ["overlay", "channel_mask"]
        stack.undo()
        assert _names(model) == ["channel_mask", "overlay"]

    def test_bypass_undo(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        step_id = model.add_step("channel_mask")
        stack.push(BypassStepCommand(model, step_id, True))
        assert model.step(step_id).bypass is True
        stack.undo()
        assert model.step(step_id).bypass is False

    def test_set_param_undo_restores_value(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 5)
        stack.push(SetParamCommand(model, step_id, "low", 50))
        assert model.step(step_id).params["low"] == 50
        stack.undo()
        assert model.step(step_id).params["low"] == 5

    def test_set_param_merges_same_cell(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 5)
        stack.push(SetParamCommand(model, step_id, "low", 10))
        stack.push(SetParamCommand(model, step_id, "low", 20))
        stack.push(SetParamCommand(model, step_id, "low", 30))
        assert stack.count() == 1  # drag merged into one step
        stack.undo()
        assert model.step(step_id).params["low"] == 5
        stack.redo()
        assert model.step(step_id).params["low"] == 30

    def test_set_param_does_not_merge_different_cells(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 5)
        stack.push(SetParamCommand(model, step_id, "low", 10))
        stack.push(SetParamCommand(model, step_id, "high", 200))
        assert stack.count() == 2

    def test_mask_binding_undo(self, model: PipelineModel) -> None:
        stack = QUndoStack()
        mask_id = model.add_step("channel_mask")
        overlay_id = model.add_step("conditional_overlay")
        stack.push(SetMaskBindingCommand(model, overlay_id, "mask", mask_id))
        assert model.step(overlay_id).mask_bindings == {"mask": mask_id}
        stack.undo()
        assert model.step(overlay_id).mask_bindings == {}

    def test_load_pipeline_clears_stack(self, model: PipelineModel, qtbot: object) -> None:
        panel = PipelineEditorPanel(model)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(panel)
        model.add_step("channel_mask")
        pipeline = Pipeline.from_dict({"steps": [{"id": "b", "block": "overlay"}]})
        panel.load_pipeline(pipeline)
        assert panel.undo_stack().count() == 0
        assert _names(model) == ["overlay"]
        assert not panel.undo_stack().canUndo()


class TestPanel:
    def test_add_block_pushes_command(self, model: PipelineModel, qtbot: object) -> None:
        panel = PipelineEditorPanel(model)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(panel)
        panel.add_block("channel_mask")
        assert model.rowCount() == 1
        assert panel.undo_stack().count() == 1

    def test_step_selected_signal(self, model: PipelineModel, qtbot: object) -> None:
        panel = PipelineEditorPanel(model)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(panel)
        seen: list[str] = []
        panel.step_selected.connect(seen.append)
        panel.add_block("channel_mask")
        panel._view.setCurrentIndex(model.index(0, 0))
        assert seen and seen[-1] == model.step_id_at(0)

    def test_remove_selected(self, model: PipelineModel, qtbot: object) -> None:
        panel = PipelineEditorPanel(model)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(panel)
        panel.add_block("channel_mask")
        panel._view.setCurrentIndex(model.index(0, 0))
        panel._remove_selected()
        assert model.rowCount() == 0

    def test_drop_mime_reorders(self, model: PipelineModel) -> None:
        model.add_step("channel_mask")
        model.add_step("overlay")
        mime = QMimeData()
        mime.setData("application/x-pic-utils-steps", b"0")
        model.dropMimeData(mime, Qt.DropAction.MoveAction, 2, 0, QModelIndex())
        assert _names(model) == ["overlay", "channel_mask"]

    def test_undo_limit_is_bounded(self, model: PipelineModel, qtbot: object) -> None:
        panel = PipelineEditorPanel(model)
        add_widget = getattr(qtbot, "addWidget", None)
        if callable(add_widget):
            add_widget(panel)
        assert panel.undo_stack().undoLimit() == 100

"""Parameter panel and widget tests (P5-04)."""

from __future__ import annotations

import pytest
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QComboBox, QFormLayout

from pic_utils.blocks.api import (
    Choice,
    FloatRange,
    IntRange,
    ParamDef,
    ParamKind,
    Seed,
)
from pic_utils.blocks.registry import BlockRegistry
from pic_utils.ui.lab.parameter_panel import ParameterPanel
from pic_utils.ui.lab.pipeline_editor import PipelineModel
from pic_utils.ui.widgets.param_widgets import make_param_widget


@pytest.fixture
def registry() -> BlockRegistry:
    return BlockRegistry(None)


@pytest.fixture
def model(registry: BlockRegistry) -> PipelineModel:
    pipeline_model = PipelineModel(registry.snapshot)
    pipeline_model.sources_available = True
    return pipeline_model


@pytest.fixture
def panel(qtbot: object, model: PipelineModel) -> ParameterPanel:
    panel = ParameterPanel(model, QUndoStack())
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    return panel


def _param(name: str, kind: ParamKind, default: object, spec: object) -> ParamDef:
    return ParamDef(
        name=name,
        kind=kind,
        annotation=type(default),
        default=default,
        required=False,
        spec=spec,
    )


class TestParamWidgets:
    def test_int_widget_edits_and_respects_range(self, qtbot: object) -> None:
        definition = _param("amount", ParamKind.INT, 5, IntRange(0, 100, step=1))
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        seen: list[object] = []
        widget.edited.connect(seen.append)
        widget._spin.setValue(42)
        assert seen == [42]
        widget.set_value(7)
        assert widget._spin.value() == 7

    def test_float_widget_reports_edits(self, qtbot: object) -> None:
        definition = _param("factor", ParamKind.FLOAT, 0.5, FloatRange(0.0, 2.0, step=0.01))
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        seen: list[object] = []
        widget.edited.connect(seen.append)
        widget._spin.setValue(1.5)
        assert seen and abs(seen[-1] - 1.5) < 1e-9

    def test_choice_widget_uses_labels(self, qtbot: object) -> None:
        definition = _param(
            "channel",
            ParamKind.CHOICE,
            "R",
            Choice(("R", "G", "B"), labels=("Red", "Green", "Blue")),
        )
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        assert widget._combo.itemText(0) == "Red"
        seen: list[object] = []
        widget.edited.connect(seen.append)
        widget._combo.setCurrentIndex(2)
        assert seen == ["B"]

    def test_bool_widget(self, qtbot: object) -> None:
        definition = _param("soft", ParamKind.BOOL, False, None)
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        seen: list[object] = []
        widget.edited.connect(seen.append)
        widget._check.setChecked(True)
        assert seen == [True]

    def test_color_widget_edits(self, qtbot: object) -> None:
        definition = _param("tint", ParamKind.COLOR, "#ff0000", None)
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        widget.set_value("#00ff00")
        assert widget._button.text() == "#00ff00"

    def test_seed_widget_auto_and_randomize(self, qtbot: object) -> None:
        definition = _param("seed", ParamKind.SEED, None, Seed())
        widget = make_param_widget(definition)
        qtbot.addWidget(widget)  # type: ignore[attr-defined]
        seen: list[object] = []
        widget.edited.connect(seen.append)
        assert not widget._spin.isEnabled()  # default is auto
        widget._auto.setChecked(False)
        assert widget._spin.isEnabled()
        widget.set_value(None)
        assert widget._auto.isChecked()
        assert not widget._spin.isEnabled()


class TestParameterPanel:
    def test_empty_when_no_step(self, panel: ParameterPanel) -> None:
        panel.show_step(None)
        assert panel._form.rowCount() == 0
        assert not panel._empty.isHidden()

    def test_rows_match_block_params(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        panel.show_step(step_id)
        names = {
            panel._form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget().text()
            for row in range(panel._form.rowCount())
        }
        assert {"channel", "low", "high", "soft", "invert"} <= names

    def test_image_params_not_shown(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("double_exposure")
        panel.show_step(step_id)
        names = {
            panel._form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget().text()
            for row in range(panel._form.rowCount())
        }
        assert "image" not in names
        assert "second" not in names
        assert "exposure_factor" in names

    def test_edit_pushes_merged_command(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        model.set_param(step_id, "low", 5)
        panel.show_step(step_id)
        widget = panel.widget("low")
        assert widget is not None
        widget._spin.setValue(10)
        widget._spin.setValue(20)
        widget._spin.setValue(30)
        assert model.step(step_id).params["low"] == 30
        assert panel._undo_stack.count() == 1  # merged
        panel._undo_stack.undo()
        assert model.step(step_id).params["low"] == 5

    def test_mask_row_has_binding_combo(self, panel: ParameterPanel, model: PipelineModel) -> None:
        mask_id = model.add_step("channel_mask")
        overlay_id = model.add_step("conditional_overlay")
        panel.show_step(overlay_id)
        combos = [
            combo
            for row in range(panel._form.rowCount())
            for combo in panel._form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            .widget()
            .findChildren(QComboBox)
        ]
        assert combos, "the mask parameter row carries a binding combo"
        items = [combos[0].itemText(i) for i in range(combos[0].count())]
        assert mask_id in items
        labels = {
            panel._form.itemAt(row, QFormLayout.ItemRole.LabelRole).widget().text()
            for row in range(panel._form.rowCount())
        }
        assert "mask" in labels

    def test_mask_binding_pushes_command(self, panel: ParameterPanel, model: PipelineModel) -> None:
        mask_id = model.add_step("channel_mask")
        overlay_id = model.add_step("conditional_overlay")
        panel.show_step(overlay_id)
        panel._on_binding_changed("mask", mask_id)
        assert model.step(overlay_id).mask_bindings == {"mask": mask_id}
        panel._undo_stack.undo()
        assert model.step(overlay_id).mask_bindings == {}

    def test_vary_state_updates(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        panel.show_step(step_id)
        events: list[tuple] = []
        panel.vary_changed.connect(lambda *args: events.append(args))
        panel._on_vary_changed("low", "sweep")
        panel._on_vary_changed("high", "sample")
        panel._on_vary_changed("low", "fixed")
        axes: list = []
        sampled: list = []
        panel.apply_vary_to(axes, sampled)
        assert axes == []
        assert sampled == [(step_id, "high")]
        assert panel.vary_state() == {(step_id, "high"): ("sample", None)}
        assert len(events) == 3

    def test_vary_payload_for_range(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        panel.show_step(step_id)
        assert panel._sweep_payload("low") == (0, 255)

    def test_vary_payload_for_choice(self, panel: ParameterPanel, model: PipelineModel) -> None:
        step_id = model.add_step("channel_mask")
        panel.show_step(step_id)
        assert panel._sweep_payload("channel") == ("R", "G", "B", "L")

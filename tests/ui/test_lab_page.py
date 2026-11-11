"""Lab page composition tests (P5-10)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QLabel, QSplitter

from pic_utils.pipeline.model import Pipeline, RunConfig, SweepAxis
from pic_utils.ui.lab.page import LabPage
from pic_utils.ui.lab.project_actions import ProjectData


@pytest.fixture
def page(qtbot: object) -> LabPage:
    widget = LabPage()
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    return widget


@pytest.fixture
def image_png(tmp_path: Path) -> Path:
    path = tmp_path / "img.png"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    return path


class TestComposition:
    def test_all_panels_present(self, page: LabPage) -> None:
        assert page.library() is not None
        assert page.editor() is not None
        assert page.parameters() is not None
        assert page.results_grid() is not None
        assert page.preview_pane() is not None
        assert page.run_controls() is not None

    def test_splitters_layout(self, page: LabPage) -> None:
        splitters = page.findChildren(QSplitter)
        assert len(splitters) >= 2  # columns + rows

    def test_no_placeholder_labels_remain(self, page: LabPage) -> None:
        names = [child.objectName() for child in page.findChildren(QLabel)]
        assert "lab_block_library" not in names
        assert "lab_pipeline_editor" not in names
        assert "lab_preview_pane" not in names
        assert "lab_parameter_panel" not in names
        assert "lab_results_grid" not in names

    def test_run_controls_exposed_for_shell(self, page: LabPage) -> None:
        assert page.run_controls() is not None

    def test_all_panels_retranslate(self, page: LabPage) -> None:
        page.retranslate()
        assert page.library()._search.placeholderText()
        assert page.results_grid()._show_discarded.text()


class TestWiring:
    def test_library_activation_adds_step(self, page: LabPage) -> None:
        page.library().block_activated.emit("channel_mask")
        assert page.editor().model().rowCount() == 1

    def test_selection_shows_parameters(self, page: LabPage) -> None:
        step_id = page.editor().model().add_step("channel_mask")
        page.editor().step_selected.emit(step_id)
        assert page.parameters().selected_step_id() == step_id

    def test_plan_count_updates_on_pipeline_change(self, page: LabPage, image_png: Path) -> None:
        page.run_controls().add_sources([image_png])
        page.editor().add_block("channel_mask")
        assert "1" in page.run_controls()._plan_count.text()

    def test_launch_disabled_without_sources(self, page: LabPage) -> None:
        page._on_pipeline_changed()
        assert not page.run_controls()._launch.isEnabled()

    def test_run_started_binds_results(self, page: LabPage) -> None:
        page._on_run_started("20260101-000000-abc123", 5)
        assert page.results_grid().current_run_id() == "20260101-000000-abc123"
        assert page.run_controls().is_running()

    def test_run_finished_updates_summary(self, page: LabPage) -> None:
        from pic_utils.runner.service import RunSummary

        notifications: list[str] = []
        page.notify.connect(notifications.append)
        page._on_run_finished(
            RunSummary(
                run_id="r", status="completed", completed=3, failed=0, cancelled=0, elapsed_ms=10
            )
        )
        assert not page.run_controls().is_running()
        assert notifications and "3" in notifications[0]

    def test_result_ready_appends(self, page: LabPage) -> None:
        from pic_utils.runner.manifest import ResultRecord

        record = ResultRecord(
            id="0000:00",
            experiment=0,
            layer=0,
            source=0,
            kind="image",
            path="results/x.png",
            thumb="thumbs/x.png",
            width=8,
            height=8,
            seed=1,
            params={},
            elapsed_ms=1,
        )
        page._on_result_ready(record)
        assert page.results_grid().model().rowCount() == 1

    def test_memory_updates(self, page: LabPage) -> None:
        page._service.memory_updated.emit(2048)
        assert "2048" in page.run_controls()._memory.text()


class TestProjectGlue:
    def test_project_data_round_trip(self, page: LabPage, image_png: Path) -> None:
        page.run_controls().add_sources([image_png])
        page.editor().add_block("channel_mask")
        data = page.project_data()
        assert data.sources == (image_png,)
        assert data.pipeline.steps[0].block == "channel_mask"

    def test_load_project_data_applies(self, page: LabPage, image_png: Path) -> None:
        page.load_project_data(
            ProjectData(
                sources=(image_png,),
                second_image=None,
                pipeline=Pipeline.from_dict({"steps": [{"id": "s", "block": "channel_permute"}]}),
                run=RunConfig(mode="grid", batch_size=16, master_seed=3),
            )
        )
        assert page.editor().model().pipeline.steps[0].block == "channel_permute"
        assert page.run_controls().all_sources() == [image_png]
        assert page.run_controls().run_config().batch_size == 16

    def test_load_project_sets_vary_state(self, page: LabPage, image_png: Path) -> None:
        page.load_project_data(
            ProjectData(
                sources=(image_png,),
                second_image=None,
                pipeline=Pipeline.from_dict({"steps": [{"id": "s", "block": "channel_mask"}]}),
                run=RunConfig(
                    mode="grid",
                    axes=(SweepAxis(step_id="s", param="low", values=(10, 20)),),
                ),
            )
        )
        state = page.parameters().vary_state()
        assert ("s", "low") in state
        assert state[("s", "low")][0] == "sweep"


class TestPreviewWiring:
    def test_pipeline_change_requests_preview(self, page: LabPage, image_png: Path) -> None:
        page.run_controls().add_sources([image_png])
        page._on_pipeline_changed()
        assert page.preview_pane()._source == str(image_png)

    def test_registry_reload_restarts_preview(self, page: LabPage, qtbot: object) -> None:
        page._registry.reload()
        qtbot.wait(50)  # type: ignore[attr-defined]

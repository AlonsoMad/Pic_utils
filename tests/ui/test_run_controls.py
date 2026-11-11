"""Run controls tests (P5-07)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from pic_utils.pipeline.model import RunConfig
from pic_utils.runner.service import RunSummary
from pic_utils.ui.lab.run_controls import RunControls


@pytest.fixture
def panel(qtbot: object) -> RunControls:
    panel = RunControls(lambda config: None)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    return panel


@pytest.fixture
def image_png(tmp_path: Path) -> Path:
    import cv2
    import numpy as np

    path = tmp_path / "img.png"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    return path


class TestSources:
    def test_add_sources_deduplicates(self, panel: RunControls, image_png: Path) -> None:
        panel.add_sources([image_png, image_png])
        assert panel.all_sources() == [image_png]
        assert panel.sources() == [image_png]

    def test_unchecking_disables_source(self, panel: RunControls, image_png: Path) -> None:
        panel.add_sources([image_png])
        item = panel._source_list.item(0)
        item.setCheckState(Qt.CheckState.Unchecked)
        assert panel.all_sources() == [image_png]
        assert panel.sources() == []

    def test_remove_source(self, panel: RunControls, image_png: Path) -> None:
        panel.add_sources([image_png])
        panel._source_list.setCurrentRow(0)
        panel._remove_source.click()
        assert panel.all_sources() == []

    def test_sources_changed_signal(self, panel: RunControls, image_png: Path) -> None:
        events: list[list] = []
        panel.sources_changed.connect(lambda: events.append(panel.sources()))
        panel.add_sources([image_png])
        assert events == [[image_png]]


class TestSecondImage:
    def test_set_and_clear(self, panel: RunControls, image_png: Path) -> None:
        panel._second = image_png
        panel._update_second_label()
        assert panel.second_image() == image_png
        panel._clear_second()
        assert panel.second_image() is None


class TestRunConfig:
    def test_defaults(self, panel: RunControls) -> None:
        config = panel.run_config()
        assert isinstance(config, RunConfig)
        assert config.mode == "grid"
        assert config.batch_size == 32
        assert config.random_samples == 0
        assert config.master_seed == 0

    def test_mode_and_values_flow_through(self, panel: RunControls) -> None:
        panel._mode.setCurrentText("hybrid")
        panel._batch_size.setValue(8)
        panel._random_samples.setValue(3)
        panel._master_seed.setValue(42)
        config = panel.run_config()
        assert config.mode == "hybrid"
        assert config.batch_size == 8
        assert config.random_samples == 3
        assert config.master_seed == 42

    def test_plan_count_updates_on_change(self, image_png: Path) -> None:
        seen: list[RunConfig] = []

        def provider(config: RunConfig) -> int | None:
            seen.append(config)
            return 12

        panel = RunControls(provider)
        panel.add_sources([image_png])
        panel._batch_size.setValue(64)
        assert len(seen) >= 2
        assert "12" in panel._plan_count.text()


class TestLaunch:
    def test_launch_emits_config(self, qtbot: object, image_png: Path) -> None:
        panel = RunControls(lambda config: 5)
        qtbot.addWidget(panel)  # type: ignore[attr-defined]
        emitted: list[RunConfig] = []
        panel.launch_requested.connect(emitted.append)
        panel.add_sources([image_png])
        panel._launch.click()
        assert len(emitted) == 1
        assert emitted[0].batch_size == 32

    def test_launch_disabled_when_invalid(self, qtbot: object) -> None:
        panel = RunControls(lambda config: None)
        qtbot.addWidget(panel)  # type: ignore[attr-defined]
        emitted: list[RunConfig] = []
        panel.launch_requested.connect(emitted.append)
        panel._launch.click()
        assert emitted == []
        assert not panel._launch.isEnabled()

    def test_running_state_swaps_buttons(self, qtbot: object) -> None:
        panel = RunControls(lambda config: 5)
        qtbot.addWidget(panel)  # type: ignore[attr-defined]
        panel.set_running(True)
        assert not panel._launch.isEnabled()
        assert panel._cancel.isEnabled()
        panel.set_running(False)
        assert panel._launch.isEnabled()
        assert not panel._cancel.isEnabled()

    def test_cancel_signal(self, qtbot: object) -> None:
        panel = RunControls(lambda config: 5)
        qtbot.addWidget(panel)  # type: ignore[attr-defined]
        cancelled: list[bool] = []
        panel.cancel_requested.connect(lambda: cancelled.append(True))
        panel.set_running(True)
        panel._cancel.click()
        assert cancelled == [True]


class TestStatus:
    def test_progress_display(self, panel: RunControls) -> None:

        panel.show_progress(3, 10)
        assert "3 / 10" in panel._summary.text()

    def test_summary_display(self, panel: RunControls) -> None:

        summary = RunSummary(
            run_id="r", status="completed", completed=7, failed=1, cancelled=2, elapsed_ms=5400
        )
        panel.show_summary(summary)
        assert "7" in panel._summary.text()
        assert "1" in panel._summary.text()
        assert not panel.is_running()

    def test_memory_display(self, panel: RunControls) -> None:
        panel.set_memory_mb(1234)
        assert "1234" in panel._memory.text()

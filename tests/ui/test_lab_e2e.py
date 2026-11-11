"""End-to-end Lab journey on a synthetic image (P5-09).

load → pipeline → preview → run → curate, through the real panels: the
block registry, the undo-backed editor, the preview process, the spawned
worker pool, and the results grid with curation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pic_utils.settings import Settings
from pic_utils.ui.lab.page import LabPage


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = Settings(path=tmp_path / "settings.ini")
    settings.set_blocks_folder(tmp_path / "blocks")
    settings.set_results_root(tmp_path / "results")
    return settings


@pytest.fixture
def source_png(tmp_path: Path) -> Path:
    gradient = np.zeros((64, 96, 3), dtype=np.uint8)
    gradient[..., 0] = np.tile(np.linspace(0, 255, 96, dtype=np.uint8), (64, 1))
    gradient[..., 1] = np.linspace(0, 255, 64, dtype=np.uint8)[:, None]
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), gradient)
    return path


@pytest.fixture
def page(qtbot: object, settings: Settings, source_png: Path) -> LabPage:
    widget = LabPage(settings)
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    page = widget
    page.run_controls().add_sources([source_png])
    return page


def test_lab_journey_end_to_end(
    qtbot: object, page: LabPage, settings: Settings, source_png: Path
) -> None:
    # 1. load: the source is registered in the run controls.
    assert page.run_controls().sources() == [source_png]

    # 2. pipeline: a block is added through the library activation signal.
    page.library().block_activated.emit("channel_permute")
    pipeline_model = page.editor().model()
    assert pipeline_model.rowCount() == 1
    # the plan count updates to one experiment
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: "1" in page.run_controls()._plan_count.text(), timeout=10_000
    )

    # 3. preview: the debounced render returns a pixmap.
    page._on_pipeline_changed()
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: page.preview_pane()._image.pixmap() is not None
        and not page.preview_pane()._image.pixmap().isNull(),
        timeout=30_000,
    )

    # 4. run: launch the batch through the run service (real worker pool).
    notifications: list[str] = []
    page.notify.connect(notifications.append)
    with qtbot.waitSignal(page._service.run_finished, timeout=120_000):
        page._launch(page.run_controls().run_config())

    # the run produced results on disk and in the grid
    run_id = page.results_grid().current_run_id()
    assert run_id != "pending"
    run_dir = settings.results_root / run_id
    assert (run_dir / "manifest.json").is_file()
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: page.results_grid().model().rowCount() >= 1, timeout=10_000
    )

    # 5. curate: accept the first result through the shortcut.
    grid_model = page.results_grid().model()
    page.results_grid()._view.setCurrentIndex(grid_model.index(0, 0))
    page.results_grid().accept_current()
    curation = grid_model._store.read_curation()
    assert curation and next(iter(curation.values())) == "accepted"

    # the manifest reports a completed run and the trash is consistent
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    assert '"status": "completed"' in manifest


def test_lab_journey_with_discard_and_restore(
    qtbot: object, page: LabPage, settings: Settings, source_png: Path
) -> None:
    page.library().block_activated.emit("channel_permute")
    with qtbot.waitSignal(page._service.run_finished, timeout=120_000):
        page._launch(page.run_controls().run_config())

    grid_model = page.results_grid().model()
    page.results_grid()._view.setCurrentIndex(grid_model.index(0, 0))
    record = grid_model._record_at(0)
    assert record is not None
    result_file = page.results_grid()._store.run_dir / record.path

    # discard: the files move to the session trash
    page.results_grid().discard_current()
    assert not result_file.exists()
    assert grid_model.rowCount() == 0

    # restore: the files come back and the decision is cleared
    grid_model.set_show_discarded(True)
    page.results_grid()._view.setCurrentIndex(grid_model.index(0, 0))
    page.results_grid().restore_current()
    assert result_file.exists()
    assert record.id not in grid_model._store.read_curation()

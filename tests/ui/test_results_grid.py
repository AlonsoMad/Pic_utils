"""Results grid tests: model, curation, shortcuts (P5-06)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import Qt

from pic_utils.results.trash import SessionTrash
from pic_utils.runner.manifest import ResultRecord, RunStore
from pic_utils.ui.lab.results_grid import (
    ROLE_CURATION,
    ROLE_RECORD,
    ResultsGrid,
    ResultsGridModel,
)


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8, 3), value, dtype=np.uint8))


@pytest.fixture
def run(tmp_path: Path) -> RunStore:
    store = RunStore(tmp_path, "20260101-000000-abc123")
    header = {
        "created_at": "2026-01-01T00:00:00Z",
        "app_version": "0.1.0",
        "pipeline": {"steps": []},
    }
    store.create(header)
    for index in range(3):
        record = ResultRecord(
            id=f"000{index}:00",
            experiment=index,
            layer=0,
            source=0,
            kind="image",
            path=f"results/result_000{index}_00.png",
            thumb=f"thumbs/result_000{index}_00.png",
            width=8,
            height=8,
            seed=index,
            params={},
            elapsed_ms=1,
        )
        _write_png(store.run_dir / record.path, 10 + index)
        _write_png(store.run_dir / record.thumb, 10 + index)
        store.append_result(record)
    return store


@pytest.fixture
def trash(tmp_path: Path) -> SessionTrash:
    from pic_utils.results.trash import prepare_trash

    return prepare_trash(tmp_path)


@pytest.fixture
def grid_model(qtbot: object, run: RunStore, trash: SessionTrash) -> ResultsGridModel:
    model = ResultsGridModel(run, trash)
    return model


class TestModel:
    def test_lists_all_records(self, grid_model: ResultsGridModel) -> None:
        assert grid_model.rowCount() == 3
        ids = [
            grid_model.data(grid_model.index(row, 0), Qt.ItemDataRole.DisplayRole)
            for row in range(grid_model.rowCount())
        ]
        assert ids == ["0000:00", "0001:00", "0002:00"]

    def test_record_role(self, grid_model: ResultsGridModel) -> None:
        record = grid_model.data(grid_model.index(0, 0), ROLE_RECORD)
        assert isinstance(record, ResultRecord)
        assert record.id == "0000:00"

    def test_curation_state_from_disk(self, run: RunStore, trash: SessionTrash) -> None:
        run.set_curation("0000:00", "accepted")
        model = ResultsGridModel(run, trash)
        state = model.data(model.index(0, 0), ROLE_CURATION)
        assert state == "accepted"

    def test_discard_hides_row_until_toggled(
        self, grid_model: ResultsGridModel, run: RunStore
    ) -> None:
        grid_model.discard(0)
        assert grid_model.rowCount() == 2
        grid_model.set_show_discarded(True)
        assert grid_model.rowCount() == 3

    def test_discard_moves_files_to_trash(
        self, grid_model: ResultsGridModel, run: RunStore, tmp_path: Path
    ) -> None:
        record = grid_model._record_at(0)
        assert record is not None
        result_file = run.run_dir / record.path
        assert result_file.is_file()
        grid_model.discard(0)
        assert not result_file.exists()
        assert (run.run_dir / record.path).exists() is False
        curation = run.read_curation()
        assert curation[record.id] == "discarded"

    def test_accept_after_discard_restores_files(
        self, grid_model: ResultsGridModel, run: RunStore
    ) -> None:
        record = grid_model._record_at(0)
        assert record is not None
        result_file = run.run_dir / record.path
        grid_model.discard(0)
        assert not result_file.exists()
        grid_model.set_show_discarded(True)
        grid_model.accept(0)
        assert result_file.exists()
        assert run.read_curation()[record.id] == "accepted"

    def test_restore_returns_to_pending(self, grid_model: ResultsGridModel, run: RunStore) -> None:
        record = grid_model._record_at(0)
        assert record is not None
        result_file = run.run_dir / record.path
        grid_model.discard(0)
        grid_model.set_show_discarded(True)
        grid_model.restore(0)
        assert result_file.exists()
        assert record.id not in run.read_curation()

    def test_reload_reflects_disk(self, run: RunStore, trash: SessionTrash) -> None:
        model = ResultsGridModel(run, trash)
        run.set_curation("0001:00", "discarded")
        model.reload()
        assert model.rowCount() == 2


class TestGrid:
    def test_accept_shortcut(self, qtbot: object, run: RunStore, trash: SessionTrash) -> None:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        grid = ResultsGrid(run, trash)
        qtbot.addWidget(grid)  # type: ignore[attr-defined]
        grid._view.setCurrentIndex(grid._model.index(0, 0))
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        grid.keyPressEvent(event)
        assert run.read_curation()["0000:00"] == "accepted"

    def test_discard_shortcut(self, qtbot: object, run: RunStore, trash: SessionTrash) -> None:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        grid = ResultsGrid(run, trash)
        qtbot.addWidget(grid)  # type: ignore[attr-defined]
        grid._view.setCurrentIndex(grid._model.index(0, 0))
        grid.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
        )
        assert run.read_curation()["0000:00"] == "discarded"

    def test_restore_shortcut(self, qtbot: object, run: RunStore, trash: SessionTrash) -> None:
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent

        grid = ResultsGrid(run, trash)
        qtbot.addWidget(grid)  # type: ignore[attr-defined]
        grid._view.setCurrentIndex(grid._model.index(0, 0))
        grid.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.NoModifier)
        )
        grid._model.set_show_discarded(True)
        grid._view.setCurrentIndex(grid._model.index(0, 0))
        grid.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier)
        )
        assert "0000:00" not in run.read_curation()

    def test_double_click_emits_view_requested(
        self, qtbot: object, run: RunStore, trash: SessionTrash
    ) -> None:
        grid = ResultsGrid(run, trash)
        qtbot.addWidget(grid)  # type: ignore[attr-defined]
        seen: list[tuple[str, object]] = []
        grid.view_requested.connect(lambda rid, record: seen.append((rid, record)))
        grid._on_double_clicked(grid._model.index(0, 0))
        assert seen and seen[0][0] == "0000:00"

    def test_show_discarded_toggle(self, qtbot: object, run: RunStore, trash: SessionTrash) -> None:
        grid = ResultsGrid(run, trash)
        qtbot.addWidget(grid)  # type: ignore[attr-defined]
        run.set_curation("0000:00", "discarded")
        grid.model().reload()
        assert grid.model().rowCount() == 2
        grid.model().set_show_discarded(True)
        assert grid.model().rowCount() == 3


class TestImageViewer:
    def test_loads_image_and_zooms(self, qtbot: object, tmp_path: Path) -> None:
        from pic_utils.ui.widgets.image_viewer import ImageViewer

        image_path = tmp_path / "full.png"
        _write_png(image_path, 128)
        viewer = ImageViewer()
        qtbot.addWidget(viewer)  # type: ignore[attr-defined]
        viewer.set_image_path(image_path)
        assert viewer._label.pixmap() is not None
        viewer.set_zoom(2.0)
        assert viewer.zoom() == 2.0
        assert viewer._label.pixmap().width() > 8

    def test_missing_image_shows_message(self, qtbot: object, tmp_path: Path) -> None:
        from pic_utils.ui.widgets.image_viewer import ImageViewer

        viewer = ImageViewer()
        qtbot.addWidget(viewer)  # type: ignore[attr-defined]
        viewer.set_image_path(tmp_path / "missing.png")
        pixmap = viewer._label.pixmap()
        assert pixmap is None or pixmap.isNull()


class TestCompareView:
    def test_set_images_loads_both(self, qtbot: object, tmp_path: Path) -> None:
        from pic_utils.ui.widgets.compare_view import CompareView

        left = tmp_path / "a.png"
        right = tmp_path / "b.png"
        _write_png(left, 10)
        _write_png(right, 200)
        view = CompareView()
        qtbot.addWidget(view)  # type: ignore[attr-defined]
        view.set_images(str(left), str(right))
        qtbot.waitUntil(  # type: ignore[attr-defined]
            lambda: view._left.pixmap() is not None and view._right.pixmap() is not None,
            timeout=5_000,
        )

    def test_swap_swaps_labels(self, qtbot: object, tmp_path: Path) -> None:
        from pic_utils.ui.widgets.compare_view import CompareView

        left = tmp_path / "a.png"
        right = tmp_path / "b.png"
        _write_png(left, 10)
        _write_png(right, 200)
        view = CompareView()
        qtbot.addWidget(view)  # type: ignore[attr-defined]
        view.set_images(str(left), str(right))
        before = (view._left_label.text(), view._right_label.text())
        view._swap.click()
        assert (view._left_label.text(), view._right_label.text()) == (before[1], before[0])

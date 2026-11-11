"""Canvas save/load/export action tests (P6-05)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.export import ExportOptions
from pic_utils.canvas.model import CanvasDoc, CanvasLayer, assets_dir_for
from pic_utils.core.image import Image
from pic_utils.settings import Settings
from pic_utils.ui.canvas.actions import CanvasActions


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(path=tmp_path / "settings.ini")
    value.set_results_root(tmp_path / "results")
    return value


@pytest.fixture
def actions(settings: Settings) -> CanvasActions:
    return CanvasActions(settings)


@pytest.fixture
def doc(tmp_path: Path) -> CanvasDoc:
    store = AssetStore(tmp_path / "assets")
    document = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
    document.add_layer(CanvasLayer(id="a", name="a", image_ref=store.put_image(solid((255, 0, 0)))))
    return document


class TestSaveLoad:
    def test_save_and_reopen(
        self,
        actions: CanvasActions,
        doc: CanvasDoc,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "collage.pucanvas"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
        saved = actions.save(doc)
        assert saved == path
        assert path.is_file()
        assert assets_dir_for(path).is_dir()

        loaded = actions.open(path)
        assert loaded is not None
        assert [layer.id for layer in loaded.layers] == ["a"]

    def test_save_as(
        self,
        actions: CanvasActions,
        doc: CanvasDoc,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        first = tmp_path / "first.pucanvas"
        second = tmp_path / "second.pucanvas"
        choices = iter([(str(first), ""), (str(second), "")])
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: next(choices))
        actions.save(doc)
        actions.save_as(doc)
        assert first.is_file()
        assert second.is_file()

    def test_open_missing_assets_reports(
        self,
        actions: CanvasActions,
        doc: CanvasDoc,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "collage.pucanvas"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
        actions.save(doc)
        reference = doc.layers[0].image_ref
        (assets_dir_for(path) / reference).unlink()
        messages: list[str] = []
        actions.notify.connect(messages.append)
        assert actions.open(path) is None
        assert messages and reference in messages[0]

    def test_saved_document_is_clean(
        self,
        actions: CanvasActions,
        doc: CanvasDoc,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "collage.pucanvas"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
        doc.dirty = True
        actions.save(doc)
        assert not doc.dirty


class TestExport:
    def test_sync_export(self, actions: CanvasActions, doc: CanvasDoc, tmp_path: Path) -> None:
        options = ExportOptions(target_dir=tmp_path / "out", fmt="png", scope="canvas")
        report = actions.export(doc, options)
        assert report is not None
        assert report.files[0].is_file()

    def test_async_export(
        self,
        actions: CanvasActions,
        doc: CanvasDoc,
        tmp_path: Path,
        qtbot: object,
    ) -> None:
        options = ExportOptions(target_dir=tmp_path / "out", fmt="png", scope="canvas")
        with qtbot.waitSignal(actions.exported, timeout=10_000) as blocker:  # type: ignore[attr-defined]
            actions.export_async(doc, options)
        assert blocker.args  # type: ignore[attr-defined]
        assert actions.wait_for_export(10_000)

    def test_export_failure_reports(
        self, actions: CanvasActions, doc: CanvasDoc, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("x", encoding="utf-8")
        messages: list[str] = []
        actions.notify.connect(messages.append)
        options = ExportOptions(target_dir=blocker, fmt="png", scope="canvas")
        assert actions.export(doc, options) is None
        assert messages

    def test_collision_safe_folders(
        self, actions: CanvasActions, doc: CanvasDoc, tmp_path: Path
    ) -> None:
        options = ExportOptions(target_dir=tmp_path / "out")
        first = actions.export(doc, options)
        second = actions.export(doc, options)
        assert first is not None and second is not None
        assert first.folder != second.folder

    def test_export_scope_uses_accepted(
        self, actions: CanvasActions, doc: CanvasDoc, tmp_path: Path
    ) -> None:
        from pic_utils.io.saver import save_image

        accepted = tmp_path / "result_0001_00.png"
        save_image(solid((1, 2, 3)), accepted)
        options = ExportOptions(target_dir=tmp_path / "out", fmt="png", scope="canvas_and_accepted")
        report = actions.export(doc, options, accepted=(accepted,))
        assert report is not None
        assert (report.folder / "results" / accepted.name).is_file()

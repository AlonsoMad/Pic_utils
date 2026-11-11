"""Canvas page composition and shell integration tests (P6-07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.export import ExportOptions
from pic_utils.canvas.model import CanvasDoc, CanvasLayer, save_canvas
from pic_utils.core.image import Image
from pic_utils.io.saver import save_image
from pic_utils.results.store import AcceptedIndex
from pic_utils.runner.manifest import ResultRecord, RunStore
from pic_utils.settings import Settings
from pic_utils.ui.canvas.asset_tray import AssetTray
from pic_utils.ui.canvas.export_dialog import ExportDialog
from pic_utils.ui.canvas.page import CanvasPage
from pic_utils.ui.i18n import LanguageManager
from pic_utils.ui.main_window import MainWindow


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(path=tmp_path / "settings.ini")
    value.set_results_root(tmp_path / "results")
    return value


@pytest.fixture
def page(qtbot: object, settings: Settings) -> CanvasPage:
    widget = CanvasPage(settings, BlockRegistry(None))
    qtbot.addWidget(widget)  # type: ignore[attr-defined]
    return widget


def make_accepted_result(root: Path) -> Path:
    store = RunStore(root, "20260101-000000-abcdef")
    store.create({})
    result_path = store.result_file(1, 0)
    thumb_path = store.thumb_file(1, 0)
    save_image(solid((255, 0, 0)), result_path)
    save_image(solid((255, 0, 0)), thumb_path)
    record = ResultRecord(
        id="0001:00",
        experiment=1,
        layer=0,
        source=0,
        kind="image",
        path="results/result_0001_00.png",
        thumb="thumbs/result_0001_00.png",
        width=8,
        height=8,
        seed=0,
        params={},
        elapsed_ms=0,
    )
    store.append_result(record)
    store.set_curation(record.id, "accepted")
    return result_path


class TestComposition:
    def test_panels_present(self, page: CanvasPage) -> None:
        assert page.asset_tray() is not None
        assert page.canvas_view() is not None
        assert page.layer_list() is not None
        assert page.layer_properties() is not None
        assert page.canvas_actions() is not None

    def test_no_placeholder_labels(self, page: CanvasPage) -> None:
        names = {label.objectName() for label in page.findChildren(QLabel)}
        assert "canvas_asset_tray" not in names
        assert "canvas_view" not in names
        assert "canvas_layer_list" not in names
        assert "canvas_layer_properties" not in names

    def test_asset_tray_lists_accepted_results(self, qtbot: object, settings: Settings) -> None:
        make_accepted_result(settings.results_root)
        tray = AssetTray(AcceptedIndex(settings.results_root))
        qtbot.addWidget(tray)  # type: ignore[attr-defined]
        assert tray._list.count() == 1
        assert tray.current_path() is None
        tray._list.setCurrentRow(0)
        assert tray.current_path() is not None

    def test_asset_tray_skips_missing_result(self, qtbot: object, settings: Settings) -> None:
        result = make_accepted_result(settings.results_root)
        result.unlink()
        tray = AssetTray(AcceptedIndex(settings.results_root))
        qtbot.addWidget(tray)  # type: ignore[attr-defined]
        assert tray._list.count() == 0

    def test_asset_tray_missing_thumb_is_iconless(self, qtbot: object, settings: Settings) -> None:
        result = make_accepted_result(settings.results_root)
        thumb = result.parent.parent / "thumbs" / result.name
        thumb.unlink()
        tray = AssetTray(AcceptedIndex(settings.results_root))
        qtbot.addWidget(tray)  # type: ignore[attr-defined]
        assert tray._list.count() == 1
        assert tray._list.item(0).icon().isNull()

    def test_asset_tray_loads_thumbnail_async(self, qtbot: object, settings: Settings) -> None:
        make_accepted_result(settings.results_root)
        tray = AssetTray(AcceptedIndex(settings.results_root))
        qtbot.addWidget(tray)  # type: ignore[attr-defined]
        qtbot.waitUntil(  # type: ignore[attr-defined]
            lambda: tray._list.count() == 1 and not tray._list.item(0).icon().isNull(),
            timeout=5_000,
        )

    def test_save_action(
        self,
        page: CanvasPage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qtbot: object,
    ) -> None:
        path = tmp_path / "page.pucanvas"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(path), ""))
        with qtbot.waitSignal(page.canvas_actions().saved, timeout=10_000):  # type: ignore[attr-defined]
            page.save_canvas()
        assert path.is_file()

    def test_export_action(
        self,
        page: CanvasPage,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        qtbot: object,
    ) -> None:
        options = ExportOptions(target_dir=tmp_path / "out", fmt="png", scope="canvas")
        monkeypatch.setattr(ExportDialog, "exec", lambda self: ExportDialog.DialogCode.Accepted)
        monkeypatch.setattr(ExportDialog, "options", lambda self: options)
        with qtbot.waitSignal(page.canvas_actions().exported, timeout=10_000):  # type: ignore[attr-defined]
            page.export_canvas()
        assert (tmp_path / "out").is_dir()


class TestShell:
    def test_canvas_page_is_wired(self, qtbot: object, settings: Settings) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        assert window.canvas_page() is not None
        assert window._save_canvas is not None
        assert window._export_canvas is not None

    def test_mode_switch_shows_canvas(self, qtbot: object, settings: Settings) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        window._mode_canvas.trigger()
        assert window._stack.currentWidget() is window.canvas_page()

    def test_new_canvas_action_clears_document(
        self,
        qtbot: object,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        image = tmp_path / "layer.png"
        save_image(solid((1, 2, 3)), image)
        window.canvas_page()._add_layer_from_path(image)
        assert window.canvas_page().doc().layers
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
        )
        window._new_canvas.trigger()
        assert window.canvas_page().doc().layers == []

    def test_run_controls_hidden_in_canvas_mode(self, qtbot: object, settings: Settings) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        window._mode_canvas.trigger()
        assert window._run_controls_host.isHidden()
        window._mode_lab.trigger()
        assert not window._run_controls_host.isHidden()

    def test_close_event_cancel_keeps_window(
        self,
        qtbot: object,
        settings: Settings,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        image = tmp_path / "layer.png"
        save_image(solid((1, 2, 3)), image)
        window.canvas_page()._add_layer_from_path(image)
        monkeypatch.setattr(
            QMessageBox,
            "question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
        )
        event = QCloseEvent()
        window.closeEvent(event)
        assert not event.isAccepted()

    def test_canvas_open_signal_loads_document(
        self, qtbot: object, settings: Settings, tmp_path: Path
    ) -> None:
        languages = LanguageManager(settings)
        window = MainWindow(settings, languages)
        qtbot.addWidget(window)  # type: ignore[attr-defined]
        store = AssetStore(tmp_path / "assets")
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(CanvasLayer(id="a", name="a", image_ref=store.put_image(solid((1, 2, 3)))))
        path = tmp_path / "open.pucanvas"
        save_canvas(path, doc)
        window.canvas_open_requested.emit(path)
        assert [layer.id for layer in window.canvas_page().doc().layers] == ["a"]

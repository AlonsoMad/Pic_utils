"""Canvas UI tests: panels, layer operations, masks, and shortcuts (P6-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QDialog, QInputDialog, QSplitter

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.image import Image, Mask
from pic_utils.io.saver import save_image
from pic_utils.settings import Settings
from pic_utils.ui.canvas.page import CanvasPage
from pic_utils.ui.dialogs.mask_dialog import MaskDialog


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


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        path = tmp_path / f"image_{index}.png"
        save_image(solid(color, size=10 + index), path)
        paths.append(path)
    return paths


def add_layers(page: CanvasPage, paths: list[Path]) -> list[str]:
    for path in paths:
        page._add_layer_from_path(path)
    return [layer.id for layer in page.doc().layers]


class TestComposition:
    def test_all_panels_present(self, page: CanvasPage) -> None:
        assert page.asset_tray() is not None
        assert page.canvas_view() is not None
        assert page.layer_list() is not None
        assert page.layer_properties() is not None

    def test_splitters_layout(self, page: CanvasPage) -> None:
        assert len(page.findChildren(QSplitter)) >= 2

    def test_undo_limit_from_settings(self, page: CanvasPage, settings: Settings) -> None:
        assert page.undo_stack().undoLimit() == settings.undo_limit

    def test_retranslate(self, page: CanvasPage) -> None:
        page.retranslate()
        assert page.layer_list()._up.text()


class TestLayers:
    def test_add_layer_from_file(self, page: CanvasPage, image_paths: list[Path]) -> None:
        page._add_layer_from_path(image_paths[0])
        assert len(page.doc().layers) == 1
        assert page.doc().size == (10, 10)
        assert page.layer_list().model().rowCount() == 1
        assert page.layer_properties().isEnabled()

    def test_layer_list_is_reverse_z_order(self, page: CanvasPage, image_paths: list[Path]) -> None:
        ids = add_layers(page, image_paths[:2])
        model = page.layer_list().model()
        assert model.layer_id_at(0) == ids[1]
        assert model.layer_id_at(1) == ids[0]

    def test_selection_survives_commands(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        assert page.layer_properties().selected_layer_id() == layer_id
        page._on_property_changed(layer_id, "opacity", 0.5)
        assert page.layer_properties().selected_layer_id() == layer_id

    def test_visibility_toggle_is_undoable(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._on_visibility_toggled(layer_id, False)
        assert not page.doc().layer(layer_id).visible
        page.undo_stack().undo()
        assert page.doc().layer(layer_id).visible

    def test_property_change_is_undoable(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._on_property_changed(layer_id, "opacity", 0.4)
        assert page.doc().layer(layer_id).opacity == pytest.approx(0.4)
        page.undo_stack().undo()
        assert page.doc().layer(layer_id).opacity == pytest.approx(1.0)

    def test_move_layer(self, page: CanvasPage, image_paths: list[Path]) -> None:
        ids = add_layers(page, image_paths)
        page._on_move_requested(0, 2)
        assert [layer.id for layer in page.doc().layers] == [ids[1], ids[2], ids[0]]
        page.undo_stack().undo()
        assert [layer.id for layer in page.doc().layers] == ids

    def test_group_and_ungroup(self, page: CanvasPage, image_paths: list[Path]) -> None:
        ids = add_layers(page, image_paths[:2])
        page._on_group_requested(ids)
        assert len(page.doc().groups) == 1
        group_id = next(iter(page.doc().groups))
        page._on_ungroup_requested(group_id)
        assert page.doc().groups == {}
        page.undo_stack().undo()
        assert group_id in page.doc().groups

    def test_group_properties_are_undoable(self, page: CanvasPage, image_paths: list[Path]) -> None:
        ids = add_layers(page, image_paths[:2])
        page._on_group_requested(ids)
        group_id = next(iter(page.doc().groups))
        page._select_layer(ids[0])
        assert page.layer_properties()._group_opacity.isEnabled()
        page._on_group_property_changed(group_id, "opacity", 0.5)
        page._on_group_property_changed(group_id, "visible", False)
        assert page.doc().groups[group_id].opacity == pytest.approx(0.5)
        assert not page.doc().groups[group_id].visible
        page.undo_stack().undo()
        assert page.doc().groups[group_id].visible
        page.undo_stack().undo()
        assert page.doc().groups[group_id].opacity == pytest.approx(1.0)

    def test_crop_offsets_layers(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._on_property_changed(layer_id, "x", 5.0)
        page._on_crop_requested(QRectF(2, 2, 6, 6))
        assert page.doc().size == (6, 6)
        assert page.doc().layer(layer_id).x == pytest.approx(3.0)
        page.undo_stack().undo()
        assert page.doc().size == (10, 10)

    def test_resize_dialog(self, page: CanvasPage, monkeypatch: pytest.MonkeyPatch) -> None:
        values = iter([(20, True), (30, True)])
        monkeypatch.setattr(QInputDialog, "getInt", lambda *args, **kwargs: next(values))
        page._resize_canvas_dialog()
        assert page.doc().size == (20, 30)
        page.undo_stack().undo()
        assert page.doc().size == (1, 1)

    def test_crop_button_enters_crop_mode(self, page: CanvasPage) -> None:
        page._crop_button.click()
        assert page.canvas_view()._crop_mode
        page.canvas_view().cancel_crop()

    def test_nudge_selected_layer(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._select_layer(layer_id)
        page._on_nudge_requested(5.0, -2.0)
        assert page.doc().layer(layer_id).x == pytest.approx(5.0)
        assert page.doc().layer(layer_id).y == pytest.approx(-2.0)

    def test_delete_selected_layer(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._select_layer(layer_id)
        page._delete_selected()
        assert page.doc().layers == []

    def test_delete_shortcut(self, page: CanvasPage, image_paths: list[Path]) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        page._select_layer(layer_id)
        event = QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        )
        page.canvas_view().keyPressEvent(event)
        assert page.doc().layers == []


class TestMasks:
    def test_mask_dialog_computes_a_mask(self, page: CanvasPage, image_paths: list[Path]) -> None:
        image = solid((255, 0, 0), size=8)
        dialog = MaskDialog(image, page._registry.snapshot())
        dialog._block_combo.setCurrentText("channel_mask")
        mask = dialog._compute_mask()
        assert isinstance(mask, Mask)
        assert mask.data.shape == (8, 8)

    def test_apply_mask_via_page(
        self,
        page: CanvasPage,
        image_paths: list[Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        layer_id = add_layers(page, image_paths[:1])[0]
        mask = Mask(np.full((10, 10), 128, dtype=np.uint8))
        monkeypatch.setattr(MaskDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        monkeypatch.setattr(MaskDialog, "result_mask", lambda self: mask)
        page._on_mask_requested(layer_id)
        reference = page.doc().layer(layer_id).mask_ref
        assert reference is not None
        assert reference.endswith(".mask.png")
        page._on_mask_removed(layer_id)
        assert page.doc().layer(layer_id).mask_ref is None


class TestAssetTray:
    def test_add_file_signal(
        self, page: CanvasPage, image_paths: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(
            QFileDialog, "getOpenFileName", lambda *args, **kwargs: (str(image_paths[0]), "")
        )
        received: list[Path] = []
        page.asset_tray().add_requested.connect(received.append)
        page.asset_tray()._add_file_dialog()
        assert received == [image_paths[0]]


class TestExportDialog:
    def test_options(self, settings: Settings) -> None:
        from pic_utils.ui.canvas.export_dialog import ExportDialog

        dialog = ExportDialog(settings)
        dialog._format.setCurrentIndex(1)
        dialog._quality.setValue(75)
        dialog._scope.setCurrentIndex(2)
        dialog._target.setText(str(settings.results_root))
        options = dialog.options()
        assert options.fmt == "jpeg"
        assert options.jpeg_quality == 75
        assert options.scope == "everything"

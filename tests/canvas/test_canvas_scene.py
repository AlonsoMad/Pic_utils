"""Scene, transform, blend, and undo-command tests (P6-02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.canvas.render import layer_transform, local_transform
from pic_utils.canvas.scene import CanvasScene
from pic_utils.core.image import BlendMode, Image, Mask
from pic_utils.ui.canvas.commands import (
    AddLayerCommand,
    ApplyMaskCommand,
    CropCanvasCommand,
    GroupCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    ResizeCanvasCommand,
    SetGroupPropertyCommand,
    SetLayerGroupCommand,
    SetLayerPropertyCommand,
    TransformLayerCommand,
    UngroupCommand,
)

pytestmark = pytest.mark.usefixtures("qapp")


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


@pytest.fixture
def store(tmp_path: Path) -> AssetStore:
    return AssetStore(tmp_path / "assets")


def make_layer(
    store: AssetStore, layer_id: str, color=(255, 0, 0), **kwargs: object
) -> CanvasLayer:
    return CanvasLayer(
        id=layer_id,
        name=layer_id,
        image_ref=store.put_image(solid(color)),
        **kwargs,  # type: ignore[arg-type]
    )


def render_scene(scene: CanvasScene, width: int, height: int) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QPainter(image)
    try:
        scene.render(painter, QRectF(image.rect()), scene.sceneRect())
    finally:
        painter.end()
    return image


class TestTransforms:
    def test_local_transform_identity(self) -> None:
        layer = CanvasLayer(id="l", name="l", image_ref="a.png")
        transform = local_transform(layer, 10, 10)
        assert transform.map(QPointF(0, 0)) == QPointF(0, 0)
        assert transform.map(QPointF(10, 10)) == QPointF(10, 10)

    def test_local_transform_scales_about_center(self) -> None:
        layer = CanvasLayer(id="l", name="l", image_ref="a.png", scale=2.0)
        transform = local_transform(layer, 10, 10)
        assert transform.map(QPointF(5, 5)) == QPointF(5, 5)
        assert transform.map(QPointF(0, 0)) == QPointF(-5, -5)

    def test_layer_transform_translates(self) -> None:
        layer = CanvasLayer(id="l", name="l", image_ref="a.png", x=100.0, y=50.0)
        transform = layer_transform(layer, 10, 10)
        assert transform.map(QPointF(5, 5)) == QPointF(105, 55)
        assert transform.map(QPointF(0, 0)) == QPointF(100, 50)

    def test_flip_maps_about_center(self) -> None:
        layer = CanvasLayer(id="l", name="l", image_ref="a.png", flip_h=True)
        transform = local_transform(layer, 10, 10)
        assert transform.map(QPointF(0, 0)) == QPointF(10, 0)


class TestScene:
    def test_items_and_z_order(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        for layer_id in ("a", "b", "c"):
            doc.add_layer(make_layer(store, layer_id))
        scene = CanvasScene()
        scene.set_doc(doc)
        assert scene.sceneRect() == QRectF(0, 0, 8, 8)
        assert scene.layer_item("a").zValue() == 0  # type: ignore[union-attr]
        assert scene.layer_item("c").zValue() == 2  # type: ignore[union-attr]

    def test_group_opacity_and_visibility(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a", opacity=0.5))
        doc.add_layer(make_layer(store, "b"))
        group = doc.group_layers(["a", "b"])
        group.opacity = 0.5
        group.visible = False
        scene = CanvasScene()
        scene.set_doc(doc)
        item = scene.layer_item("a")
        assert item is not None
        assert item.opacity() == pytest.approx(0.25)
        assert not item.isVisible()

    def test_mask_multiplies_alpha(self, store: AssetStore) -> None:
        mask = Mask(np.array([[0, 255], [0, 255]], dtype=np.uint8))
        mask_ref = store.put_mask(mask)
        doc = CanvasDoc(size=(2, 2), background=None, assets_dir=store.assets_dir)
        layer = CanvasLayer(
            id="a",
            name="a",
            image_ref=store.put_image(solid((255, 0, 0), size=2)),
            mask_ref=mask_ref,
        )
        doc.add_layer(layer)
        scene = CanvasScene()
        scene.set_doc(doc)
        image = render_scene(scene, 2, 2)
        assert image.pixelColor(0, 0).alpha() == 0
        assert image.pixelColor(1, 0).alpha() == 255

    def test_scene_blend_modes(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(4, 4), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (0, 0, 255)))
        doc.add_layer(make_layer(store, "fg", (255, 0, 0), blend=BlendMode.MULTIPLY))
        scene = CanvasScene()
        scene.set_doc(doc)
        assert render_scene(scene, 4, 4).pixelColor(1, 1).name() == "#000000"

    def test_transform_handle_emits_scale(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(10, 10), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        scene = CanvasScene()
        scene.set_doc(doc)
        item = scene.layer_item("a")
        assert item is not None
        item.setSelected(True)
        received: list[tuple[str, float, float]] = []
        item.transform_finished.connect(
            lambda layer_id, scale, rotation: received.append((layer_id, scale, rotation))
        )
        center = item.mapToScene(item.boundingRect().center())
        item.handle_press("se", QPointF(center.x() + 10, center.y() + 10))
        item.handle_drag("se", QPointF(center.x() + 20, center.y() + 20), final=True)
        assert received
        assert received[0][0] == "a"
        assert received[0][1] == pytest.approx(2.0)

    def test_proxy_is_drawn_at_natural_size(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), background=None, assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a", (255, 0, 0)))
        scene = CanvasScene()
        scene.set_proxy_long_side(4)
        scene.set_doc(doc)
        item = scene.layer_item("a")
        assert item is not None
        assert item.boundingRect() == QRectF(0, 0, 8, 8)
        image = render_scene(scene, 8, 8)
        assert image.pixelColor(7, 7).alpha() == 255


class TestCommands:
    def test_add_and_remove(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        layer = make_layer(store, "a")
        add = AddLayerCommand(doc, layer, natural_size=(8, 8))
        add.redo()
        assert doc.size == (8, 8)
        assert doc.layer("a") is layer
        add.undo()
        assert doc.layers == []
        add.redo()
        remove = RemoveLayerCommand(doc, "a")
        remove.redo()
        assert doc.layers == []
        remove.undo()
        assert doc.layer("a") is layer

    def test_move_layer(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        for layer_id in ("a", "b", "c"):
            doc.add_layer(make_layer(store, layer_id))
        command = MoveLayerCommand(doc, 0, 2)
        command.redo()
        assert [item.id for item in doc.layers] == ["b", "c", "a"]
        command.undo()
        assert [item.id for item in doc.layers] == ["a", "b", "c"]

    def test_transform_command_and_merge(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        first = TransformLayerCommand(doc, "a", 1.0, 2.0, 1.0, 0.0)
        first.redo()
        second = TransformLayerCommand(doc, "a", 3.0, 4.0, 2.0, 10.0)
        assert first.id() == second.id()
        assert first.mergeWith(second)
        first.redo()
        assert (doc.layer("a").x, doc.layer("a").y) == (3.0, 4.0)
        first.undo()
        assert (doc.layer("a").x, doc.layer("a").y) == (0.0, 0.0)

    def test_property_command_and_merge(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        first = SetLayerPropertyCommand(doc, "a", "opacity", 0.5)
        first.redo()
        second = SetLayerPropertyCommand(doc, "a", "opacity", 0.25)
        assert first.mergeWith(second)
        first.redo()
        assert doc.layer("a").opacity == pytest.approx(0.25)
        first.undo()
        assert doc.layer("a").opacity == pytest.approx(1.0)

    def test_group_property_command_and_merge(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        group = doc.group_layers(["a"])
        first = SetGroupPropertyCommand(doc, group.id, "opacity", 0.5)
        first.redo()
        second = SetGroupPropertyCommand(doc, group.id, "opacity", 0.25)
        assert first.mergeWith(second)
        first.redo()
        assert doc.groups[group.id].opacity == pytest.approx(0.25)
        first.undo()
        assert doc.groups[group.id].opacity == pytest.approx(1.0)

    def test_set_layer_group_command(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        for layer_id in ("a", "b", "c"):
            doc.add_layer(make_layer(store, layer_id))
        group = doc.group_layers(["a", "b"])
        command = SetLayerGroupCommand(doc, "c", group.id)
        command.redo()
        assert doc.layer("c").group_id == group.id
        assert [layer.id for layer in doc.layers] == ["a", "b", "c"]
        command.undo()
        assert doc.layer("c").group_id is None
        assert [layer.id for layer in doc.layers] == ["a", "b", "c"]

    def test_apply_mask_command(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        command = ApplyMaskCommand(doc, "a", "mask.mask.png")
        command.redo()
        assert doc.layer("a").mask_ref == "mask.mask.png"
        command.undo()
        assert doc.layer("a").mask_ref is None

    def test_crop_command(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(100, 100), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a", x=10.0, y=20.0))
        command = CropCanvasCommand(doc, 5.0, 5.0, 50.0, 60.0)
        command.redo()
        assert doc.size == (50, 60)
        assert (doc.layer("a").x, doc.layer("a").y) == (5.0, 15.0)
        command.undo()
        assert doc.size == (100, 100)
        assert (doc.layer("a").x, doc.layer("a").y) == (10.0, 20.0)

    def test_resize_command(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(100, 100), assets_dir=store.assets_dir)
        command = ResizeCanvasCommand(doc, 20, 30)
        command.redo()
        assert doc.size == (20, 30)
        command.undo()
        assert doc.size == (100, 100)

    def test_group_and_ungroup_commands(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        for layer_id in ("a", "b", "c"):
            doc.add_layer(make_layer(store, layer_id))
        group = GroupCommand(doc, ["a", "c"])
        group.redo()
        assert [item.id for item in doc.layers] == ["b", "a", "c"]
        assert len(doc.groups) == 1
        group.undo()
        assert [item.id for item in doc.layers] == ["a", "b", "c"]
        assert doc.groups == {}

        group.redo()
        group_id = next(iter(doc.groups))
        ungroup = UngroupCommand(doc, group_id)
        ungroup.redo()
        assert doc.groups == {}
        ungroup.undo()
        assert group_id in doc.groups
        assert [item.id for item in doc.layers] == ["b", "a", "c"]

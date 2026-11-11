"""Full-resolution export renderer tests (P6-03, §10.7, §11)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage, QPainter

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.export import (
    EXPORT_FORMAT,
    _LayerLoader,
    export_canvas,
    render_canvas,
)
from pic_utils.canvas.model import CanvasDoc, CanvasLayer
from pic_utils.canvas.scene import CanvasScene
from pic_utils.core.errors import ExportError
from pic_utils.core.image import BlendMode, Image, Mask
from pic_utils.io.saver import save_image

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


class TestRender:
    def test_pixel_parity_with_scene_at_1x1(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (0, 0, 255)))
        doc.add_layer(make_layer(store, "fg", (255, 0, 0), blend=BlendMode.MULTIPLY))
        scene = CanvasScene()
        scene.set_doc(doc)
        screen = render_scene(scene, 8, 8)
        exported = render_canvas(doc)
        for x in (0, 3, 7):
            for y in (0, 3, 7):
                assert screen.pixelColor(x, y) == exported.pixelColor(x, y)

    @pytest.mark.parametrize(
        ("blend", "expected"),
        [
            (BlendMode.NORMAL, "#ff0000"),
            (BlendMode.MULTIPLY, "#000000"),
            (BlendMode.SCREEN, "#ff00ff"),
            (BlendMode.OVERLAY, "#0000ff"),
            (BlendMode.ADD, "#ff00ff"),
            (BlendMode.DIFFERENCE, "#ff00ff"),
        ],
    )
    def test_six_blend_modes(self, store: AssetStore, blend: BlendMode, expected: str) -> None:
        doc = CanvasDoc(size=(4, 4), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (0, 0, 255)))
        doc.add_layer(make_layer(store, "fg", (255, 0, 0), blend=blend))
        assert render_canvas(doc).pixelColor(1, 1).name() == expected

    def test_opacity_composes_with_blend(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(4, 4), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (0, 0, 255)))
        doc.add_layer(make_layer(store, "fg", (255, 0, 0), blend=BlendMode.MULTIPLY, opacity=0.5))
        assert render_canvas(doc).pixelColor(1, 1).name() == "#000080"

    def test_group_opacity_is_per_child(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(4, 4), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (255, 255, 255)))
        doc.add_layer(make_layer(store, "a", (255, 0, 0)))
        doc.add_layer(make_layer(store, "b", (255, 0, 0)))
        group = doc.group_layers(["a", "b"])
        group.opacity = 0.5
        assert render_canvas(doc).pixelColor(1, 1).name() == "#ff4040"

    def test_mask_multiplies_alpha(self, store: AssetStore) -> None:
        mask = Mask(np.array([[0, 255], [0, 255]], dtype=np.uint8))
        doc = CanvasDoc(size=(2, 2), background=None, assets_dir=store.assets_dir)
        doc.add_layer(
            CanvasLayer(
                id="a",
                name="a",
                image_ref=store.put_image(solid((255, 0, 0), size=2)),
                mask_ref=store.put_mask(mask),
            )
        )
        image = render_canvas(doc)
        assert image.pixelColor(0, 0).alpha() == 0
        assert image.pixelColor(1, 0).alpha() == 255

    def test_scale_changes_output_size(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 4), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "bg", (0, 0, 255)))
        image = render_canvas(doc, scale=0.5)
        assert (image.width(), image.height()) == (4, 2)

    def test_scale_preserves_layer_position(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(10, 10), background=None, assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a", (255, 0, 0), x=2.0, y=3.0))
        image = render_canvas(doc, scale=0.5)
        assert image.pixelColor(1, 2).alpha() == 255
        assert image.pixelColor(0, 2).alpha() == 0
        assert image.pixelColor(4, 4).alpha() == 255

    def test_layer_loader_cache_is_bounded(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        for index, layer_id in enumerate(("a", "b", "c")):
            doc.add_layer(make_layer(store, layer_id, (index * 60, 0, 0)))
        loader = _LayerLoader(doc, max_cached=2)
        for layer in doc.layers:
            loader.qimage(layer)
        assert loader.cache_size == 2


class TestExport:
    def test_canvas_only(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        report = export_canvas(doc, tmp_path, fmt="png", scope="canvas")
        assert report.folder.is_dir()
        assert [path.name for path in report.files] == ["canvas.png"]
        assert not (report.folder / "results").exists()
        manifest = json.loads((report.folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format"] == EXPORT_FORMAT
        assert manifest["scope"] == "canvas"
        assert manifest["image_format"] == "png"

    def test_canvas_and_accepted(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        result = tmp_path / "result_0001_00.png"
        save_image(solid((1, 2, 3)), result)
        report = export_canvas(
            doc,
            tmp_path / "out",
            fmt="png",
            scope="canvas_and_accepted",
            accepted=(result,),
        )
        assert (report.folder / "results" / "result_0001_00.png").is_file()
        assert len(report.files) == 2

    def test_everything_uses_run_results(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        results = []
        for index in range(2):
            path = tmp_path / f"result_{index}.png"
            save_image(solid((index, 0, 0)), path)
            results.append(path)
        report = export_canvas(
            doc,
            tmp_path / "out",
            fmt="png",
            scope="everything",
            run_results=tuple(results),
        )
        assert len(list((report.folder / "results").iterdir())) == 2

    @pytest.mark.parametrize(
        ("fmt", "extension"),
        [("png", ".png"), ("jpeg", ".jpg"), ("tiff", ".tiff")],
    )
    def test_formats_and_srgb_profile(
        self, tmp_path: Path, store: AssetStore, fmt: str, extension: str
    ) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        report = export_canvas(doc, tmp_path, fmt=fmt, jpeg_quality=80)
        assert report.files[0].name == f"canvas{extension}"
        with PILImage.open(report.files[0]) as image:
            assert image.info.get("icc_profile")

    def test_jpeg_quality_recorded(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        report = export_canvas(doc, tmp_path, fmt="jpeg", jpeg_quality=55)
        manifest = json.loads((report.folder / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["jpeg_quality"] == 55
        assert report.jpeg_quality == 55

    def test_collision_safe_folder(
        self, tmp_path: Path, store: AssetStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FixedDatetime:
            @classmethod
            def now(cls, tz=None):  # type: ignore[no-untyped-def]
                return datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz or UTC)

        import pic_utils.canvas.export as export_module

        monkeypatch.setattr(export_module, "datetime", FixedDatetime)
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        first = export_canvas(doc, tmp_path)
        second = export_canvas(doc, tmp_path)
        assert first.folder.name == "export_20260102-030405"
        assert second.folder.name == "export_20260102-030405-2"

    def test_result_name_collisions_are_suffixed(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        first = tmp_path / "a" / "result.png"
        second = tmp_path / "b" / "result.png"
        for path in (first, second):
            path.parent.mkdir()
            save_image(solid((1, 1, 1)), path)
        report = export_canvas(
            doc,
            tmp_path / "out",
            scope="canvas_and_accepted",
            accepted=(first, second),
        )
        names = sorted(path.name for path in (report.folder / "results").iterdir())
        assert names == ["result-2.png", "result.png"]

    def test_export_cancels_before_writing(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
        doc.add_layer(make_layer(store, "a"))
        with pytest.raises(ExportError, match="cancelled"):
            export_canvas(doc, tmp_path, cancel=lambda: True)

    def test_invalid_options_rejected(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        with pytest.raises(ValueError):
            export_canvas(doc, tmp_path, fmt="webp")
        with pytest.raises(ValueError):
            export_canvas(doc, tmp_path, scope="everything_else")

"""Canvas integration tests: save/load, render parity, and export (P6-06)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.export import export_canvas, render_canvas
from pic_utils.canvas.model import CanvasDoc, CanvasLayer, load_canvas, save_canvas
from pic_utils.core.image import BlendMode, Image, Mask
from pic_utils.io.saver import save_image

pytestmark = pytest.mark.usefixtures("qapp")


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


def build_document(store: AssetStore) -> CanvasDoc:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:, 4:] = 255
    mask_ref = store.put_mask(Mask(mask))
    doc = CanvasDoc(size=(8, 8), assets_dir=store.assets_dir)
    doc.add_layer(
        CanvasLayer(id="bg", name="bg", image_ref=store.put_image(solid((255, 255, 255))))
    )
    doc.add_layer(
        CanvasLayer(
            id="red",
            name="red",
            image_ref=store.put_image(solid((255, 0, 0))),
            mask_ref=mask_ref,
        )
    )
    doc.add_layer(
        CanvasLayer(
            id="blue",
            name="blue",
            image_ref=store.put_image(solid((0, 0, 255))),
            blend=BlendMode.MULTIPLY,
            opacity=0.5,
        )
    )
    group = doc.group_layers(["red", "blue"], name="Overlay")
    group.opacity = 0.75
    return doc


def test_save_load_render_and_export(tmp_path: Path) -> None:
    store = AssetStore(tmp_path / "source-assets")
    doc = build_document(store)
    original = render_canvas(doc)

    canvas_path = tmp_path / "collage.pucanvas"
    save_canvas(canvas_path, doc)
    loaded = load_canvas(canvas_path)

    assert [layer.id for layer in loaded.layers] == [layer.id for layer in doc.layers]
    assert loaded.groups[next(iter(loaded.groups))].opacity == 0.75
    restored = render_canvas(loaded)
    for x in range(8):
        for y in range(8):
            assert restored.pixelColor(x, y) == original.pixelColor(x, y)

    accepted = tmp_path / "result_0001_00.png"
    save_image(solid((10, 20, 30)), accepted)
    report = export_canvas(
        loaded,
        tmp_path / "exports",
        fmt="png",
        scope="canvas_and_accepted",
        accepted=(accepted,),
    )
    assert (report.folder / "canvas.png").is_file()
    assert (report.folder / "results" / "result_0001_00.png").is_file()

    loaded.layers[0].opacity = 0.25
    loaded.dirty = True
    save_canvas(canvas_path, loaded)
    reloaded = load_canvas(canvas_path)
    assert reloaded.layers[0].opacity == 0.25

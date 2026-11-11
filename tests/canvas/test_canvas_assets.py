"""Content-addressed asset store tests (P6-01)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pic_utils.canvas.assets import AssetStore
from pic_utils.core.errors import CanvasFileError
from pic_utils.core.image import Image, Mask
from pic_utils.io.loader import load_image, load_mask
from pic_utils.io.saver import save_image


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


class TestContentAddressing:
    def test_same_bytes_same_reference(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        first = store.put_png_bytes(b"same")
        second = store.put_png_bytes(b"same")
        assert first == second
        assert (store.assets_dir / first).is_file()

    def test_different_bytes_different_reference(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        assert store.put_png_bytes(b"one") != store.put_png_bytes(b"two")

    def test_image_round_trip(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        image = solid((10, 20, 30))
        reference = store.put_image(image)
        loaded = load_image(store.resolve(reference))
        assert np.array_equal(loaded.rgb, image.rgb)

    def test_mask_round_trip(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        mask = Mask(np.arange(64, dtype=np.uint8).reshape(8, 8))
        reference = store.put_mask(mask)
        assert reference.endswith(".mask.png")
        loaded = load_mask(store.resolve(reference))
        assert np.array_equal(loaded.data, mask.data)

    def test_put_file_normalizes_to_png(self, tmp_path: Path) -> None:
        source = tmp_path / "source.png"
        save_image(solid((1, 2, 3)), source)
        store = AssetStore(tmp_path / "assets")
        reference = store.put_file(source)
        assert reference.endswith(".png")
        assert np.array_equal(load_image(store.resolve(reference)).rgb, solid((1, 2, 3)).rgb)


class TestResolution:
    def test_missing_reference_raises(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        with pytest.raises(CanvasFileError, match="missing asset"):
            store.resolve("nope.png")

    def test_exists_is_false_for_missing(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        assert not store.exists("nope.png")

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"x")
        store = AssetStore(tmp_path / "assets")
        with pytest.raises(CanvasFileError, match="escapes"):
            store.resolve("../outside.png")


class TestGarbageCollection:
    def test_gc_keeps_referenced_and_removes_others(self, tmp_path: Path) -> None:
        store = AssetStore(tmp_path / "assets")
        keep = store.put_image(solid((1, 1, 1)))
        drop = store.put_image(solid((2, 2, 2)))
        store.gc({keep})
        assert (store.assets_dir / keep).is_file()
        assert not (store.assets_dir / drop).exists()

    def test_copy_from(self, tmp_path: Path) -> None:
        source = AssetStore(tmp_path / "source")
        target = AssetStore(tmp_path / "target")
        reference = source.put_image(solid((4, 5, 6)))
        target.copy_from(source, reference)
        assert np.array_equal(
            load_image(target.resolve(reference)).rgb,
            solid((4, 5, 6)).rgb,
        )

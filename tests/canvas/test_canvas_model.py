"""Canvas document model and file-format tests (P6-01)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pic_utils.canvas.assets import AssetStore
from pic_utils.canvas.model import (
    CANVAS_FORMAT,
    CANVAS_VERSION,
    CanvasDoc,
    CanvasGroup,
    CanvasLayer,
    assets_dir_for,
    load_canvas,
    save_canvas,
)
from pic_utils.core.errors import CanvasFileError
from pic_utils.core.image import BlendMode, Image, Mask


def solid(color: tuple[int, int, int], size: int = 8) -> Image:
    return Image.from_rgb(np.full((size, size, 3), color, dtype=np.uint8))


@pytest.fixture
def store(tmp_path: Path) -> AssetStore:
    return AssetStore(tmp_path / "assets")


def reference(store: AssetStore, color: tuple[int, int, int] = (255, 0, 0)) -> str:
    return store.put_image(solid(color))


def layer(
    store: AssetStore,
    layer_id: str = "l1",
    color: tuple[int, int, int] = (255, 0, 0),
    **kwargs: object,
) -> CanvasLayer:
    return CanvasLayer(
        id=layer_id,
        name=layer_id,
        image_ref=reference(store, color),
        **kwargs,  # type: ignore[arg-type]
    )


class TestModel:
    def test_defaults(self) -> None:
        doc = CanvasDoc()
        assert doc.size == (1, 1)
        assert doc.background == "#ffffff"
        assert doc.layers == []
        assert doc.groups == {}

    def test_layer_round_trip(self) -> None:
        original = CanvasLayer(
            id="l1",
            name="Layer",
            image_ref="abc.png",
            x=1.5,
            y=-2.0,
            scale=1.25,
            rotation=-8.0,
            flip_h=True,
            opacity=0.8,
            blend=BlendMode.SCREEN,
            mask_ref="mask.mask.png",
            group_id="g1",
            visible=False,
            locked=True,
        )
        assert CanvasLayer.from_dict(original.to_dict()) == original

    def test_group_round_trip(self) -> None:
        group = CanvasGroup(id="g1", name="Group", visible=False, opacity=0.5)
        assert CanvasGroup.from_dict(group.to_dict()) == group

    def test_document_schema(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(20, 10), assets_dir=store.assets_dir)
        doc.add_layer(layer(store))
        payload = doc.to_dict()
        assert payload["format"] == CANVAS_FORMAT
        assert payload["version"] == CANVAS_VERSION
        assert payload["size"] == [20, 10]
        assert payload["layers"][0]["image"] == doc.layers[0].image_ref
        assert "mask" in payload["layers"][0]
        assert "group" in payload["layers"][0]

    def test_duplicate_layer_id_rejected(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store))
        with pytest.raises(ValueError, match="duplicate"):
            doc.add_layer(layer(store))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"opacity": 2.0},
            {"scale": 0.0},
            {"blend": "bogus"},
            {"x": "nope"},
        ],
    )
    def test_invalid_layer_values_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            CanvasLayer(id="l", name="l", image_ref="a.png", **kwargs)  # type: ignore[arg-type]


class TestGeometry:
    def test_canvas_expands_and_never_shrinks(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"), natural_size=(10, 10))
        assert doc.size == (10, 10)
        doc.add_layer(layer(store, "b"), natural_size=(20, 5))
        assert doc.size == (20, 10)
        doc.add_layer(layer(store, "c"), natural_size=(5, 5))
        assert doc.size == (20, 10)

    def test_crop_offsets_layers(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(100, 100), assets_dir=store.assets_dir)
        doc.add_layer(layer(store, x=10.0, y=20.0))
        doc.crop(5.0, 5.0, 50.0, 60.0)
        assert doc.size == (50, 60)
        assert (doc.layers[0].x, doc.layers[0].y) == (5.0, 15.0)

    def test_resize_keeps_positions(self, store: AssetStore) -> None:
        doc = CanvasDoc(size=(100, 100), assets_dir=store.assets_dir)
        doc.add_layer(layer(store, x=10.0, y=20.0))
        doc.resize(20, 30)
        assert doc.size == (20, 30)
        assert (doc.layers[0].x, doc.layers[0].y) == (10.0, 20.0)


class TestGroups:
    def test_grouping_makes_members_contiguous(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        for layer_id in ("a", "b", "c", "d"):
            doc.add_layer(layer(store, layer_id))
        group = doc.group_layers(["b", "d"])
        assert [item.id for item in doc.layers] == ["a", "c", "b", "d"]
        assert {item.group_id for item in doc.layers if item.id in ("b", "d")} == {group.id}

    def test_ungroup_clears_members(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"))
        doc.add_layer(layer(store, "b"))
        group = doc.group_layers(["a", "b"])
        doc.ungroup(group.id)
        assert doc.groups == {}
        assert all(item.group_id is None for item in doc.layers)

    def test_set_layer_group_same_group_is_a_noop(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"))
        doc.add_layer(layer(store, "b"))
        group = doc.group_layers(["a", "b"])
        order = [item.id for item in doc.layers]
        doc.set_layer_group("a", group.id)
        assert [item.id for item in doc.layers] == order

    def test_removing_last_member_drops_empty_group(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"))
        doc.add_layer(layer(store, "b"))
        group = doc.group_layers(["a", "b"])
        doc.set_layer_group("a", None)
        doc.set_layer_group("b", None)
        assert group.id not in doc.groups

    def test_group_needs_a_layer(self, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        with pytest.raises(ValueError, match="at least one"):
            doc.group_layers(["missing"])


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(size=(20, 20), background="#101010", assets_dir=store.assets_dir)
        mask_ref = store.put_mask(Mask(np.full((8, 8), 128, dtype=np.uint8)))
        doc.add_layer(layer(store, "a", mask_ref=mask_ref))
        doc.add_layer(layer(store, "b", blend=BlendMode.ADD, rotation=45.0))
        group = doc.group_layers(["a", "b"], name="Pair")

        path = tmp_path / "collage.canvas"
        save_canvas(path, doc)
        loaded = load_canvas(path)

        assert loaded.size == doc.size
        assert loaded.background == doc.background
        assert [item.id for item in loaded.layers] == ["a", "b"]
        assert loaded.layers[0].mask_ref == mask_ref
        assert loaded.layers[1].blend is BlendMode.ADD
        assert loaded.groups[group.id].name == "Pair"
        assert loaded.assets_dir == assets_dir_for(path)

    def test_save_adds_only_new_assets(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"))
        path = tmp_path / "canvas.canvas"
        save_canvas(path, doc)
        assets = assets_dir_for(path)
        first = {item.name for item in assets.iterdir()}
        save_canvas(path, doc)
        assert {item.name for item in assets.iterdir()} == first
        doc.add_layer(
            CanvasLayer(
                id="b",
                name="b",
                image_ref=AssetStore(doc.assets_dir).put_image(solid((0, 255, 0))),
            )
        )
        save_canvas(path, doc)
        second = {item.name for item in assets.iterdir()}
        assert first < second

    def test_missing_asset_refuses_open(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(layer(store, "a"))
        path = tmp_path / "canvas.canvas"
        save_canvas(path, doc)
        reference = doc.layers[0].image_ref
        (assets_dir_for(path) / reference).unlink()
        with pytest.raises(CanvasFileError) as error:
            load_canvas(path)
        assert reference in str(error.value)
        assert "missing assets" in str(error.value)

    def test_save_missing_asset_raises(self, tmp_path: Path, store: AssetStore) -> None:
        doc = CanvasDoc(assets_dir=store.assets_dir)
        doc.add_layer(CanvasLayer(id="a", name="a", image_ref="missing.png"))
        with pytest.raises(CanvasFileError, match="missing asset"):
            save_canvas(tmp_path / "canvas.canvas", doc)

    def test_wrong_format_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.canvas"
        path.write_text(json.dumps({"format": "other"}), encoding="utf-8")
        with pytest.raises(CanvasFileError, match="format"):
            load_canvas(path)

    def test_zero_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "v0.canvas"
        path.write_text(
            json.dumps({"format": CANVAS_FORMAT, "version": 0}),
            encoding="utf-8",
        )
        with pytest.raises(CanvasFileError, match="version"):
            load_canvas(path)

    def test_newer_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "future.canvas"
        path.write_text(
            json.dumps({"format": CANVAS_FORMAT, "version": 99}),
            encoding="utf-8",
        )
        with pytest.raises(CanvasFileError, match="version"):
            load_canvas(path)

    def test_assets_dir_derivation(self, tmp_path: Path) -> None:
        assert assets_dir_for(tmp_path / "a.canvas") == tmp_path / "a.canvas.assets"

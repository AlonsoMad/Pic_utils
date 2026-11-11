"""Canvas document model and file format (architecture §10.1, §10.8).

A :class:`CanvasDoc` is a flat, z-ordered list of :class:`CanvasLayer`
objects plus a dictionary of :class:`CanvasGroup` objects. The document is
mutable; undo commands in ``ui/canvas/commands.py`` mutate it through the
methods here and the scene re-syncs from the result.

The canvas file is JSON (``pic_utils.canvas``, version 1) with a sibling
content-addressed assets folder. The JSON write is the commit point: assets
are copied first, then the document is written atomically, then unreferenced
assets are collected best-effort (§10.2).
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pic_utils.canvas.assets import AssetStore
from pic_utils.core.color import parse_hex_color
from pic_utils.core.errors import CanvasFileError
from pic_utils.core.image import BlendMode
from pic_utils.core.jsonio import read_json, write_json_atomic

__all__ = [
    "CANVAS_FORMAT",
    "CANVAS_VERSION",
    "CanvasDoc",
    "CanvasGroup",
    "CanvasLayer",
    "assets_dir_for",
    "load_canvas",
    "save_canvas",
]

CANVAS_FORMAT = "pic_utils.canvas"
CANVAS_VERSION = 1


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _opacity(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {number}")
    return number


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


@dataclass(slots=True)
class CanvasGroup:
    """Visibility and opacity for a contiguous run of layers (§10.1)."""

    id: str
    name: str
    visible: bool = True
    opacity: float = 1.0

    def __post_init__(self) -> None:
        self.id = _require_text(self.id, "group id")
        self.name = _require_text(self.name, "group name")
        self.visible = _boolean(self.visible, "group visible")
        self.opacity = _opacity(self.opacity, "group opacity")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe group record (§10.8)."""
        return {
            "id": self.id,
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
        }

    @classmethod
    def from_dict(cls, raw: object) -> CanvasGroup:
        """Rebuild a group from its JSON record."""
        if not isinstance(raw, dict):
            raise ValueError("group must be a JSON object")
        try:
            return cls(
                id=raw["id"],
                name=raw["name"],
                visible=raw.get("visible", True),
                opacity=raw.get("opacity", 1.0),
            )
        except KeyError as error:
            raise ValueError(f"group is missing {error.args[0]!r}") from error


@dataclass(slots=True)
class CanvasLayer:
    """One placed image with its transform, blend, and optional mask."""

    id: str
    name: str
    image_ref: str
    x: float = 0.0
    y: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    opacity: float = 1.0
    blend: BlendMode = BlendMode.NORMAL
    mask_ref: str | None = None
    group_id: str | None = None
    visible: bool = True
    locked: bool = False

    def __post_init__(self) -> None:
        self.id = _require_text(self.id, "layer id")
        self.name = _require_text(self.name, "layer name")
        self.image_ref = _require_text(self.image_ref, "layer image")
        self.x = _number(self.x, "layer x")
        self.y = _number(self.y, "layer y")
        self.scale = _number(self.scale, "layer scale")
        if self.scale <= 0.0:
            raise ValueError(f"layer scale must be positive, got {self.scale}")
        self.rotation = _number(self.rotation, "layer rotation")
        self.flip_h = _boolean(self.flip_h, "layer flip_h")
        self.flip_v = _boolean(self.flip_v, "layer flip_v")
        self.opacity = _opacity(self.opacity, "layer opacity")
        if isinstance(self.blend, str):
            self.blend = BlendMode(self.blend)
        elif not isinstance(self.blend, BlendMode):
            raise ValueError(f"layer blend must be a BlendMode, got {self.blend!r}")
        if self.mask_ref is not None:
            self.mask_ref = _require_text(self.mask_ref, "layer mask")
        if self.group_id is not None:
            self.group_id = _require_text(self.group_id, "layer group")
        self.visible = _boolean(self.visible, "layer visible")
        self.locked = _boolean(self.locked, "layer locked")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe layer record (§10.8)."""
        return {
            "id": self.id,
            "name": self.name,
            "image": self.image_ref,
            "x": self.x,
            "y": self.y,
            "scale": self.scale,
            "rotation": self.rotation,
            "flip_h": self.flip_h,
            "flip_v": self.flip_v,
            "opacity": self.opacity,
            "blend": str(self.blend),
            "mask": self.mask_ref,
            "group": self.group_id,
            "visible": self.visible,
            "locked": self.locked,
        }

    @classmethod
    def from_dict(cls, raw: object) -> CanvasLayer:
        """Rebuild a layer from its JSON record."""
        if not isinstance(raw, dict):
            raise ValueError("layer must be a JSON object")
        try:
            return cls(
                id=raw["id"],
                name=raw["name"],
                image_ref=raw["image"],
                x=raw.get("x", 0.0),
                y=raw.get("y", 0.0),
                scale=raw.get("scale", 1.0),
                rotation=raw.get("rotation", 0.0),
                flip_h=raw.get("flip_h", False),
                flip_v=raw.get("flip_v", False),
                opacity=raw.get("opacity", 1.0),
                blend=raw.get("blend", BlendMode.NORMAL),
                mask_ref=raw.get("mask"),
                group_id=raw.get("group"),
                visible=raw.get("visible", True),
                locked=raw.get("locked", False),
            )
        except KeyError as error:
            raise ValueError(f"layer is missing {error.args[0]!r}") from error


@dataclass(slots=True)
class CanvasDoc:
    """One open canvas: size, background, layers, and groups (§10.1)."""

    size: tuple[int, int] = (1, 1)
    background: str | None = "#ffffff"
    layers: list[CanvasLayer] = field(default_factory=list)
    groups: dict[str, CanvasGroup] = field(default_factory=dict)
    assets_dir: Path | None = None
    dirty: bool = False
    _temporary_assets: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        width, height = self.size
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("canvas size must be two positive integers")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError(f"canvas size must be two positive integers, got {self.size!r}")
        self.size = (width, height)
        if self.background is not None:
            parse_hex_color(self.background)
        self.layers = list(self.layers)
        self.groups = dict(self.groups)
        if self.assets_dir is not None:
            self.assets_dir = Path(self.assets_dir)
        else:
            self._temporary_assets = False
        self.dirty = bool(self.dirty)

    # --- geometry -----------------------------------------------------------

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self) -> int:
        return self.size[1]

    def expand_for(self, width: int, height: int) -> bool:
        """Grow the canvas to fit a newly placed image; never shrink (§10.1)."""
        if width < 1 or height < 1:
            raise ValueError(f"image size must be positive, got {width}x{height}")
        grown = (max(self.width, width), max(self.height, height))
        if grown == self.size:
            return False
        self.size = grown
        self.dirty = True
        return True

    def crop(self, x: float, y: float, width: float, height: float) -> None:
        """Set the canvas size and shift every layer by ``-x, -y`` (§10.6)."""
        if width < 1 or height < 1:
            raise ValueError(f"crop size must be positive, got {width}x{height}")
        self.size = (round(width), round(height))
        for layer in self.layers:
            layer.x -= x
            layer.y -= y
        self.dirty = True

    def resize(self, width: int, height: int) -> None:
        """Set the canvas size; layers keep their coordinates (§10.6)."""
        if width < 1 or height < 1:
            raise ValueError(f"canvas size must be positive, got {width}x{height}")
        self.size = (int(width), int(height))
        self.dirty = True

    # --- layers -------------------------------------------------------------

    def layer(self, layer_id: str) -> CanvasLayer:
        """Return the layer with ``layer_id``; raises ``KeyError`` if absent."""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(f"unknown layer {layer_id!r}")

    def find_layer(self, layer_id: str) -> CanvasLayer | None:
        """Return the layer with ``layer_id`` or ``None``."""
        return next((layer for layer in self.layers if layer.id == layer_id), None)

    def layer_index(self, layer_id: str) -> int:
        """Return the z-order index of ``layer_id`` (``-1`` if absent)."""
        for index, layer in enumerate(self.layers):
            if layer.id == layer_id:
                return index
        return -1

    def add_layer(
        self,
        layer: CanvasLayer,
        *,
        natural_size: tuple[int, int] | None = None,
        index: int | None = None,
    ) -> None:
        """Insert ``layer``; the first layer initializes the canvas size."""
        if self.find_layer(layer.id) is not None:
            raise ValueError(f"duplicate layer id {layer.id!r}")
        if natural_size is not None:
            if not self.layers and self.size == (1, 1):
                self.size = (max(1, int(natural_size[0])), max(1, int(natural_size[1])))
            else:
                self.expand_for(*natural_size)
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(max(0, min(index, len(self.layers))), layer)
        self.dirty = True

    def remove_layer(self, layer_id: str) -> tuple[int, CanvasLayer]:
        """Remove a layer and return its old index and object."""
        index = self.layer_index(layer_id)
        if index < 0:
            raise KeyError(f"unknown layer {layer_id!r}")
        layer = self.layers.pop(index)
        self.dirty = True
        return index, layer

    def move_layer(self, from_index: int, to_index: int) -> None:
        """Move a layer to a new z-order index (§10.6)."""
        if not 0 <= from_index < len(self.layers):
            raise IndexError(f"layer index out of range: {from_index}")
        if not 0 <= to_index < len(self.layers):
            raise IndexError(f"layer index out of range: {to_index}")
        if from_index == to_index:
            return
        layer = self.layers.pop(from_index)
        self.layers.insert(to_index, layer)
        self.dirty = True

    # --- groups -------------------------------------------------------------

    def group_layers(self, layer_ids: list[str], name: str | None = None) -> CanvasGroup:
        """Create a group from ``layer_ids``; members become contiguous (§10.1)."""
        unique = list(dict.fromkeys(layer_ids))
        members = [layer for layer in (self.find_layer(lid) for lid in unique) if layer is not None]
        if not members:
            raise ValueError("a group needs at least one layer")
        member_ids = {layer.id for layer in members}
        top = max(self.layer_index(layer.id) for layer in members)
        insert_at = sum(
            1
            for index, layer in enumerate(self.layers)
            if index < top and layer.id not in member_ids
        )
        group = CanvasGroup(
            id=uuid.uuid4().hex,
            name=name or f"Group {len(self.groups) + 1}",
        )
        self.layers = [layer for layer in self.layers if layer.id not in member_ids]
        self.layers[insert_at:insert_at] = members
        for layer in members:
            layer.group_id = group.id
        self.groups[group.id] = group
        self._drop_empty_groups()
        self.dirty = True
        return group

    def ungroup(self, group_id: str) -> CanvasGroup:
        """Remove a group and clear its members' group reference."""
        group = self.groups.pop(group_id)
        for layer in self.layers:
            if layer.group_id == group_id:
                layer.group_id = None
        self.dirty = True
        return group

    def set_layer_group(self, layer_id: str, group_id: str | None) -> None:
        """Assign a layer to a group, keeping members contiguous."""
        layer = self.layer(layer_id)
        if group_id is None:
            layer.group_id = None
            self.dirty = True
            return
        if group_id not in self.groups:
            raise KeyError(f"unknown group {group_id!r}")
        self.layers.remove(layer)
        member_indices = [
            index for index, other in enumerate(self.layers) if other.group_id == group_id
        ]
        insert_at = max(member_indices) + 1 if member_indices else len(self.layers)
        self.layers.insert(insert_at, layer)
        layer.group_id = group_id
        self.dirty = True

    def _drop_empty_groups(self) -> None:
        used = {layer.group_id for layer in self.layers if layer.group_id is not None}
        for group_id in list(self.groups):
            if group_id not in used:
                del self.groups[group_id]

    # --- assets and persistence --------------------------------------------

    def referenced_assets(self) -> set[str]:
        """Every asset reference used by a layer image or mask."""
        references: set[str] = set()
        for layer in self.layers:
            references.add(layer.image_ref)
            if layer.mask_ref is not None:
                references.add(layer.mask_ref)
        return references

    def ensure_assets_dir(self) -> Path:
        """Return the assets folder, creating a temporary one for new docs."""
        if self.assets_dir is None:
            self.assets_dir = Path(tempfile.mkdtemp(prefix="pic_utils-canvas-"))
            self._temporary_assets = True
        return self.assets_dir

    def discard_temporary_assets(self) -> None:
        """Remove the temporary assets folder of an unsaved document."""
        if not self._temporary_assets:
            return
        if self.assets_dir is not None:
            shutil.rmtree(self.assets_dir, ignore_errors=True)
        self.assets_dir = None
        self._temporary_assets = False

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON document (§10.8)."""
        return {
            "format": CANVAS_FORMAT,
            "version": CANVAS_VERSION,
            "size": [self.width, self.height],
            "background": self.background,
            "groups": [group.to_dict() for group in self.groups.values()],
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, raw: object, *, assets_dir: Path | None = None) -> CanvasDoc:
        """Rebuild a document from its JSON record.

        Raises:
            CanvasFileError: wrong format/version or malformed records.
        """
        if not isinstance(raw, dict):
            raise CanvasFileError("canvas file must be a JSON object")
        if raw.get("format") != CANVAS_FORMAT:
            raise CanvasFileError(f"format must be {CANVAS_FORMAT!r}")
        version = raw.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version > CANVAS_VERSION:
            raise CanvasFileError(f"unsupported canvas version {version!r}")

        size = raw.get("size", [1, 1])
        if (
            not isinstance(size, list | tuple)
            or len(size) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in size)
        ):
            raise CanvasFileError("'size' must be two integers")
        background = raw.get("background", "#ffffff")
        if background is not None and not isinstance(background, str):
            raise CanvasFileError("'background' must be a hex color or null")

        groups: dict[str, CanvasGroup] = {}
        raw_groups = raw.get("groups", [])
        if not isinstance(raw_groups, list):
            raise CanvasFileError("'groups' must be a list")
        try:
            for item in raw_groups:
                group = CanvasGroup.from_dict(item)
                if group.id in groups:
                    raise CanvasFileError(f"duplicate group id {group.id!r}")
                groups[group.id] = group

            layers: list[CanvasLayer] = []
            raw_layers = raw.get("layers", [])
            if not isinstance(raw_layers, list):
                raise CanvasFileError("'layers' must be a list")
            seen: set[str] = set()
            for item in raw_layers:
                layer = CanvasLayer.from_dict(item)
                if layer.id in seen:
                    raise CanvasFileError(f"duplicate layer id {layer.id!r}")
                if layer.group_id is not None and layer.group_id not in groups:
                    raise CanvasFileError(f"layer {layer.id!r} references an unknown group")
                seen.add(layer.id)
                layers.append(layer)
        except (TypeError, ValueError) as error:
            raise CanvasFileError(str(error)) from error

        try:
            return cls(
                size=(size[0], size[1]),
                background=background,
                layers=layers,
                groups=groups,
                assets_dir=assets_dir,
            )
        except (TypeError, ValueError) as error:
            raise CanvasFileError(str(error)) from error


def assets_dir_for(path: str | Path) -> Path:
    """Return the sibling assets folder for a canvas file (§10.2)."""
    return Path(f"{Path(path)}.assets")


def save_canvas(
    path: str | Path,
    doc: CanvasDoc,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Save ``doc`` to ``path``, copying assets before the JSON commit.

    Existing content-addressed files are never rewritten. The JSON write is
    the commit point; unreferenced assets are collected afterwards.

    Raises:
        CanvasFileError: a referenced asset cannot be found.
    """
    target = Path(path)
    store = AssetStore(assets_dir_for(target))
    source = AssetStore(doc.assets_dir) if doc.assets_dir is not None else None
    was_temporary = doc._temporary_assets
    previous_assets = doc.assets_dir
    references = sorted(doc.referenced_assets())
    total = len(references)
    for done, reference in enumerate(references, start=1):
        if not store.exists(reference):
            if source is None or not source.exists(reference):
                raise CanvasFileError(f"missing asset: {reference}")
            store.copy_from(source, reference)
        if progress is not None:
            progress(done, total)

    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, doc.to_dict())
    doc.assets_dir = store.assets_dir
    doc._temporary_assets = False
    doc.dirty = False
    if was_temporary and previous_assets is not None and previous_assets != store.assets_dir:
        shutil.rmtree(previous_assets, ignore_errors=True)
    store.gc(doc.referenced_assets())


def load_canvas(path: str | Path) -> CanvasDoc:
    """Load a canvas file and verify that every referenced asset exists.

    Raises:
        CanvasFileError: unreadable/malformed file or missing assets. The
            message lists every missing reference; no partial canvas opens.
    """
    source = Path(path)
    try:
        raw = read_json(source)
    except (OSError, ValueError) as error:
        raise CanvasFileError(f"cannot read {source}: {error}") from error
    assets_dir = assets_dir_for(source)
    doc = CanvasDoc.from_dict(raw, assets_dir=assets_dir)
    store = AssetStore(assets_dir)
    missing = sorted(
        reference for reference in doc.referenced_assets() if not store.exists(reference)
    )
    if missing:
        raise CanvasFileError(f"{source}: missing assets: {', '.join(missing)}")
    doc.dirty = False
    return doc

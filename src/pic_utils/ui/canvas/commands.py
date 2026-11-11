"""Canvas undo commands (architecture §12, AD-14).

Commands mutate :class:`~pic_utils.canvas.model.CanvasDoc` through its raw
methods; the canvas page pushes them onto a ``QUndoStack`` and re-syncs the
scene from the result. Transform and property commands merge consecutive
edits of the same layer/property into one undo step.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QUndoCommand

from pic_utils.canvas.model import CanvasDoc, CanvasGroup, CanvasLayer
from pic_utils.core.image import BlendMode

__all__ = [
    "AddLayerCommand",
    "ApplyMaskCommand",
    "CropCanvasCommand",
    "GroupCommand",
    "MoveLayerCommand",
    "RemoveLayerCommand",
    "ResizeCanvasCommand",
    "SetGroupPropertyCommand",
    "SetLayerGroupCommand",
    "SetLayerPropertyCommand",
    "TransformLayerCommand",
    "UngroupCommand",
]


def _coerce_property(prop: str, value: Any) -> Any:
    if prop == "blend":
        return BlendMode(value)
    if prop in ("x", "y", "scale", "rotation"):
        number = float(value)
        if prop == "scale" and number <= 0:
            raise ValueError(f"scale must be positive, got {number}")
        return number
    if prop == "opacity":
        return max(0.0, min(1.0, float(value)))
    if prop in ("flip_h", "flip_v", "visible", "locked"):
        return bool(value)
    if prop == "name":
        return str(value)
    raise ValueError(f"unsupported layer property {prop!r}")


def _reorder(doc: CanvasDoc, order: list[str]) -> None:
    by_id = {layer.id: layer for layer in doc.layers}
    doc.layers[:] = [by_id[layer_id] for layer_id in order if layer_id in by_id]


class AddLayerCommand(QUndoCommand):
    """Insert a layer at the top (or at ``index``)."""

    def __init__(
        self,
        doc: CanvasDoc,
        layer: CanvasLayer,
        *,
        natural_size: tuple[int, int] | None = None,
        index: int | None = None,
    ) -> None:
        super().__init__()
        self._doc = doc
        self._layer = layer
        self._natural_size = natural_size
        self._index = index
        self._old_size = doc.size

    def redo(self) -> None:
        self._doc.add_layer(self._layer, natural_size=self._natural_size, index=self._index)

    def undo(self) -> None:
        self._doc.remove_layer(self._layer.id)
        self._doc.size = self._old_size

    def text(self) -> str:
        return f"add layer {self._layer.name}"


class RemoveLayerCommand(QUndoCommand):
    """Remove one layer; undo reinserts it at its old index."""

    def __init__(self, doc: CanvasDoc, layer_id: str) -> None:
        super().__init__()
        self._doc = doc
        self._layer_id = layer_id
        self._index: int | None = None
        self._layer: CanvasLayer | None = None

    def redo(self) -> None:
        self._index, self._layer = self._doc.remove_layer(self._layer_id)

    def undo(self) -> None:
        if self._layer is not None and self._index is not None:
            self._doc.add_layer(self._layer, index=self._index)

    def referenced_assets(self) -> set[str]:
        """Assets this command can restore to the document."""
        if self._layer is None:
            return set()
        references = {self._layer.image_ref}
        if self._layer.mask_ref is not None:
            references.add(self._layer.mask_ref)
        return references

    def text(self) -> str:
        return f"remove layer {self._layer_id}"


class MoveLayerCommand(QUndoCommand):
    """Move a layer to a new z-order index (drag-release pushes once)."""

    def __init__(self, doc: CanvasDoc, from_index: int, to_index: int) -> None:
        super().__init__()
        self._doc = doc
        self._from = from_index
        self._to = to_index

    def redo(self) -> None:
        self._doc.move_layer(self._from, self._to)

    def undo(self) -> None:
        self._doc.move_layer(self._to, self._from)

    def text(self) -> str:
        return f"move layer {self._from} → {self._to}"


class TransformLayerCommand(QUndoCommand):
    """Set x/y/scale/rotation; consecutive drags of one layer merge."""

    def __init__(
        self,
        doc: CanvasDoc,
        layer_id: str,
        x: float,
        y: float,
        scale: float,
        rotation: float,
    ) -> None:
        super().__init__()
        self._doc = doc
        self._layer_id = layer_id
        layer = doc.layer(layer_id)
        self._old = (layer.x, layer.y, layer.scale, layer.rotation)
        self._new = (float(x), float(y), float(scale), float(rotation))

    def id(self) -> int:
        return hash((TransformLayerCommand, self._layer_id)) & 0x7FFFFFFF

    def _changed_fields(self) -> tuple[bool, bool, bool, bool]:
        return (
            self._old[0] != self._new[0],
            self._old[1] != self._new[1],
            self._old[2] != self._new[2],
            self._old[3] != self._new[3],
        )

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, TransformLayerCommand):
            return False
        if other._layer_id != self._layer_id:
            return False
        if self._changed_fields() != other._changed_fields():
            return False
        self._new = other._new
        return True

    def redo(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    def _apply(self, values: tuple[float, float, float, float]) -> None:
        layer = self._doc.layer(self._layer_id)
        layer.x, layer.y, layer.scale, layer.rotation = values
        self._doc.dirty = True

    def text(self) -> str:
        return f"transform layer {self._layer_id}"


class SetLayerPropertyCommand(QUndoCommand):
    """Set one layer property; consecutive edits of the same cell merge."""

    def __init__(self, doc: CanvasDoc, layer_id: str, prop: str, value: Any) -> None:
        super().__init__()
        self._doc = doc
        self._layer_id = layer_id
        self._prop = prop
        self._new = _coerce_property(prop, value)
        self._old = _coerce_property(prop, getattr(doc.layer(layer_id), prop))

    def id(self) -> int:
        return hash((SetLayerPropertyCommand, self._layer_id, self._prop)) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, SetLayerPropertyCommand):
            return False
        if (other._layer_id, other._prop) != (self._layer_id, self._prop):
            return False
        self._new = other._new
        return True

    def redo(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    def _apply(self, value: Any) -> None:
        setattr(self._doc.layer(self._layer_id), self._prop, value)
        self._doc.dirty = True

    def text(self) -> str:
        return f"set {self._prop} on {self._layer_id}"


class SetGroupPropertyCommand(QUndoCommand):
    """Set one group property; consecutive edits of the same cell merge."""

    def __init__(self, doc: CanvasDoc, group_id: str, prop: str, value: Any) -> None:
        super().__init__()
        self._doc = doc
        self._group_id = group_id
        self._prop = prop
        self._new = self._coerce(prop, value)
        self._old = self._coerce(prop, getattr(doc.groups[group_id], prop))

    @staticmethod
    def _coerce(prop: str, value: Any) -> Any:
        if prop == "opacity":
            return max(0.0, min(1.0, float(value)))
        if prop == "visible":
            return bool(value)
        if prop == "name":
            return str(value)
        raise ValueError(f"unsupported group property {prop!r}")

    def id(self) -> int:
        return hash((SetGroupPropertyCommand, self._group_id, self._prop)) & 0x7FFFFFFF

    def mergeWith(self, other: QUndoCommand) -> bool:
        if not isinstance(other, SetGroupPropertyCommand):
            return False
        if (other._group_id, other._prop) != (self._group_id, self._prop):
            return False
        self._new = other._new
        return True

    def redo(self) -> None:
        self._apply(self._new)

    def undo(self) -> None:
        self._apply(self._old)

    def _apply(self, value: Any) -> None:
        setattr(self._doc.groups[self._group_id], self._prop, value)
        self._doc.dirty = True

    def text(self) -> str:
        return f"set group {self._prop}"


class SetLayerGroupCommand(QUndoCommand):
    """Add or remove one layer's group membership, keeping members contiguous."""

    def __init__(self, doc: CanvasDoc, layer_id: str, group_id: str | None) -> None:
        super().__init__()
        self._doc = doc
        self._layer_id = layer_id
        self._new = group_id
        self._old = doc.layer(layer_id).group_id
        self._old_groups = dict(doc.groups)
        self._old_order = [layer.id for layer in doc.layers]
        self._old_assignments = {layer.id: layer.group_id for layer in doc.layers}

    def redo(self) -> None:
        self._doc.set_layer_group(self._layer_id, self._new)

    def undo(self) -> None:
        self._doc.groups.clear()
        self._doc.groups.update(self._old_groups)
        for layer in self._doc.layers:
            layer.group_id = self._old_assignments.get(layer.id)
        _reorder(self._doc, self._old_order)
        self._doc.dirty = True

    def text(self) -> str:
        return f"set group membership for {self._layer_id}"


class ApplyMaskCommand(QUndoCommand):
    """Apply or remove a layer mask reference."""

    def __init__(self, doc: CanvasDoc, layer_id: str, mask_ref: str | None) -> None:
        super().__init__()
        self._doc = doc
        self._layer_id = layer_id
        self._new = mask_ref
        self._old = doc.layer(layer_id).mask_ref

    def redo(self) -> None:
        self._doc.layer(self._layer_id).mask_ref = self._new
        self._doc.dirty = True

    def undo(self) -> None:
        self._doc.layer(self._layer_id).mask_ref = self._old
        self._doc.dirty = True

    def referenced_assets(self) -> set[str]:
        """The mask asset this command can restore to the document."""
        return {self._old} if self._old is not None else set()

    def text(self) -> str:
        return f"{'apply' if self._new else 'remove'} mask on {self._layer_id}"


class CropCanvasCommand(QUndoCommand):
    """Crop the canvas and offset every layer by ``-x, -y`` (§10.6)."""

    def __init__(self, doc: CanvasDoc, x: float, y: float, width: float, height: float) -> None:
        super().__init__()
        self._doc = doc
        self._x = float(x)
        self._y = float(y)
        self._width = float(width)
        self._height = float(height)
        self._old_size = doc.size
        self._old_positions = {layer.id: (layer.x, layer.y) for layer in doc.layers}

    def redo(self) -> None:
        self._doc.crop(self._x, self._y, self._width, self._height)

    def undo(self) -> None:
        self._doc.size = self._old_size
        for layer in self._doc.layers:
            position = self._old_positions.get(layer.id)
            if position is not None:
                layer.x, layer.y = position
        self._doc.dirty = True

    def text(self) -> str:
        return "crop canvas"


class ResizeCanvasCommand(QUndoCommand):
    """Resize the canvas; layers keep their coordinates (§10.6)."""

    def __init__(self, doc: CanvasDoc, width: int, height: int) -> None:
        super().__init__()
        self._doc = doc
        self._width = int(width)
        self._height = int(height)
        self._old_size = doc.size

    def redo(self) -> None:
        self._doc.resize(self._width, self._height)

    def undo(self) -> None:
        self._doc.size = self._old_size
        self._doc.dirty = True

    def text(self) -> str:
        return "resize canvas"


class GroupCommand(QUndoCommand):
    """Create a group from a selection; undo restores the old order."""

    def __init__(self, doc: CanvasDoc, layer_ids: list[str], name: str | None = None) -> None:
        super().__init__()
        self._doc = doc
        self._layer_ids = list(layer_ids)
        self._name = name
        self._group_id: str | None = None
        self._old_order: list[str] = []
        self._old_groups: dict[str, CanvasGroup] = {}
        self._old_assignments: dict[str, str | None] = {}

    def redo(self) -> None:
        self._old_order = [layer.id for layer in self._doc.layers]
        self._old_groups = dict(self._doc.groups)
        self._old_assignments = {layer.id: layer.group_id for layer in self._doc.layers}
        group = self._doc.group_layers(self._layer_ids, self._name)
        self._group_id = group.id

    def undo(self) -> None:
        if self._group_id is not None:
            self._doc.groups.clear()
            self._doc.groups.update(self._old_groups)
        for layer in self._doc.layers:
            layer.group_id = self._old_assignments.get(layer.id)
        _reorder(self._doc, self._old_order)
        self._doc.dirty = True

    def text(self) -> str:
        return "group layers"


class UngroupCommand(QUndoCommand):
    """Remove a group; undo restores the group and its members."""

    def __init__(self, doc: CanvasDoc, group_id: str) -> None:
        super().__init__()
        self._doc = doc
        self._group_id = group_id
        self._group: CanvasGroup | None = None
        self._old_order: list[str] = []
        self._old_assignments: dict[str, str | None] = {}

    def redo(self) -> None:
        self._old_order = [layer.id for layer in self._doc.layers]
        self._old_assignments = {layer.id: layer.group_id for layer in self._doc.layers}
        self._group = self._doc.ungroup(self._group_id)

    def undo(self) -> None:
        if self._group is not None:
            self._doc.groups[self._group.id] = self._group
        for layer in self._doc.layers:
            layer.group_id = self._old_assignments.get(layer.id)
        _reorder(self._doc, self._old_order)
        self._doc.dirty = True

    def text(self) -> str:
        return "ungroup layers"

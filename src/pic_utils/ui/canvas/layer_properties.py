"""Layer properties: transform, opacity, blend, flips, mask, and flags (§13.1).

Every edit emits ``property_changed``; the page pushes a
``SetLayerPropertyCommand`` (merged per layer/property) so slider and
spinbox drags become one undo step.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from pic_utils.canvas.model import CanvasGroup, CanvasLayer
from pic_utils.core.image import BlendMode

__all__ = ["LayerPropertiesPanel"]

_BLEND_ORDER = (
    BlendMode.NORMAL,
    BlendMode.MULTIPLY,
    BlendMode.SCREEN,
    BlendMode.OVERLAY,
    BlendMode.ADD,
    BlendMode.DIFFERENCE,
)


class LayerPropertiesPanel(QWidget):
    """Form for the selected layer; disabled when nothing is selected."""

    property_changed = Signal(str, str, object)
    group_property_changed = Signal(str, str, object)
    group_membership_changed = Signal(str, object)
    mask_requested = Signal(str)
    mask_removed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_id: str | None = None
        self._group_id: str | None = None
        self._updating = False
        self._labels: dict[str, QLabel] = {}

        self._name = QLineEdit(self)
        self._name.textEdited.connect(lambda text: self._emit("name", text))

        self._x = self._number(-1_000_000, 1_000_000, 1.0, 2)
        self._y = self._number(-1_000_000, 1_000_000, 1.0, 2)
        self._scale = self._number(0.01, 100.0, 0.01, 3)
        self._rotation = self._number(-360.0, 360.0, 1.0, 2)
        for key, widget in (
            ("x", self._x),
            ("y", self._y),
            ("scale", self._scale),
            ("rotation", self._rotation),
        ):
            widget.valueChanged.connect(lambda value, prop=key: self._emit(prop, float(value)))

        self._flip_h = QCheckBox(self)
        self._flip_h.toggled.connect(lambda value: self._emit("flip_h", value))
        self._flip_v = QCheckBox(self)
        self._flip_v.toggled.connect(lambda value: self._emit("flip_v", value))
        self._visible = QCheckBox(self)
        self._visible.toggled.connect(lambda value: self._emit("visible", value))
        self._locked = QCheckBox(self)
        self._locked.toggled.connect(lambda value: self._emit("locked", value))

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._opacity_slider.setRange(0, 100)
        self._opacity_spin = QSpinBox(self)
        self._opacity_spin.setRange(0, 100)
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider)
        self._opacity_spin.valueChanged.connect(self._on_opacity_spin)
        opacity_row = QWidget(self)
        opacity_layout = QHBoxLayout(opacity_row)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_layout.addWidget(self._opacity_slider, 1)
        opacity_layout.addWidget(self._opacity_spin)

        self._blend = QComboBox(self)
        for mode in _BLEND_ORDER:
            self._blend.addItem(mode.value.title(), mode.value)
        self._blend.currentIndexChanged.connect(self._on_blend)

        self._group = QComboBox(self)
        self._group.currentIndexChanged.connect(self._on_group_membership)
        self._group_opacity = QSpinBox(self)
        self._group_opacity.setRange(0, 100)
        self._group_opacity.valueChanged.connect(self._on_group_opacity)
        self._group_visible = QCheckBox(self)
        self._group_visible.toggled.connect(lambda value: self._emit_group("visible", value))
        self._mask = QLabel(self)
        self._mask_button = QPushButton(self)
        self._mask_button.clicked.connect(self._request_mask)
        self._mask_remove = QPushButton(self)
        self._mask_remove.clicked.connect(self._remove_mask)
        mask_row = QWidget(self)
        mask_layout = QHBoxLayout(mask_row)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_layout.addWidget(self._mask, 1)
        mask_layout.addWidget(self._mask_button)
        mask_layout.addWidget(self._mask_remove)

        self._form = QFormLayout(self)
        self._row("name", self._name)
        self._row("x", self._x)
        self._row("y", self._y)
        self._row("scale", self._scale)
        self._row("rotation", self._rotation)
        self._row("flip_h", self._flip_h)
        self._row("flip_v", self._flip_v)
        self._row("opacity", opacity_row)
        self._row("blend", self._blend)
        self._row("group", self._group)
        self._row("group_opacity", self._group_opacity)
        self._row("group_visible", self._group_visible)
        self._row("mask", mask_row)
        self._row("visible", self._visible)
        self._row("locked", self._locked)

        self.set_layer(None, {})
        self.retranslate()

    # --- construction helpers ------------------------------------------------

    def _number(self, minimum: float, maximum: float, step: float, decimals: int) -> QDoubleSpinBox:
        spin = QDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        return spin

    def _row(self, key: str, widget: QWidget) -> None:
        label = QLabel(self)
        self._labels[key] = label
        self._form.addRow(label, widget)

    # --- public API ----------------------------------------------------------

    def selected_layer_id(self) -> str | None:
        return self._layer_id

    def set_layer(self, layer: CanvasLayer | None, groups: dict[str, CanvasGroup]) -> None:
        """Show ``layer``; ``groups`` supplies the group display name."""
        self._layer_id = layer.id if layer is not None else None
        self._group_id = layer.group_id if layer is not None else None
        self.setEnabled(layer is not None)
        if layer is None:
            self._updating = True
            self._name.clear()
            self._group.clear()
            self._group_opacity.setValue(0)
            self._group_visible.setChecked(False)
            self._group_opacity.setEnabled(False)
            self._group_visible.setEnabled(False)
            self._mask.setText("")
            self._updating = False
            return
        self._updating = True
        try:
            self._name.setText(layer.name)
            self._x.setValue(layer.x)
            self._y.setValue(layer.y)
            self._scale.setValue(layer.scale)
            self._rotation.setValue(layer.rotation)
            self._flip_h.setChecked(layer.flip_h)
            self._flip_v.setChecked(layer.flip_v)
            self._visible.setChecked(layer.visible)
            self._locked.setChecked(layer.locked)
            percent = round(layer.opacity * 100)
            self._opacity_slider.setValue(percent)
            self._opacity_spin.setValue(percent)
            index = self._blend.findData(str(layer.blend))
            self._blend.setCurrentIndex(max(index, 0))
            group = groups.get(layer.group_id) if layer.group_id is not None else None
            self._group.clear()
            self._group.addItem(self.tr("None"), None)
            for group_id, item in groups.items():
                self._group.addItem(item.name, group_id)
            index = self._group.findData(layer.group_id) if layer.group_id is not None else 0
            self._group.setCurrentIndex(max(index, 0))
            self._group_opacity.setEnabled(group is not None)
            self._group_visible.setEnabled(group is not None)
            self._group_opacity.setValue(round(group.opacity * 100) if group else 0)
            self._group_visible.setChecked(group.visible if group else False)
            self._mask.setText(layer.mask_ref if layer.mask_ref else self.tr("None"))
            self._mask_remove.setEnabled(layer.mask_ref is not None)
        finally:
            self._updating = False

    # --- signals -------------------------------------------------------------

    def _emit(self, prop: str, value: object) -> None:
        if self._updating or self._layer_id is None:
            return
        self.property_changed.emit(self._layer_id, prop, value)

    def _on_opacity_slider(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._opacity_spin.setValue(value)
        self._updating = False
        self._emit("opacity", value / 100.0)

    def _on_opacity_spin(self, value: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._opacity_slider.setValue(value)
        self._updating = False
        self._emit("opacity", value / 100.0)

    def _on_group_membership(self, _index: int) -> None:
        if self._updating or self._layer_id is None:
            return
        self.group_membership_changed.emit(self._layer_id, self._group.currentData())

    def _emit_group(self, prop: str, value: object) -> None:
        if self._updating or self._group_id is None:
            return
        self.group_property_changed.emit(self._group_id, prop, value)

    def _on_group_opacity(self, value: int) -> None:
        self._emit_group("opacity", value / 100.0)

    def _on_blend(self, index: int) -> None:
        value = self._blend.itemData(index)
        if value is not None:
            self._emit("blend", value)

    def _request_mask(self) -> None:
        if self._layer_id is not None:
            self.mask_requested.emit(self._layer_id)

    def _remove_mask(self) -> None:
        if self._layer_id is not None:
            self.mask_removed.emit(self._layer_id)

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        labels = {
            "name": self.tr("Name"),
            "x": self.tr("X"),
            "y": self.tr("Y"),
            "scale": self.tr("Scale"),
            "rotation": self.tr("Rotation"),
            "flip_h": self.tr("Flip horizontally"),
            "flip_v": self.tr("Flip vertically"),
            "opacity": self.tr("Opacity"),
            "blend": self.tr("Blend mode"),
            "group": self.tr("Group"),
            "group_opacity": self.tr("Group opacity"),
            "group_visible": self.tr("Group visible"),
            "mask": self.tr("Mask"),
            "visible": self.tr("Visible"),
            "locked": self.tr("Locked"),
        }
        for key, text in labels.items():
            self._labels[key].setText(text)
        blend_labels = (
            self.tr("Normal"),
            self.tr("Multiply"),
            self.tr("Screen"),
            self.tr("Overlay"),
            self.tr("Add"),
            self.tr("Difference"),
        )
        for index, text in enumerate(blend_labels):
            self._blend.setItemText(index, text)
        if self._group.count():
            self._group.setItemText(0, self.tr("None"))
        self._mask_button.setText(self.tr("Generate mask…"))
        self._mask_remove.setText(self.tr("Remove mask"))

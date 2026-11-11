"""Parameter widgets: one per ``ParamKind`` (architecture §13.4).

Every widget emits :attr:`ParamWidget.edited` with the new value; the panel
pushes a ``SetParamCommand`` (merged per drag, §12). The seed widget has an
"auto" checkbox (``None`` = auto) and a randomize button.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QSpinBox,
    QWidget,
)

from pic_utils.blocks.api import Choice, FloatRange, IntRange, ParamDef, ParamKind
from pic_utils.pipeline.model import derive_seed

__all__ = ["ParamWidget", "make_param_widget"]


class ParamWidget(QWidget):
    """Base contract: ``edited`` fires with the new value; ``set_value`` syncs."""

    edited = Signal(object)

    def __init__(self, spec: object, definition: ParamDef, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spec = spec
        self._definition = definition

    def set_value(self, value: Any) -> None:
        """Set the widget's value without emitting :attr:`edited`."""


class _IntWidget(ParamWidget):
    """``QSlider`` + ``QSpinBox`` synced, range/step from ``IntRange`` (§13.4)."""

    def __init__(self, spec: IntRange, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._spin = QSpinBox()
        self._spin.setRange(spec.min, spec.max)
        self._spin.setStepType(QSpinBox.StepType.AdaptiveDecimalStepType)
        if spec.step > 1:
            self._spin.setSingleStep(spec.step)
        self._slider.setRange(spec.min, spec.max)

        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._on_spin)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

    def _on_spin(self, value: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
        self.edited.emit(value)

    def set_value(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(int(value))
        self._spin.blockSignals(False)
        self._slider.blockSignals(True)
        self._slider.setValue(int(value))
        self._slider.blockSignals(False)


class _FloatWidget(ParamWidget):
    """``QDoubleSpinBox`` + slider mapped through the range's step (§13.4)."""

    def __init__(self, spec: FloatRange, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._range = spec
        resolution = max(round((spec.max - spec.min) / spec.step), 1)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, resolution)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(spec.min, spec.max)
        self._spin.setDecimals(4)
        self._spin.setSingleStep(spec.step)

        self._slider.valueChanged.connect(self._spin.setValue)
        self._spin.valueChanged.connect(self._on_spin)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._spin)

    def _position(self, value: float) -> int:
        return round((value - self._range.min) / self._range.step)

    def _on_spin(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._position(value))
        self._slider.blockSignals(False)
        self.edited.emit(value)

    def set_value(self, value: Any) -> None:
        self._spin.blockSignals(True)
        self._spin.setValue(float(value))
        self._spin.blockSignals(False)
        self._slider.blockSignals(True)
        self._slider.setValue(self._position(float(value)))
        self._slider.blockSignals(False)


class _ChoiceWidget(ParamWidget):
    """``QComboBox`` showing labels when provided (§13.4)."""

    def __init__(self, spec: Choice, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._values = list(spec.options)
        self._combo = QComboBox()
        self._combo.addItems([str(option) for option in self._values])
        if spec.labels:
            for index, label in enumerate(spec.labels):
                self._combo.setItemText(index, label)
        self._combo.currentIndexChanged.connect(self._on_index)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo)

    def _on_index(self, index: int) -> None:
        self.edited.emit(self._values[index])

    def set_value(self, value: Any) -> None:
        index = self._values.index(value) if value in self._values else -1
        if index >= 0:
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(index)
            self._combo.blockSignals(False)


class _BoolWidget(ParamWidget):
    """``QCheckBox`` (§13.4)."""

    def __init__(self, spec: object, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._check = QCheckBox()
        self._check.toggled.connect(self.edited.emit)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._check)

    def set_value(self, value: Any) -> None:
        self._check.blockSignals(True)
        self._check.setChecked(bool(value))
        self._check.blockSignals(False)


class _ColorWidget(ParamWidget):
    """Color swatch button opening a ``QColorDialog`` (§13.4)."""

    def __init__(self, spec: object, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._swatch = QFrame()
        self._swatch.setFixedSize(20, 20)
        self._swatch.setFrameShape(QFrame.Shape.Box)
        self._button = QPushButton()
        self._button.clicked.connect(self._choose)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._swatch)
        layout.addWidget(self._button)

    def _choose(self) -> None:
        current = QColor(self._button.text()) if self._button.text() else QColor("#ffffff")
        color = QColorDialog.getColor(current, self._button)
        if color.isValid():
            self._apply(color.name())
            self.edited.emit(color.name())

    def _apply(self, name: str) -> None:
        self._button.setText(name)
        self._swatch.setStyleSheet(f"background-color: {name};")

    def set_value(self, value: Any) -> None:
        self._apply(str(value))


class _SeedWidget(ParamWidget):
    """``QSpinBox`` + "auto" checkbox + randomize button (§13.4).

    ``None`` means auto: the executor derives the seed per experiment.
    """

    def __init__(self, spec: object, definition: ParamDef) -> None:
        super().__init__(spec, definition)
        self._param_name = definition.name
        self._auto = QCheckBox()
        self._auto.setChecked(True)
        self._randomize = QPushButton()
        self._randomize.setEnabled(False)
        self._spin = QSpinBox()
        self._spin.setRange(0, 0x7FFFFFFF)
        self._spin.setEnabled(False)

        self._auto.toggled.connect(self._on_auto)
        self._randomize.clicked.connect(self._on_randomize)
        self._spin.valueChanged.connect(self.edited.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._auto)
        layout.addWidget(self._randomize)
        layout.addWidget(self._spin, 1)

    def _on_auto(self, enabled: bool) -> None:
        self._spin.setEnabled(not enabled)
        self._randomize.setEnabled(not enabled)
        self.edited.emit(None if enabled else self._spin.value())

    def _on_randomize(self) -> None:
        value = derive_seed(self._spin.value() or 0, self._param_name, "randomize")
        self._spin.setValue(value)

    def set_value(self, value: Any) -> None:
        auto = value is None
        self._auto.blockSignals(True)
        self._auto.setChecked(auto)
        self._auto.blockSignals(False)
        self._spin.setEnabled(not auto)
        self._randomize.setEnabled(not auto)
        if value is not None:
            self._spin.blockSignals(True)
            self._spin.setValue(int(value))
            self._spin.blockSignals(False)


_KIND_WIDGETS: dict[ParamKind, type[ParamWidget]] = {
    ParamKind.INT: _IntWidget,
    ParamKind.FLOAT: _FloatWidget,
    ParamKind.CHOICE: _ChoiceWidget,
    ParamKind.BOOL: _BoolWidget,
    ParamKind.COLOR: _ColorWidget,
    ParamKind.SEED: _SeedWidget,
}


def make_param_widget(definition: ParamDef) -> ParamWidget:
    """Build the widget for one value-typed ``ParamDef`` (§13.4 table).

    ``IMAGE``/``MASK`` kinds are injected inputs (chain input, second image,
    mask bindings) and never get a value widget.
    """
    widget: ParamWidget
    if definition.kind is ParamKind.INT:
        assert isinstance(definition.spec, IntRange)
        widget = _IntWidget(definition.spec, definition)
    elif definition.kind is ParamKind.FLOAT:
        assert isinstance(definition.spec, FloatRange)
        widget = _FloatWidget(definition.spec, definition)
    elif definition.kind is ParamKind.CHOICE:
        assert isinstance(definition.spec, Choice)
        widget = _ChoiceWidget(definition.spec, definition)
    elif definition.kind in _KIND_WIDGETS:
        widget = _KIND_WIDGETS[definition.kind](definition.spec, definition)
    else:
        raise ValueError(f"parameter kind {definition.kind} has no value widget")
    return widget

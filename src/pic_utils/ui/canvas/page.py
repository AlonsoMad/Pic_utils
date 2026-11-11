"""Canvas mode composition root (architecture §13.1).

Left asset tray; center canvas view; right layer list + layer properties.
The initial version placeholders each slot; P6-07 injects the real panels
via the ``set_*`` methods, which is the stable layout API.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

__all__ = ["CanvasPage"]


def _placeholder(object_name: str) -> QLabel:
    label = QLabel()
    label.setObjectName(object_name)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class CanvasPage(QWidget):
    """Left-to-right Canvas layout with an injected-panel API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._asset_tray = _placeholder("canvas_asset_tray")
        self._canvas_view = _placeholder("canvas_view")
        self._layer_list = _placeholder("canvas_layer_list")
        self._layer_properties = _placeholder("canvas_layer_properties")

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self._layer_list)
        right.addWidget(self._layer_properties)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(self._asset_tray)
        columns.addWidget(self._canvas_view)
        columns.addWidget(right)
        columns.setStretchFactor(0, 1)
        columns.setStretchFactor(1, 4)
        columns.setStretchFactor(2, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(columns)

        self.retranslate()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._asset_tray.setText(self.tr("Assets"))
        self._canvas_view.setText(self.tr("Canvas"))
        self._layer_list.setText(self.tr("Layers"))
        self._layer_properties.setText(self.tr("Layer properties"))

    # --- panel injection (stable layout API) -------------------------------

    def set_asset_tray(self, widget: QWidget) -> None:
        self._replace(self._asset_tray, widget)

    def set_canvas_view(self, widget: QWidget) -> None:
        self._replace(self._canvas_view, widget)

    def set_layer_list(self, widget: QWidget) -> None:
        self._replace(self._layer_list, widget)

    def set_layer_properties(self, widget: QWidget) -> None:
        self._replace(self._layer_properties, widget)

    def _replace(self, placeholder: QWidget, widget: QWidget) -> None:
        parent = placeholder.parentWidget()
        assert isinstance(parent, QSplitter)
        assert parent is not None, "placeholder must be in the layout"
        index = parent.indexOf(placeholder)
        stretch = parent.stretchFactor(placeholder) if hasattr(parent, "stretchFactor") else 0
        parent.replaceWidget(index, widget)
        placeholder.deleteLater()
        if stretch:
            parent.setStretchFactor(index, stretch)

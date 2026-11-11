"""Lab mode composition root (architecture §13.1).

The page owns the splitter layout: left block library; center pipeline
editor + preview pane; right parameter panel; bottom results grid. The
initial version placeholders each slot; P5-10 injects the real panels via
the ``set_*`` methods, which is the stable layout API.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

__all__ = ["LabPage"]

_CENTER_STRETCH = 3
_SIDE_STRETCH = 1


def _placeholder(object_name: str) -> QLabel:
    label = QLabel()
    label.setObjectName(object_name)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return label


class LabPage(QWidget):
    """Left-to-right Lab layout with an injected-panel API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._block_library = _placeholder("lab_block_library")
        self._pipeline_editor = _placeholder("lab_pipeline_editor")
        self._preview_pane = _placeholder("lab_preview_pane")
        self._parameter_panel = _placeholder("lab_parameter_panel")
        self._results_grid = _placeholder("lab_results_grid")

        center = QSplitter(Qt.Orientation.Vertical)
        center.addWidget(self._pipeline_editor)
        center.addWidget(self._preview_pane)

        columns = QSplitter(Qt.Orientation.Horizontal)
        columns.addWidget(self._block_library)
        columns.addWidget(center)
        columns.addWidget(self._parameter_panel)
        columns.setStretchFactor(0, _SIDE_STRETCH)
        columns.setStretchFactor(1, _CENTER_STRETCH)
        columns.setStretchFactor(2, _SIDE_STRETCH)

        rows = QSplitter(Qt.Orientation.Vertical)
        rows.addWidget(columns)
        rows.addWidget(self._results_grid)
        rows.setStretchFactor(0, 3)
        rows.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(rows)

        self.retranslate()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self._block_library.setText(self.tr("Block library"))
        self._pipeline_editor.setText(self.tr("Pipeline editor"))
        self._preview_pane.setText(self.tr("Preview"))
        self._parameter_panel.setText(self.tr("Parameters"))
        self._results_grid.setText(self.tr("Results"))

    # --- panel injection (stable layout API) -------------------------------

    def set_block_library(self, widget: QWidget) -> None:
        self._replace(self._block_library, widget)

    def set_pipeline_editor(self, widget: QWidget) -> None:
        self._replace(self._pipeline_editor, widget)

    def set_preview_pane(self, widget: QWidget) -> None:
        self._replace(self._preview_pane, widget)

    def set_parameter_panel(self, widget: QWidget) -> None:
        self._replace(self._parameter_panel, widget)

    def set_results_grid(self, widget: QWidget) -> None:
        self._replace(self._results_grid, widget)

    def _replace(self, placeholder: QWidget, widget: QWidget) -> None:
        parent = placeholder.parentWidget()
        assert parent is not None, "placeholder must be in the layout"
        index = parent.indexOf(placeholder)
        stretch = parent.stretchFactor(placeholder) if hasattr(parent, "stretchFactor") else 0
        parent.replaceWidget(index, widget)
        placeholder.deleteLater()
        if stretch:
            parent.setStretchFactor(index, stretch)

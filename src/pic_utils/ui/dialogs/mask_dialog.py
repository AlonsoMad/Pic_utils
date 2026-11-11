"""Mask generation dialog for a canvas layer (FR-CAN-05, §10.5).

The user picks a mask block, edits its parameters, previews the result on
the selected layer's image, and accepts it. The resulting :class:`Mask` is
stored as a grayscale PNG asset by the caller; no pipeline intermediates are
persisted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pic_utils.blocks.api import OutputKind, ParamKind
from pic_utils.blocks.snapshot import BlockSnapshot
from pic_utils.core.image import Image, Mask
from pic_utils.ui.widgets.param_widgets import make_param_widget

__all__ = ["MaskDialog"]

_PREVIEW_SIZE = 256


class MaskDialog(QDialog):
    """Pick a mask block, edit parameters, preview, and accept a mask."""

    def __init__(
        self,
        image: Image,
        snapshot: BlockSnapshot,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image = image
        self._snapshot = snapshot
        self._values: dict[str, Any] = {}
        self._mask: Mask | None = None

        self._block_combo = QComboBox(self)
        for name in sorted(snapshot):
            definition = snapshot.get(name)
            if definition.output is OutputKind.MASK and not definition.mask_inputs:
                self._block_combo.addItem(name, name)
        self._block_combo.currentIndexChanged.connect(self._rebuild)

        self._form_host = QWidget(self)
        self._form = QFormLayout(self._form_host)
        self._form.setContentsMargins(0, 0, 0, 0)

        self._preview = QLabel(self)
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        self._preview.setScaledContents(False)

        self._preview_button = QPushButton(self)
        self._preview_button.clicked.connect(self._preview_mask)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._block_combo)
        layout.addWidget(self._form_host)
        layout.addWidget(self._preview_button)
        layout.addWidget(self._preview, 1)
        layout.addWidget(self._buttons)

        self._rebuild()
        self.retranslate()

    # --- public API ----------------------------------------------------------

    def result_mask(self) -> Mask | None:
        return self._mask

    def block_name(self) -> str | None:
        data = self._block_combo.currentData()
        return str(data) if data is not None else None

    # --- form ----------------------------------------------------------------

    def _rebuild(self) -> None:
        while self._form.rowCount():
            self._form.removeRow(0)
        self._values.clear()
        self._preview.clear()
        name = self.block_name()
        if name is None:
            self._preview_button.setEnabled(False)
            return
        definition = self._snapshot.get(name)
        for param in definition.params:
            if param.kind in (ParamKind.IMAGE, ParamKind.MASK):
                continue
            widget = make_param_widget(param)
            widget.set_value(param.default)
            self._values[param.name] = param.default
            widget.edited.connect(lambda value, key=param.name: self._on_edited(key, value))
            self._form.addRow(param.name, widget)
        self._preview_button.setEnabled(True)

    def _on_edited(self, name: str, value: Any) -> None:
        self._values[name] = value

    def _compute_mask(self) -> Mask:
        name = self.block_name()
        if name is None:
            raise ValueError("no mask block selected")
        definition = self._snapshot.get(name)
        result = definition.fn(self._image, **self._values)
        if not isinstance(result, Mask):
            raise TypeError(f"block {name!r} did not return a Mask")
        return result

    # --- preview and accept --------------------------------------------------

    def _preview_mask(self) -> None:
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            mask = self._compute_mask()
        except Exception as error:
            self._preview.setText(str(error))
            return
        finally:
            self.unsetCursor()
        data = np.ascontiguousarray(mask.data)
        height, width = data.shape
        qimage = QImage(
            data.data,
            width,
            height,
            int(data.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            _PREVIEW_SIZE,
            _PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(pixmap)

    def accept(self) -> None:
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            mask = self._compute_mask()
        except Exception as error:
            QMessageBox.warning(self, self.tr("Mask generation failed"), str(error))
            return
        finally:
            self.unsetCursor()
        self._mask = mask
        super().accept()

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Generate mask"))
        self._preview_button.setText(self.tr("Preview"))
        self._preview.setText(self.tr("No preview"))

"""Export dialog: scope, format, JPEG quality, and target folder (§11).

The dialog only collects options; :func:`pic_utils.canvas.export.export_canvas`
does the rendering and writing, including collision-safe folder naming.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pic_utils.canvas.export import ExportOptions
from pic_utils.settings import Settings

__all__ = ["ExportDialog"]

_FORMATS = (("png", "PNG"), ("jpeg", "JPEG"), ("tiff", "TIFF"))
_SCOPES = (
    ("canvas", "Canvas only"),
    ("canvas_and_accepted", "Canvas + accepted results"),
    ("everything", "Canvas + current run"),
)


class ExportDialog(QDialog):
    """Collects one export request; ``options()`` returns the result."""

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._labels: dict[str, QLabel] = {}

        self._format = QComboBox(self)
        for value, label in _FORMATS:
            self._format.addItem(label, value)
        self._format.currentIndexChanged.connect(self._on_format_changed)

        self._quality = QSpinBox(self)
        self._quality.setRange(1, 100)
        self._quality.setValue(settings.jpeg_quality)

        self._scope = QComboBox(self)
        for value, label in _SCOPES:
            self._scope.addItem(label, value)

        self._target = QLineEdit(str(settings.results_root), self)
        self._target.textChanged.connect(self._update_ok)
        self._browse = QPushButton(self)
        self._browse.clicked.connect(self._choose_target)
        target_row = QWidget(self)
        target_layout = QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self._target, 1)
        target_layout.addWidget(self._browse)

        form = QFormLayout(self)
        self._row(form, "format", self._format)
        self._row(form, "quality", self._quality)
        self._row(form, "scope", self._scope)
        self._row(form, "target", target_row)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._buttons)

        self._on_format_changed()
        self.retranslate()

    # --- public API ----------------------------------------------------------

    def options(self) -> ExportOptions:
        """Return the selected export options."""
        return ExportOptions(
            target_dir=Path(self._target.text().strip()),
            fmt=str(self._format.currentData()),
            jpeg_quality=self._quality.value(),
            scope=str(self._scope.currentData()),
        )

    # --- construction helpers ------------------------------------------------

    def _row(self, form: QFormLayout, key: str, widget: QWidget) -> None:
        label = QLabel(self)
        self._labels[key] = label
        form.addRow(label, widget)

    # --- actions -------------------------------------------------------------

    def _on_format_changed(self) -> None:
        self._quality.setEnabled(self._format.currentData() == "jpeg")

    def _choose_target(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, self.tr("Export folder"), self._target.text()
        )
        if chosen:
            self._target.setText(chosen)

    def _update_ok(self) -> None:
        button = self._buttons.button(QDialogButtonBox.StandardButton.Ok)
        if button is not None:
            button.setEnabled(bool(self._target.text().strip()))

    # --- i18n ----------------------------------------------------------------

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Export canvas"))
        self._labels["format"].setText(self.tr("Format"))
        self._labels["quality"].setText(self.tr("JPEG quality"))
        self._labels["scope"].setText(self.tr("Scope"))
        self._labels["target"].setText(self.tr("Export folder"))
        self._browse.setText(self.tr("Browse…"))
        for index, (_, label) in enumerate(_FORMATS):
            self._format.setItemText(index, self.tr(label))
        for index, (_, label) in enumerate(_SCOPES):
            self._scope.setItemText(index, self.tr(label))

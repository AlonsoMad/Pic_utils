"""Relink dialog: re-point missing project sources (FR-PERS-05, §13.6).

Modal; only for decisions. Lists the missing files with a Browse button per
row; OK returns the old→new mapping (unresolved rows stay missing and the
caller decides whether to skip them).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

__all__ = ["RelinkDialog"]


class RelinkDialog(QDialog):
    """One row per missing file; returns the remapping on accept."""

    def __init__(self, missing: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._missing = list(missing)
        self._picks: dict[Path, Path] = {}
        self._rows: list[tuple[QLabel, QPushButton, QLabel]] = []

        layout = QVBoxLayout(self)
        intro = QLabel(self)
        layout.addWidget(intro)
        self._intro = intro

        for missing_path in self._missing:
            row = QHBoxLayout()
            label = QLabel(str(missing_path), self)
            button = QPushButton(self)
            status = QLabel(self)
            button.clicked.connect(
                lambda _=False, label=label, status=status: self._browse(label, status)
            )
            row.addWidget(label, 1)
            row.addWidget(button)
            row.addWidget(status)
            layout.addLayout(row)
            self._rows.append((label, button, status))

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.retranslate()

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Relink missing files"))
        self._intro.setText(
            self.tr("These files are missing. Choose a new location for each, or cancel.")
        )
        for _label, button, status in self._rows:
            button.setText(self.tr("Browse…"))
            status.setText("")

    def remapping(self) -> dict[Path, Path]:
        """The old→new mapping the user chose; unresolved entries are absent."""
        return dict(self._picks)

    # --- internals ------------------------------------------------------------

    def _browse(self, label: QLabel, status: QLabel) -> None:
        old = Path(label.text())
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Relink"),
            str(old.parent),
            self.tr("Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)"),
        )
        if not path:
            return
        new = Path(path)
        self._picks[old] = new
        status.setText(self.tr("→") + f" {new.name}")

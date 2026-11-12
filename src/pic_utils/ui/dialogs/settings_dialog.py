"""Settings dialog (architecture §13.6, §16).

Modal; only decisions live here. The language applies immediately (live
preview, AC-12); every other value is written on accept. Settings changes
that affect a running service apply to the next run.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from pic_utils.settings import LANGUAGE_OPTIONS, LOG_LEVELS, THEME_OPTIONS, Settings
from pic_utils.ui.i18n import LanguageManager

__all__ = ["SettingsDialog"]


def _browse_row(line: QLineEdit, parent: QDialog, title: str) -> tuple[QWidget, QPushButton]:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line)
    button = QPushButton(QCoreApplication.translate("SettingsDialog", "Browse…"))
    layout.addWidget(button)
    button.clicked.connect(lambda: _pick_folder(line, parent, title))
    return row, button


def _pick_folder(line: QLineEdit, parent: QDialog, title: str) -> None:
    current = line.text() or ""
    chosen = QFileDialog.getExistingDirectory(parent, title, current)
    if chosen:
        line.setText(chosen)


class SettingsDialog(QDialog):
    """Edits the §16 keys; language applies live, the rest on accept."""

    def __init__(
        self,
        settings: Settings,
        languages: LanguageManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._languages = languages

        self._language = QComboBox()
        self._language.addItems(list(LANGUAGE_OPTIONS))
        self._language.setCurrentText(settings.language)
        self._theme = QComboBox()
        self._theme.addItems(list(THEME_OPTIONS))
        self._theme.setCurrentText(settings.theme)

        self._blocks_folder = QLineEdit(str(settings.blocks_folder))
        self._results_root = QLineEdit(str(settings.results_root))

        self._workers = QSpinBox()
        self._workers.setRange(1, 8)
        self._workers.setValue(settings.workers)
        self._memory_ceiling = QSpinBox()
        self._memory_ceiling.setRange(512, 8192)
        self._memory_ceiling.setSingleStep(512)
        self._memory_ceiling.setValue(settings.memory_ceiling_mb)
        self._timeout = QSpinBox()
        self._timeout.setRange(1, 86_400)
        self._timeout.setValue(settings.experiment_timeout_s)
        self._auto_reload = QCheckBox()
        self._auto_reload.setChecked(settings.auto_reload_blocks)

        self._proxy = QSpinBox()
        self._proxy.setRange(128, 4096)
        self._proxy.setSingleStep(128)
        self._proxy.setValue(settings.proxy_long_side)
        self._debounce = QSpinBox()
        self._debounce.setRange(0, 5_000)
        self._debounce.setSingleStep(10)
        self._debounce.setValue(settings.debounce_ms)

        self._undo_limit = QSpinBox()
        self._undo_limit.setRange(1, 10_000)
        self._undo_limit.setValue(settings.undo_limit)
        self._log_level = QComboBox()
        self._log_level.addItems(list(LOG_LEVELS))
        self._log_level.setCurrentText(settings.log_level)

        self._labels: dict[str, QLabel] = {}
        self._browse_buttons: dict[str, QPushButton] = {}

        form = QFormLayout()
        language_row = QWidget(self)
        language_layout = QHBoxLayout(language_row)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.addWidget(self._language)
        note = QLabel()
        language_layout.addWidget(note)
        self._language_note = note
        form.addRow(self._row("language", self._language), language_row)
        form.addRow(self._row("theme", self._theme), self._theme)
        blocks_row, blocks_button = _browse_row(self._blocks_folder, self, self.tr("Blocks folder"))
        self._browse_buttons["blocks_folder"] = blocks_button
        form.addRow(self._row("blocks_folder", blocks_row), blocks_row)
        results_row, results_button = _browse_row(self._results_root, self, self.tr("Results root"))
        self._browse_buttons["results_root"] = results_button
        form.addRow(self._row("results_root", results_row), results_row)

        runner_form = QFormLayout()
        runner_form.addRow(self._row("workers", self._workers), self._workers)
        runner_form.addRow(self._row("memory_ceiling", self._memory_ceiling), self._memory_ceiling)
        runner_form.addRow(self._row("timeout", self._timeout), self._timeout)
        runner_form.addRow(self._row("auto_reload", self._auto_reload), self._auto_reload)
        self._runner_box = QGroupBox(self.tr("Runner"), self)
        self._runner_box.setLayout(runner_form)

        preview_form = QFormLayout()
        preview_form.addRow(self._row("proxy", self._proxy), self._proxy)
        preview_form.addRow(self._row("debounce", self._debounce), self._debounce)
        self._preview_box = QGroupBox(self.tr("Preview"), self)
        self._preview_box.setLayout(preview_form)

        general_form = QFormLayout()
        general_form.addRow(self._row("undo_limit", self._undo_limit), self._undo_limit)
        general_form.addRow(self._row("log_level", self._log_level), self._log_level)
        self._general_box = QGroupBox(self.tr("General"), self)
        self._general_box.setLayout(general_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._runner_box)
        layout.addWidget(self._preview_box)
        layout.addWidget(self._general_box)
        layout.addWidget(buttons)

        self._language.currentTextChanged.connect(self._apply_language)
        languages.language_changed.connect(self.retranslate)
        self.retranslate()

    def _row(self, key: str, buddy: QWidget) -> QLabel:
        label = QLabel(self)
        label.setBuddy(buddy)
        self._labels[key] = label
        return label

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Settings"))
        self._labels["language"].setText(self.tr("&Language"))
        self._labels["theme"].setText(self.tr("&Theme"))
        self._labels["blocks_folder"].setText(self.tr("&Blocks folder"))
        self._labels["results_root"].setText(self.tr("&Results root"))
        self._labels["workers"].setText(self.tr("&Workers"))
        self._labels["memory_ceiling"].setText(self.tr("&Memory ceiling (MB)"))
        self._labels["timeout"].setText(self.tr("Experiment &timeout (s)"))
        self._labels["auto_reload"].setText(self.tr("Auto-reload &blocks"))
        self._labels["proxy"].setText(self.tr("Proxy long &side"))
        self._labels["debounce"].setText(self.tr("&Debounce (ms)"))
        self._labels["undo_limit"].setText(self.tr("&Undo limit"))
        self._labels["log_level"].setText(self.tr("&Log level"))
        self._runner_box.setTitle(self.tr("Runner"))
        self._preview_box.setTitle(self.tr("Preview"))
        self._general_box.setTitle(self.tr("General"))
        for button in self._browse_buttons.values():
            button.setText(self.tr("Browse…"))
        self._language_note.setText(self.tr("applies immediately"))
        self._auto_reload.setText(self.tr("enabled"))

    # --- slots -------------------------------------------------------------

    def _apply_language(self, value: str) -> None:
        """Live-apply the language so the user sees the effect (AC-12)."""
        self._settings.set_language(value)
        self._languages.set_language(value)
        self.retranslate()

    def accept(self) -> None:
        """Write every edited key (except language, already applied)."""
        self._settings.set_theme(self._theme.currentText())
        self._settings.set_blocks_folder(self._browse_value(self._blocks_folder))
        self._settings.set_results_root(self._browse_value(self._results_root))
        self._settings.set_workers(self._workers.value())
        self._settings.set_memory_ceiling_mb(self._memory_ceiling.value())
        self._settings.set_experiment_timeout_s(self._timeout.value())
        self._settings.set_auto_reload_blocks(self._auto_reload.isChecked())
        self._settings.set_proxy_long_side(self._proxy.value())
        self._settings.set_debounce_ms(self._debounce.value())
        self._settings.set_undo_limit(self._undo_limit.value())
        self._settings.set_log_level(self._log_level.currentText())
        super().accept()

    @staticmethod
    def _browse_value(line: QLineEdit) -> Path:
        return Path(line.text().strip())

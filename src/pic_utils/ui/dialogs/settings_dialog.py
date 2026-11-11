"""Settings dialog (architecture §13.6, §16).

Modal; only decisions live here. The language applies immediately (live
preview, AC-12); every other value is written on accept. Settings changes
that affect a running service apply to the next run.
"""

from __future__ import annotations

from pathlib import Path

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


def _browse_row(line: QLineEdit, parent: QDialog, title: str) -> QWidget:
    row = QWidget(parent)
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(line)
    button = QPushButton(parent.tr("Browse…"))
    layout.addWidget(button)
    button.clicked.connect(lambda: _pick_folder(line, parent, title))
    return row


def _pick_folder(line: QLineEdit, parent: QDialog, title: str) -> None:
    current = line.text() or None
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

        form = QFormLayout()
        language_row = QWidget(self)
        language_layout = QHBoxLayout(language_row)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.addWidget(self._language)
        note = QLabel()
        language_layout.addWidget(note)
        self._language_note = note
        form.addRow(self._label_row(self.tr("&Language"), self._language), language_row)
        form.addRow(self._label_row(self.tr("&Theme"), self._theme), self._theme)
        form.addRow(
            self.tr("&Blocks folder"),
            _browse_row(self._blocks_folder, self, self.tr("Blocks folder")),
        )
        form.addRow(
            self.tr("&Results root"),
            _browse_row(self._results_root, self, self.tr("Results root")),
        )

        runner_form = QFormLayout()
        runner_form.addRow(self._label_row(self.tr("&Workers"), self._workers), self._workers)
        runner_form.addRow(
            self._label_row(self.tr("&Memory ceiling (MB)"), self._memory_ceiling),
            self._memory_ceiling,
        )
        runner_form.addRow(
            self._label_row(self.tr("Experiment &timeout (s)"), self._timeout), self._timeout
        )
        runner_form.addRow(self.tr("Auto-reload &blocks"), self._auto_reload)
        runner_box = QGroupBox(self.tr("Runner"))
        runner_box.setLayout(runner_form)

        preview_form = QFormLayout()
        preview_form.addRow(
            self._label_row(self.tr("Proxy long &side"), self._proxy), self._proxy
        )
        preview_form.addRow(
            self._label_row(self.tr("&Debounce (ms)"), self._debounce), self._debounce
        )
        preview_box = QGroupBox(self.tr("Preview"))
        preview_box.setLayout(preview_form)

        general_form = QFormLayout()
        general_form.addRow(
            self._label_row(self.tr("&Undo limit"), self._undo_limit), self._undo_limit
        )
        general_form.addRow(self._label_row(self.tr("&Log level"), self._log_level), self._log_level)
        general_box = QGroupBox(self.tr("General"))
        general_box.setLayout(general_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(runner_box)
        layout.addWidget(preview_box)
        layout.addWidget(general_box)
        layout.addWidget(buttons)

        self._language.currentTextChanged.connect(self._apply_language)
        self.retranslate()

    @staticmethod
    def _label_row(text: str, buddy: QWidget) -> QLabel:
        label = QLabel(text)
        label.setBuddy(buddy)
        return label

    def retranslate(self) -> None:
        """Set every user-visible string (§15)."""
        self.setWindowTitle(self.tr("Settings"))
        self._language_note.setText(self.tr("applies immediately"))
        self._auto_reload.setText(self.tr("enabled"))

    # --- slots -------------------------------------------------------------

    def _apply_language(self, value: str) -> None:
        """Live-apply the language so the user sees the effect (AC-12)."""
        self._settings.set_language(value)
        self._languages.set_language(value)
        retranslate = getattr(self.parent(), "retranslate", None)
        if callable(retranslate):
            retranslate()

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

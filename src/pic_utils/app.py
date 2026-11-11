"""Application entry point (architecture §21.3).

Wires the shell together: ``QApplication``, typed settings, logging,
language manager, theme, session trash, and the main window. The console
script ``pic-utils`` calls :func:`main`.
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import sys

from PySide6.QtWidgets import QApplication

from pic_utils.logging_setup import setup_logging
from pic_utils.results.trash import prepare_trash
from pic_utils.settings import Settings, logs_dir
from pic_utils.ui.i18n import LanguageManager
from pic_utils.ui.main_window import MainWindow
from pic_utils.ui.theme import apply_theme

__all__ = ["main"]

_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def main(argv: list[str] | None = None) -> int:
    """Launch the app; return the Qt exit code."""
    # Workers and the preview process always use spawn (§1.2, §8.2).
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)
    app = QApplication(sys.argv if argv is None else argv)
    settings = Settings()
    setup_logging(level=_LOG_LEVELS[settings.log_level], log_dir=logs_dir())
    settings.ensure_paths()
    languages = LanguageManager(settings)
    apply_theme(app, settings.theme)

    trash = prepare_trash(settings.results_root)
    window = MainWindow(settings=settings, languages=languages)
    settings.changed.connect(lambda key: _on_setting_changed(key, settings, app, window))
    window.show()
    code = app.exec()
    trash.cleanup_current()
    return code


def _on_setting_changed(
    key: str, settings: Settings, app: QApplication, window: MainWindow
) -> None:
    """React to live settings: theme switches immediately (§16)."""
    if key == "ui/theme":
        apply_theme(app, settings.theme)

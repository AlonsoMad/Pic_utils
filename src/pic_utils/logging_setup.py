"""Application and per-run logging (architecture §16, §18.3).

stdlib ``logging`` only — no Qt, no ``print``. :func:`setup_logging`
configures the root logger with a rotating file handler (5 MB x 3 backups)
and a console handler; :func:`run_logger` returns a logger writing
``<run_dir>/run.log`` for a single batch run (the parent is the single
writer; workers send records through the event queue).
"""

from __future__ import annotations

import logging
import logging.handlers
import os

from pic_utils.core.paths import ensure_dir

logger = logging.getLogger(__name__)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_LOG_FILE_NAME = "app.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file, 3 backups (architecture §18.3)
_BACKUP_COUNT = 3


class _ConsoleHandler(logging.StreamHandler):
    """Console handler installed by :func:`setup_logging` (idempotency marker)."""


class _AppFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating app-log handler installed by :func:`setup_logging` (idempotency marker)."""


def _remove_installed(target: logging.Logger) -> None:
    """Remove and close handlers previously installed by :func:`setup_logging`."""
    for handler in list(target.handlers):
        if isinstance(handler, _ConsoleHandler | _AppFileHandler):
            target.removeHandler(handler)
            handler.close()


def setup_logging(level: int = logging.INFO, log_dir: str | os.PathLike[str] | None = None) -> None:
    """Configure the root logger for the application.

    Installs a console handler and, when ``log_dir`` is given, a rotating
    file handler writing ``<log_dir>/app.log`` (5 MB per file, 3 backups).
    Both handlers use ``level``. Idempotent: re-calling replaces the handlers
    installed by a previous call instead of duplicating them.
    """
    root = logging.getLogger()
    root.setLevel(level)
    _remove_installed(root)
    formatter = logging.Formatter(_LOG_FORMAT)

    console = _ConsoleHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is not None:
        file_handler = _AppFileHandler(
            ensure_dir(log_dir) / _LOG_FILE_NAME,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def run_logger(run_dir: str | os.PathLike[str]) -> logging.Logger:
    """Return a logger that writes ``<run_dir>/run.log``.

    The file is opened in append mode and the logger level is ``DEBUG`` so
    the run log captures full detail. Calling again for the same directory
    returns the same cached logger without adding a second handler.
    """
    directory = ensure_dir(run_dir)
    log_path = directory / "run.log"
    run_log = logging.getLogger(f"pic_utils.run.{os.path.abspath(log_path)}")
    run_log.setLevel(logging.DEBUG)
    target = os.path.abspath(log_path)
    already_installed = any(
        isinstance(handler, logging.FileHandler) and handler.baseFilename == target
        for handler in run_log.handlers
    )
    if not already_installed:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        run_log.addHandler(handler)
    return run_log

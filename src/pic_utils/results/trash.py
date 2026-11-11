"""Session trash lifecycle (architecture §9.3).

Discarded results move to ``<results_root>/trash/<session_id>/<run_id>/…``
preserving their layout inside the run, so a restore is the inverse move.
The order of operations is fixed: move first, then record the curation
decision — a crash in between leaves an orphan file in trash, which is
harmless. A crash after recording but before moving cannot happen because
the move comes first.

A session id is generated at app start (``YYYYMMDD-HHMMSS-<pid>``). At
startup every trash directory except the current session's is deleted; on a
clean exit the current session's trash is deleted; a crash leaves it for the
next startup.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path, PurePosixPath

from pic_utils.core.paths import ensure_dir

__all__ = ["SessionTrash", "new_session_id"]

_TRASH_DIR = "trash"


def new_session_id(now: datetime | None = None) -> str:
    """Return ``YYYYMMDD-HHMMSS-<pid>`` (§9.3)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{os.getpid()}"


def _relative_run_path(relative: str) -> Path:
    """Validate a run-relative path from a result record.

    Paths come from ``results.jsonl``, which is user-shareable data, so the
    trash boundary rejects absolute paths and ``.``/``..`` components rather
    than trusting them: ``os.replace`` would otherwise move files outside
    the run's trash subtree.
    """
    if "\\" in relative or "\x00" in relative:
        raise ValueError(f"result path {relative!r} must use '/' separators")
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or ".." in parts or "." in parts:
        raise ValueError(f"result path {relative!r} must be relative to the run directory")
    return Path(*parts)


class SessionTrash:
    """Moves discarded run files into (and out of) this session's trash."""

    def __init__(self, results_root: Path, session_id: str | None = None) -> None:
        self.results_root = Path(results_root)
        self._session_id = session_id or new_session_id()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def trash_root(self) -> Path:
        """``<results_root>/trash`` — shared by all sessions."""
        return self.results_root / _TRASH_DIR

    @property
    def session_dir(self) -> Path:
        return self.trash_root / self._session_id

    def discard(self, run_id: str, relative_paths: list[str]) -> list[Path]:
        """Move run files into this session's trash; return destinations.

        ``relative_paths`` are relative to the run directory (as stored in
        :class:`~pic_utils.runner.manifest.ResultRecord.path`). Missing
        sources are skipped so a partially written experiment can still be
        discarded cleanly.
        """
        destinations: list[Path] = []
        for relative in relative_paths:
            safe_relative = _relative_run_path(relative)
            source = self.results_root / run_id / safe_relative
            if not source.is_file():
                continue
            destination = self.session_dir / run_id / safe_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            destinations.append(destination)
        return destinations

    def restore(self, run_id: str, relative_paths: list[str]) -> list[Path]:
        """Move trashed files back into their run; return original paths.

        A missing trash file is skipped: the caller then decides whether the
        curation record should be cleared anyway (it is stale either way).
        """
        restored: list[Path] = []
        for relative in relative_paths:
            safe_relative = _relative_run_path(relative)
            trashed = self.session_dir / run_id / safe_relative
            if not trashed.is_file():
                continue
            original = self.results_root / run_id / safe_relative
            original.parent.mkdir(parents=True, exist_ok=True)
            os.replace(trashed, original)
            restored.append(original)
        return restored

    def cleanup_other_sessions(self) -> None:
        """Delete every trash session directory except this one (startup)."""
        if not self.trash_root.is_dir():
            return
        for child in self.trash_root.iterdir():
            if child.name != self._session_id:
                self._delete(child)

    def cleanup_current(self) -> None:
        """Delete this session's trash directory (clean exit)."""
        self._delete(self.session_dir)

    @staticmethod
    def _delete(directory: Path) -> None:
        shutil.rmtree(directory, ignore_errors=True)


def prepare_trash(results_root: Path, session_id: str | None = None) -> SessionTrash:
    """Create the session trash and drop other sessions' leftovers.

    Convenience for app startup: constructs the session's trash and prunes
    directories left behind by crashed sessions.
    """
    trash = SessionTrash(Path(results_root), session_id)
    ensure_dir(trash.session_dir)
    trash.cleanup_other_sessions()
    return trash

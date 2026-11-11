"""Results, curation, and trash (architecture §9).

``store`` lists runs and indexes accepted results for the canvas tray;
``trash`` owns the per-session discard/restore lifecycle. Both sit on top of
the run layout owned by :mod:`pic_utils.runner.manifest`.
"""

from __future__ import annotations

from pic_utils.results.store import (
    AcceptedIndex,
    AcceptedResult,
    RunRef,
    discard_result,
    list_runs,
    load_run,
    restore_result,
)
from pic_utils.results.trash import SessionTrash, new_session_id, prepare_trash

__all__ = [
    "AcceptedIndex",
    "AcceptedResult",
    "RunRef",
    "SessionTrash",
    "discard_result",
    "list_runs",
    "load_run",
    "new_session_id",
    "prepare_trash",
    "restore_result",
]

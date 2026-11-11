"""Run listing and the accepted-results index (architecture §9.3-§9.4).

``list_runs`` enumerates the runs under a results root (newest first).
``AcceptedIndex`` keeps the accepted results in memory for the canvas asset
tray; it is refreshed from disk on demand and updated incrementally on
curation events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pic_utils.core.jsonio import read_json
from pic_utils.results.trash import SessionTrash
from pic_utils.runner.manifest import RUN_FORMAT, ResultRecord, RunData, RunStore

__all__ = [
    "AcceptedIndex",
    "AcceptedResult",
    "RunRef",
    "discard_result",
    "list_runs",
    "load_run",
    "restore_result",
]

_TRASH_DIR = "trash"


@dataclass(frozen=True)
class RunRef:
    """One run directory found under a results root."""

    run_id: str
    path: Path
    status: str


def list_runs(results_root: Path) -> tuple[RunRef, ...]:
    """Return every run under ``results_root``, newest first.

    A directory counts as a run when it contains a ``manifest.json`` with the
    Pic Utils run format; directories without one (including ``trash``) are
    ignored.
    """
    root = Path(results_root)
    if not root.is_dir():
        return ()
    runs: list[RunRef] = []
    for child in sorted((path for path in root.iterdir() if path.is_dir()), reverse=True):
        if child.name == _TRASH_DIR:
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        header = read_json(manifest_path)
        if not isinstance(header, dict) or header.get("format") != RUN_FORMAT:
            continue
        status = header.get("status")
        runs.append(
            RunRef(
                run_id=child.name,
                path=child,
                status=status if isinstance(status, str) else "unknown",
            )
        )
    return tuple(runs)


def load_run(results_root: Path, run_id: str) -> RunData:
    """Load one run's manifest, results, and curation."""
    return RunStore(Path(results_root), run_id).read()


def discard_result(store: RunStore, trash: SessionTrash, record: ResultRecord) -> None:
    """Discard one result: move file and thumb to trash, then record it.

    The move happens first (§9.3): a crash between the move and the curation
    write leaves an orphan file in trash, which is harmless, while recording
    first could point the grid at a file that is no longer there.
    """
    trash.discard(store.run_id, [record.path, record.thumb])
    store.set_curation(record.id, "discarded")


def restore_result(store: RunStore, trash: SessionTrash, record: ResultRecord) -> None:
    """Restore one result: move file and thumb back, then clear the state."""
    trash.restore(store.run_id, [record.path, record.thumb])
    store.clear_curation(record.id)


@dataclass(frozen=True)
class AcceptedResult:
    """One accepted result, tagged with the run that produced it."""

    run_id: str
    record: ResultRecord


class AcceptedIndex:
    """In-memory index of accepted results for the canvas asset tray (§9.3)."""

    def __init__(self, results_root: Path) -> None:
        self._results_root = Path(results_root)
        self._accepted: dict[str, dict[str, ResultRecord]] = {}  # run_id -> id -> record

    @property
    def results_root(self) -> Path:
        """The results root this index scans."""
        return self._results_root

    def refresh(self) -> None:
        """Rescan every run and rebuild the index from disk."""
        indexed: dict[str, dict[str, ResultRecord]] = {}
        for run_ref in list_runs(self._results_root):
            run_data = load_run(self._results_root, run_ref.run_id)
            accepted = _accepted_from(run_data)
            if accepted:
                indexed[run_ref.run_id] = accepted
        self._accepted = indexed

    def apply(self, run_id: str, result_id_value: str, state: str | None) -> None:
        """Apply one curation event without a full rescan.

        ``state`` is ``"accepted"``, ``"discarded"``, or ``None`` (pending
        again). The record must already be known from a previous
        :meth:`refresh`; an unknown id is ignored — the next
        :meth:`refresh` reconciles the index with disk.
        """
        if state == "accepted":
            run_results = self._accepted.get(run_id)
            if run_results is None:
                record = self._load_record(run_id, result_id_value)
                if record is None:
                    return
                self._accepted.setdefault(run_id, {})[result_id_value] = record
            return
        run_results = self._accepted.get(run_id)
        if run_results is not None:
            run_results.pop(result_id_value, None)
            if not run_results:
                del self._accepted[run_id]

    def results(self) -> tuple[AcceptedResult, ...]:
        """Return every accepted result, newest run first."""
        run_ids = sorted(self._accepted, reverse=True)
        return tuple(
            AcceptedResult(run_id=run_id, record=record)
            for run_id in run_ids
            for record in self._accepted[run_id].values()
        )

    def _load_record(self, run_id: str, result_id_value: str) -> ResultRecord | None:
        try:
            run_data = load_run(self._results_root, run_id)
        except FileNotFoundError:
            # Unknown run: ignore until the next refresh reconciles with disk.
            return None
        return next((record for record in run_data.results if record.id == result_id_value), None)


def _accepted_from(run_data: RunData) -> dict[str, ResultRecord]:
    by_id: dict[str, ResultRecord] = {record.id: record for record in run_data.results}
    accepted: dict[str, ResultRecord] = {}
    for result_id_value, state in run_data.curation.items():
        record = by_id.get(result_id_value)
        if record is not None and state == "accepted":
            accepted[result_id_value] = record
    return accepted

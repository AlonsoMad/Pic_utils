"""Run folder layout, manifest, and result records (architecture §9.1-§9.2).

A run is one directory under the results root::

    <results_root>/<run_id>/
        manifest.json     # header, config, pipeline, sources, summary
        results.jsonl     # append-only result records, one JSON per line
        curation.json     # {"<result_id>": "accepted"|"discarded"}
        results/          # full-resolution outputs
        thumbs/           # 256 px PNG thumbnails
        blocks/           # copied user-block sources for this run
        run.log           # written by the parent via logging_setup.run_logger

``manifest.json`` and ``curation.json`` are small and are rewritten
atomically with :func:`pic_utils.core.jsonio.write_json_atomic`.
``results.jsonl`` is append-only and flushed after every record, so a crash
costs at most the record being written — never previously written ones.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pic_utils.core.jsonio import read_json, write_json_atomic

__all__ = [
    "RUN_FORMAT",
    "RUN_FORMAT_VERSION",
    "ResultRecord",
    "RunData",
    "RunStore",
    "new_run_id",
    "result_id",
]

RUN_FORMAT = "pic_utils.run"
RUN_FORMAT_VERSION = 1

_MANIFEST_NAME = "manifest.json"
_RESULTS_NAME = "results.jsonl"
_CURATION_NAME = "curation.json"
_RESULTS_DIR = "results"
_THUMBS_DIR = "thumbs"
_BLOCKS_DIR = "blocks"


def result_id(experiment: int, layer: int) -> str:
    """Return the stable result identifier ``"{experiment:04d}:{layer:02d}"``."""
    return f"{experiment:04d}:{layer:02d}"


def new_run_id(now: datetime | None = None) -> str:
    """Return ``YYYYMMDD-HHMMSS-<6 hex>`` (local time + random suffix, §9.1)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


@dataclass(frozen=True)
class ResultRecord:
    """One produced result row (``results.jsonl`` line, §9.2)."""

    id: str  # "{experiment:04d}:{layer:02d}"
    experiment: int
    layer: int
    source: int
    kind: str  # "image"
    path: str  # relative to run_dir
    thumb: str  # relative to run_dir
    width: int
    height: int
    seed: int
    params: dict[str, dict[str, Any]]
    elapsed_ms: int
    accepted: bool | None = None  # None = pending; filled from curation.json

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe dict written to ``results.jsonl``."""
        return {
            "id": self.id,
            "experiment": self.experiment,
            "layer": self.layer,
            "source": self.source,
            "kind": self.kind,
            "path": self.path,
            "thumb": self.thumb,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "params": self.params,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, raw: object) -> ResultRecord:
        """Rebuild a record from a ``results.jsonl`` line.

        Raises:
            ValueError: if the line is not a result record.
        """
        if not isinstance(raw, dict):
            raise ValueError("result record must be a JSON object")
        try:
            return cls(
                id=raw["id"],
                experiment=raw["experiment"],
                layer=raw["layer"],
                source=raw["source"],
                kind=raw["kind"],
                path=raw["path"],
                thumb=raw["thumb"],
                width=raw["width"],
                height=raw["height"],
                seed=raw["seed"],
                params=raw["params"],
                elapsed_ms=raw["elapsed_ms"],
            )
        except KeyError as error:
            raise ValueError(f"result record is missing {error.args[0]!r}") from error
        except TypeError as error:
            raise ValueError(f"result record has a field of the wrong type: {error}") from error


@dataclass(frozen=True)
class RunData:
    """Everything the results grid and canvas tray need about one run."""

    header: dict[str, Any]  # manifest.json
    results: tuple[ResultRecord, ...]
    curation: dict[str, str]  # result id -> "accepted" | "discarded"


class RunStore:
    """Owns the on-disk layout of one run and its atomic/append-only writes."""

    def __init__(self, results_root: Path, run_id: str) -> None:
        self.results_root = Path(results_root)
        self.run_id = run_id

    @property
    def run_dir(self) -> Path:
        return self.results_root / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / _MANIFEST_NAME

    @property
    def results_jsonl_path(self) -> Path:
        return self.run_dir / _RESULTS_NAME

    @property
    def curation_path(self) -> Path:
        return self.run_dir / _CURATION_NAME

    def create(self, header: dict[str, Any]) -> Path:
        """Create the run folder tree and write the initial manifest.

        ``header`` must contain the run's identity fields; ``status`` is
        forced to ``"running"``. Called once, at run start (§8.7 step 2).
        """
        for directory in (_RESULTS_DIR, _THUMBS_DIR, _BLOCKS_DIR):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.curation_path, {})
        self.write_manifest({**header, "status": "running"})
        return self.run_dir

    def write_manifest(self, header: dict[str, Any]) -> None:
        """Atomically rewrite ``manifest.json`` (start, finish, and updates)."""
        payload = {
            "format": RUN_FORMAT,
            "version": RUN_FORMAT_VERSION,
            "run_id": self.run_id,
            **header,
        }
        write_json_atomic(self.manifest_path, payload)

    def append_result(self, record: ResultRecord) -> None:
        """Append one result record to ``results.jsonl`` and flush it.

        The file is opened in append mode per record: a crash between writes
        never damages earlier lines, and there is no O(n²) rewrite.
        """
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with self.results_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def read_curation(self) -> dict[str, str]:
        """Return ``curation.json`` contents; missing file means no decisions."""
        if not self.curation_path.exists():
            return {}
        curation = read_json(self.curation_path)
        if not isinstance(curation, dict):
            raise ValueError(f"{self.curation_path}: curation must be a JSON object")
        return curation

    def set_curation(self, result_id_value: str, state: str) -> None:
        """Record one curation decision and atomically rewrite the file.

        ``state`` is ``"accepted"`` or ``"discarded"``. Callers move files
        first and record second (§9.3): a crash between the two leaves an
        orphan in trash, which is harmless.
        """
        if state not in ("accepted", "discarded"):
            raise ValueError(f"curation state must be 'accepted' or 'discarded', got {state!r}")
        curation = self.read_curation()
        curation[result_id_value] = state
        write_json_atomic(self.curation_path, curation)

    def clear_curation(self, result_id_value: str) -> None:
        """Remove one curation decision (restore to pending)."""
        curation = self.read_curation()
        if curation.pop(result_id_value, None) is not None:
            write_json_atomic(self.curation_path, curation)

    def read(self) -> RunData:
        """Load manifest, results, and curation for display.

        A trailing line torn by a crash is skipped; a malformed line in the
        middle is an error, not silently dropped data.
        """
        header = read_json(self.manifest_path)
        results = tuple(self._read_results())
        return RunData(header=header, results=results, curation=self.read_curation())

    def _read_results(self) -> list[ResultRecord]:
        if not self.results_jsonl_path.exists():
            return []
        lines = self.results_jsonl_path.read_text(encoding="utf-8").splitlines()
        records: list[ResultRecord] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(ResultRecord.from_dict(json.loads(line)))
            except ValueError as error:
                if index == len(lines) - 1:
                    # A crash mid-append can tear the final line only.
                    break
                raise ValueError(f"{self.results_jsonl_path} line {index + 1}: {error}") from error
        return records

    def result_file(self, experiment: int, layer: int) -> Path:
        """Full-resolution output path ``results/result_<exp:04d>_<layer:02d>.png``."""
        return self.run_dir / _RESULTS_DIR / f"result_{experiment:04d}_{layer:02d}.png"

    def thumb_file(self, experiment: int, layer: int) -> Path:
        """Thumbnail path ``thumbs/result_<exp:04d>_<layer:02d>.png``."""
        return self.run_dir / _THUMBS_DIR / f"result_{experiment:04d}_{layer:02d}.png"

    def blocks_dir(self) -> Path:
        """The run's copied user-blocks folder (§8.2)."""
        return self.run_dir / _BLOCKS_DIR

    @staticmethod
    def now_utc_iso() -> str:
        """UTC ISO-8601 timestamp for manifest fields (§2 conventions)."""
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

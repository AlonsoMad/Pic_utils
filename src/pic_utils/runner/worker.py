"""Spawn-safe worker entrypoint (architecture §8.2-§8.3).

Workers are plain processes: ``init_worker`` sets up queue-routed logging
and the block snapshot, ``run_experiment`` executes one task and writes its
results to disk. Neither imports Qt — the parent owns ``run.log`` and
receives outcomes and events through the queue.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import pkgutil
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import psutil  # type: ignore[import-untyped]  # psutil ships no py.typed; types-psutil is not an approved dependency

from pic_utils.blocks import builtin as builtin_package
from pic_utils.blocks.api import BlockDef, BlockLoadError
from pic_utils.blocks.userblocks import collect_block_defs, load_user_module
from pic_utils.core.errors import (
    BLOCK_INVALID_OUTPUT,
    BLOCK_RAISED,
    BLOCK_SNAPSHOT_MISMATCH,
    IO_WRITE,
    MEMORY_LIMIT,
    BlockError,
    ExportError,
    InvalidBlockOutput,
    RunError,
)
from pic_utils.core.paths import content_hash
from pic_utils.io.saver import save_image
from pic_utils.pipeline.execute import execute_pipeline
from pic_utils.pipeline.model import Pipeline
from pic_utils.runner.manifest import RunStore
from pic_utils.runner.memory import MemoryGuard
from pic_utils.runner.protocol import (
    EventQueue,
    ExperimentOutcome,
    ExperimentStarted,
    ExperimentTask,
    ResultFile,
    WorkerInit,
    WorkerLog,
    WorkerReady,
)
from pic_utils.runner.shm import attach_image

logger = logging.getLogger(__name__)

_THUMB_LONG_SIDE = 256
_MIB = 1024 * 1024

if TYPE_CHECKING:
    # blocks/registry.py imports Qt; the type reference is compile-time only.
    from pic_utils.blocks.registry import BlockSnapshot


_worker_snapshot: _WorkerSnapshot | None = None
_worker_snapshot_hash: str = ""
_worker_snapshot_errors: tuple[BlockLoadError, ...] = ()
_worker_events: EventQueue | None = None
_worker_pid: int = 0


@dataclass(frozen=True, slots=True)
class _WorkerSnapshot:
    """Structural mirror of ``BlockSnapshot`` (registry.py imports Qt).

    Kept field-for-field with ``BlockRegistry._snapshot_hash`` so the worker
    can verify the run's hash without a Qt import. If ``BlockSnapshot``
    changes shape, this mirror and the hash payload must change with it —
    the failure mode is a loud ``block.snapshot_mismatch`` on every run,
    never a silent mismatch.
    """

    defs: Mapping[str, BlockDef]
    hash: str

    def get(self, name: str) -> BlockDef:
        return self.defs[name]

    def __contains__(self, name: object) -> bool:
        return name in self.defs

    def __iter__(self) -> Iterator[str]:
        return iter(self.defs)

    def __len__(self) -> int:
        return len(self.defs)


class _QueueLogHandler(logging.Handler):
    """Forwards worker log records to the parent's run.log via the queue."""

    def __init__(self, events: EventQueue, pid: int) -> None:
        super().__init__(level=logging.DEBUG)
        self._events = events
        self._pid = pid

    def emit(self, record: logging.LogRecord) -> None:
        try:
            traceback_text = None
            if record.exc_info is not None or record.exc_text is not None:
                traceback_text = self.format(record)
            self._events.put_nowait(
                WorkerLog(
                    pid=self._pid,
                    level=record.levelno,
                    logger_name=record.name,
                    message=record.getMessage(),
                    traceback=traceback_text,
                )
            )
        except Exception:  # pragma: no cover - logging must never raise
            self.handleError(record)


def init_worker(init: WorkerInit, events: EventQueue) -> None:
    """Initialize one worker process (§8.2).

    1. Record the PID in the event queue (``WorkerReady``).
    2. Install a queue log handler so worker log records reach ``run.log``
       through the parent (single writer, no cross-process interleaving).
    3. Add the run's copied blocks folder to ``sys.path``.
    4. Build the block snapshot from built-ins + the copied user blocks and
       verify its hash against the run's ``block_snapshot_hash``; a mismatch
       fails every task with ``block.snapshot_mismatch``.
    """
    global _worker_events, _worker_pid
    _worker_events = events
    _worker_pid = os.getpid()
    events.put(WorkerReady(pid=_worker_pid))
    # §18.3: the parent sends the configured level; without this the root
    # logger stays at WARNING and INFO/DEBUG records never reach run.log.
    level = logging.getLevelName(init.log_level.upper())
    logging.getLogger().setLevel(level if isinstance(level, int) else logging.WARNING)
    _install_queue_log_handler(events, _worker_pid)

    blocks_dir = Path(init.blocks_dir) if init.blocks_dir else None
    if blocks_dir is not None and blocks_dir.is_dir():
        sys.path.append(str(blocks_dir))
    _set_snapshot(init.block_snapshot_hash, blocks_dir)


def _install_queue_log_handler(events: EventQueue, pid: int) -> None:
    handler = _QueueLogHandler(events, pid)
    logging.getLogger().addHandler(handler)


def _set_snapshot(expected_hash: str, blocks_dir: Path | None) -> None:
    """Build the worker snapshot and verify the run's hash (§8.2)."""
    global _worker_snapshot, _worker_snapshot_hash, _worker_snapshot_errors
    definitions: dict[str, BlockDef] = {}
    errors: list[BlockLoadError] = []
    for module_info in pkgutil.iter_modules(builtin_package.__path__):
        if module_info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{builtin_package.__name__}.{module_info.name}")
        except Exception:
            # The registry (§5.4) degrades gracefully per module; the worker
            # mirrors that instead of dying in the pool initializer.
            logger.exception("built-in module %s failed to import", module_info.name)
            continue
        for definition in collect_block_defs(module, source=None, content_hash=""):
            if definition.name in definitions:
                errors.append(
                    BlockLoadError(
                        reason=f"duplicate block name {definition.name!r}",
                        block=definition.name,
                    )
                )
                continue
            definitions[definition.name] = definition

    if blocks_dir is not None and blocks_dir.is_dir():
        for path in sorted(blocks_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            try:
                module = load_user_module(path, blocks_dir)
            except Exception:
                logger.exception("user block %s failed to load", path)
                continue
            digest = content_hash(path)
            for definition in collect_block_defs(module, source=path, content_hash=digest):
                if definition.name in definitions:
                    errors.append(
                        BlockLoadError(
                            reason=f"duplicate block name {definition.name!r}",
                            block=definition.name,
                        )
                    )
                    continue
                definitions[definition.name] = definition

    # Mirror of BlockRegistry._snapshot_hash (P2-02): the worker must compute
    # the identical hash without importing the registry (Qt). If the two
    # drift, every run fails with block.snapshot_mismatch — loud, not silent.
    builtin_entries = sorted(
        (definition.name, definition.content_hash, definition.api_version)
        for definition in definitions.values()
        if definition.source is None
    )
    user_entries: list[tuple[str, str]] = []
    if blocks_dir is not None and blocks_dir.is_dir():
        # hash_user_files (§5.5) hashes every *.py under the folder via rglob,
        # excluding only __pycache__; native separators match the registry.
        user_entries = sorted(
            (str(path.relative_to(blocks_dir)), content_hash(path))
            for path in blocks_dir.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    payload = json.dumps(
        {"builtins": builtin_entries, "user": user_entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    _worker_snapshot_hash = digest
    _worker_snapshot_errors = tuple(errors)
    if expected_hash and digest != expected_hash:
        _worker_snapshot = None
        return
    _worker_snapshot = _WorkerSnapshot(defs=MappingProxyType(definitions), hash=digest)


def run_experiment(task: ExperimentTask) -> ExperimentOutcome:
    """Execute one experiment and write its results to disk (§8.3)."""
    started = time.perf_counter()
    if _worker_events is not None:
        # The parent attributes in-flight experiments to PIDs for crash
        # recovery (§8.5); it learns which experiment from this event.
        _worker_events.put(
            ExperimentStarted(pid=_worker_pid, experiment_index=task.experiment.index)
        )
    if _worker_snapshot is None:
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=BLOCK_SNAPSHOT_MISMATCH,
            error_message=_snapshot_error_message(),
            elapsed_ms=0,
            results=(),
        )
    try:
        pipeline = Pipeline.from_dict(task.pipeline)
        store = RunStore(Path(task.run_dir).parent, task.run_id)
        if task.block_snapshot_hash != _worker_snapshot_hash:
            # §8.3 step 2: rebuild when the hash differs; the service stamps
            # one hash per run, so a difference means the run's blocks
            # changed under us — fail loudly rather than run stale code.
            raise RunError(code=BLOCK_SNAPSHOT_MISMATCH)
        source = attach_image(task.source)
        second = attach_image(task.second) if task.second is not None else None
        # _WorkerSnapshot mirrors BlockSnapshot field-for-field; the real
        # class lives in blocks/registry.py, which imports Qt and therefore
        # cannot be imported here (§1.3: worker code must not import Qt).
        snapshot: BlockSnapshot = _worker_snapshot  # type: ignore[assignment]
        # §8.6: a worker aborts with memory.limit when the OS has less
        # available memory than the experiment's estimate.
        estimate_mb = MemoryGuard().estimate_experiment_mb(source, pipeline, snapshot)
        available = psutil.virtual_memory().available
        if available < estimate_mb * _MIB:
            logger.warning(
                "experiment %s aborted: %.0f MiB available, %d MiB estimated",
                task.experiment.index,
                available / _MIB,
                estimate_mb,
            )
            return ExperimentOutcome(
                experiment_index=task.experiment.index,
                status="failed",
                error_code=MEMORY_LIMIT,
                error_message=(
                    f"not enough available memory: {available // _MIB} MiB free, "
                    f"{estimate_mb} MiB estimated"
                ),
                elapsed_ms=_elapsed_ms(started),
                results=(),
            )
        output = execute_pipeline(pipeline, snapshot, source, second, task.experiment)
        results: list[ResultFile] = []
        for layer_index, layer in enumerate(output.layers):
            result_path = store.result_file(task.experiment.index, layer_index)
            thumb_path = store.thumb_file(task.experiment.index, layer_index)
            save_image(layer.image, result_path)
            save_image(layer.image.to_proxy(_THUMB_LONG_SIDE), thumb_path)
            results.append(
                ResultFile(
                    layer_index=layer_index,
                    path=result_path.relative_to(store.run_dir).as_posix(),
                    thumb_path=thumb_path.relative_to(store.run_dir).as_posix(),
                    width=layer.image.width,
                    height=layer.image.height,
                )
            )
    except InvalidBlockOutput as error:
        logger.exception("experiment %s failed: invalid block output", task.experiment.index)
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=BLOCK_INVALID_OUTPUT,
            error_message=str(error),
            elapsed_ms=_elapsed_ms(started),
            results=(),
        )
    except ExportError as error:
        # Result/thumbnail write failures (§18.2: io.write — the batch
        # continues); the writes live inside the try so they are covered.
        logger.exception("experiment %s failed: could not write results", task.experiment.index)
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=IO_WRITE,
            error_message=str(error),
            elapsed_ms=_elapsed_ms(started),
            results=(),
        )
    except RunError as error:
        # Infrastructure failures (shm.unavailable, snapshot mismatch) keep
        # their own stable code instead of being relabeled block.raised.
        logger.exception("experiment %s failed: %s", task.experiment.index, error.code)
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=error.code,
            error_message=str(error),
            elapsed_ms=_elapsed_ms(started),
            results=(),
        )
    except BlockError as error:
        logger.exception("experiment %s failed: block raised", task.experiment.index)
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=BLOCK_RAISED,
            error_message=str(error),
            elapsed_ms=_elapsed_ms(started),
            results=(),
        )
    except Exception as error:
        # A block that raises a plain exception (ZeroDivisionError, ...) is
        # still ``block.raised`` (§18.2): the experiment fails, the traceback
        # lands in run.log, and the batch continues (FR-RUN-06, FR-BLK-10).
        logger.exception("experiment %s failed: unexpected error", task.experiment.index)
        return ExperimentOutcome(
            experiment_index=task.experiment.index,
            status="failed",
            error_code=BLOCK_RAISED,
            error_message=str(error),
            elapsed_ms=_elapsed_ms(started),
            results=(),
        )
    return ExperimentOutcome(
        experiment_index=task.experiment.index,
        status="ok",
        error_code=None,
        error_message=None,
        elapsed_ms=_elapsed_ms(started),
        results=tuple(results),
    )


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000.0)


def _snapshot_error_message() -> str:
    """One-line message for the snapshot mismatch that blocked this task."""
    detail = "; ".join(error.reason for error in _worker_snapshot_errors)
    return f"worker block snapshot hash {_worker_snapshot_hash!r} does not match the run's" + (
        f" ({detail})" if detail else ""
    )

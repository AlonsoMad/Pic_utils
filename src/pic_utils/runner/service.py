"""Run service: one batch run at a time, off the GUI thread (§8.7).

``RunService`` is the only runner component that touches Qt: it is a
``QObject`` whose signals are emitted from the run thread and the pool's
drain thread; Qt's queued connections deliver them on the GUI thread. The
start sequence (validate → run folder → blocks copy → shared memory → plan
→ dispatch → collect → finalize) runs on a dedicated thread; the GUI thread
only calls :meth:`start` / :meth:`cancel` and receives signals.

The runner-facing settings arrive as a :class:`RunnerSettings` value object
(workers, memory ceiling, experiment timeout); the Qt settings layer
(Phase 6, §16) maps its keys onto it. The runner itself never imports Qt
settings machinery beyond this facade.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from pic_utils import __version__
from pic_utils.blocks.api import BLOCK_API_VERSION
from pic_utils.core.errors import (
    BLOCK_TIMEOUT,
    MEMORY_LIMIT,
    POOL_UNSTABLE,
    WORKER_CRASHED,
    RunError,
)
from pic_utils.core.image import Image
from pic_utils.core.paths import ensure_dir
from pic_utils.io.loader import load_image
from pic_utils.logging_setup import run_logger
from pic_utils.pipeline.model import ExperimentSpec, Pipeline, RunConfig
from pic_utils.pipeline.plan import build_plan
from pic_utils.pipeline.validate import (
    Issue,
    is_blocking,
    validate_pipeline,
    validate_run_config,
)
from pic_utils.runner.manifest import ResultRecord, RunStore, new_run_id
from pic_utils.runner.memory import MemoryGuard
from pic_utils.runner.pool import PoolCallbacks, SupervisedPool
from pic_utils.runner.protocol import (
    EventQueue,
    ExperimentOutcome,
    ExperimentTask,
    ImageHandle,
    WorkerInit,
    WorkerLog,
)
from pic_utils.runner.shm import SharedImageStore

if TYPE_CHECKING:
    from pic_utils.blocks.registry import BlockSnapshot

__all__ = ["RunRequest", "RunService", "RunSummary", "RunnerSettings"]

logger = logging.getLogger(__name__)

_SOURCE_NONE = "source.none"
_RUN_ERROR = "run.error"
_MIB = 1024 * 1024
_POLL_INTERVAL_S = 0.5


def _spawn_queue() -> EventQueue:
    """A fresh event queue on the spawn context (§8.2)."""
    from multiprocessing import get_context

    queue: EventQueue = get_context("spawn").Queue()
    return queue


@dataclass(frozen=True)
class RunnerSettings:
    """The runner-relevant slice of the app settings (§16 runner keys)."""

    workers: int = 2
    memory_ceiling_mb: int = 8192
    experiment_timeout_s: float = 300.0


@dataclass(frozen=True)
class RunRequest:
    """Everything one batch run needs (§8.7).

    ``user_blocks_dir`` is the folder the snapshot's user blocks were loaded
    from; the whole folder (helpers included, so the snapshot hash matches)
    is copied into the run's ``blocks/`` directory.
    """

    pipeline: Pipeline
    snapshot: BlockSnapshot
    sources: tuple[Path, ...]
    second: Path | None
    run_config: RunConfig
    results_root: Path
    settings: RunnerSettings
    user_blocks_dir: Path | None = None


@dataclass(frozen=True)
class RunSummary:
    """The end-of-run report (§8.7, FR-RUN-09)."""

    run_id: str
    status: str  # "completed" | "cancelled" | "failed"
    completed: int
    failed: int
    cancelled: int
    elapsed_ms: int


class RunService(QObject):
    """Orchestrates one batch run end to end (§8.7)."""

    run_started = Signal(str, int)  # run_id, total experiments
    progress = Signal(int, int)  # done, total
    result_ready = Signal(object)  # ResultRecord
    run_finished = Signal(object)  # RunSummary
    run_failed = Signal(str, str)  # code, message
    memory_updated = Signal(int)  # total RSS MB

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._running = False
        self._cancel_requested = threading.Event()
        self._force_cancel = threading.Event()
        self._run_thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, request: RunRequest) -> None:
        """Begin one run on a dedicated thread; one run at a time (§8.7)."""
        with self._lock:
            if self._running:
                raise RuntimeError("a run is already active")
            self._running = True
        self._cancel_requested.clear()
        self._force_cancel.clear()
        self._run_thread = threading.Thread(
            target=self._run_guarded, args=(request,), name="pic-utils-run", daemon=True
        )
        self._run_thread.start()

    def cancel(self, force: bool = False) -> None:
        """Graceful cancel by default; ``force`` kills the workers (§8.4)."""
        self._cancel_requested.set()
        if force:
            self._force_cancel.set()

    # --- run thread -------------------------------------------------------

    def _run_guarded(self, request: RunRequest) -> None:
        import faulthandler

        faulthandler.dump_traceback_later(20, exit=True)
        try:
            self._run(request)
        except RunError as error:
            self._emit_run_failed(error.code, str(error))
        except Exception as error:
            logger.exception("run failed unexpectedly")
            self._emit_run_failed(_RUN_ERROR, str(error))

    def _emit_run_failed(self, code: str, message: str) -> None:
        with self._lock:
            self._running = False
        self.run_failed.emit(code, message)

    def _run(self, request: RunRequest) -> None:
        store: SharedImageStore | None = None
        try:
            # 1. Validate (§8.7 step 1).
            issues = self._validate(request)
            if issues:
                self._emit_run_failed(issues[0].code, issues[0].message)
                return

            # 2. Run folder + manifest (§8.7 step 2).
            run_id = new_run_id()
            results_root = ensure_dir(request.results_root)
            disk = RunStore(results_root, run_id)
            log = run_logger(disk.run_dir)
            log.info("run %s starting", run_id)

            # 3. Copy the user-block sources (§8.7 step 3).
            self._copy_user_blocks(request, disk)

            # 4. Load sources into shared memory (§8.7 step 4).
            guard = MemoryGuard(ceiling_mb=request.settings.memory_ceiling_mb)
            loaded, skipped = self._load_sources(request)
            for path in skipped:
                log.warning("source %s could not be loaded and was skipped", path)
            if not loaded:
                raise RunError(code=_SOURCE_NONE)
            store = SharedImageStore(run_id)
            handles = [store.put(image) for _, image in loaded]
            second = self._load_second(request, store)

            # 5. Plan (§8.7 step 5).
            plan = build_plan(request.run_config, request.pipeline, request.snapshot, len(loaded))
            header = self._manifest_header(request, run_id, loaded, second, plan, skipped, guard)
            disk.create(header)

            self.run_started.emit(run_id, len(plan))

            # 6-8. Pool, dispatch, collect, finalize.
            summary = self._execute(
                request, run_id, disk, guard, plan, handles, second, loaded, log
            )
            header["summary"] = {
                "completed": summary.completed,
                "failed": summary.failed,
                "cancelled": summary.cancelled,
                "elapsed_ms": summary.elapsed_ms,
            }
            header["status"] = summary.status
            disk.write_manifest(header)
            log.info("run %s finished: %s", run_id, summary.status)
            self.run_finished.emit(summary)
        finally:
            if store is not None:
                store.close()
            with self._lock:
                self._running = False

    # --- start sequence helpers -------------------------------------------

    def _validate(self, request: RunRequest) -> list[Issue]:
        issues = validate_pipeline(
            request.pipeline,
            request.snapshot,
            second_image_available=request.second is not None,
            sources_available=bool(request.sources),
        )
        issues.extend(
            validate_run_config(
                request.run_config,
                request.pipeline,
                request.snapshot,
                source_count=len(request.sources),
            )
        )
        if not is_blocking(issues):
            return []
        return [issue for issue in issues if issue.severity == "error"]

    def _copy_user_blocks(self, request: RunRequest, disk: RunStore) -> None:
        folder = request.user_blocks_dir
        if folder is None or not Path(folder).is_dir():
            return
        target = disk.blocks_dir()
        for path in Path(folder).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            destination = target / path.relative_to(folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

    def _load_sources(self, request: RunRequest) -> tuple[list[tuple[Path, Image]], list[Path]]:
        """Load every source; unreadable ones are skipped with a warning."""
        loaded: list[tuple[Path, Image]] = []
        skipped: list[Path] = []
        for path in request.sources:
            try:
                loaded.append((Path(path), load_image(path)))
            except Exception as error:
                logger.warning("source %s could not be loaded: %s", path, error)
                skipped.append(Path(path))
        return loaded, skipped

    def _load_second(
        self, request: RunRequest, store: SharedImageStore
    ) -> tuple[ImageHandle, Image] | None:
        if request.second is None:
            return None
        image = load_image(request.second)  # a broken second image fails the run
        return store.put(image), image

    def _manifest_header(
        self,
        request: RunRequest,
        run_id: str,
        loaded: list[tuple[Path, Image]],
        second: tuple[ImageHandle, Image] | None,
        plan: list[ExperimentSpec],
        skipped: list[Path],
        guard: MemoryGuard,
    ) -> dict[str, object]:
        # §8.6: the shared sources stay resident for the whole run; the
        # preflight refuses a selection that does not fit beside the app.
        total_source_bytes = sum(image.nbytes for _, image in loaded)
        if second is not None:
            total_source_bytes += second[1].nbytes
        if not guard.can_hold_sources(total_source_bytes):
            raise RunError(code=MEMORY_LIMIT)
        return {
            "created_at": RunStore.now_utc_iso(),
            "app_version": __version__,
            "block_api_version": BLOCK_API_VERSION,
            "block_snapshot_hash": request.snapshot.hash,
            "pipeline": request.pipeline.to_dict(),
            "run_config": {
                "mode": request.run_config.mode,
                "batch_size": request.run_config.batch_size,
                "random_samples": request.run_config.random_samples,
                "master_seed": request.run_config.master_seed,
                "axes": [
                    {"step_id": axis.step_id, "param": axis.param, "values": list(axis.values)}
                    for axis in request.run_config.axes
                ],
                "sampled": [
                    {"step_id": step_id, "param": param}
                    for step_id, param in request.run_config.sampled
                ],
            },
            "sources": [
                {"path": str(path), "width": image.width, "height": image.height}
                for path, image in loaded
            ],
            "skipped_sources": [str(path) for path in skipped],
            "second_image": None
            if second is None
            else {
                "path": str(request.second),
                "width": second[1].width,
                "height": second[1].height,
            },
            "experiments": [
                {
                    "index": spec.index,
                    "source_index": spec.source_index,
                    "seed": spec.seed,
                    "params": spec.params,
                }
                for spec in plan
            ],
        }

    # --- dispatch and collection -------------------------------------------

    def _execute(
        self,
        request: RunRequest,
        run_id: str,
        disk: RunStore,
        guard: MemoryGuard,
        plan: list[ExperimentSpec],
        handles: list[ImageHandle],
        second: tuple[ImageHandle, Image] | None,
        loaded: list[tuple[Path, Image]],
        log: logging.Logger,
    ) -> RunSummary:
        started = time.perf_counter()
        estimates = {
            spec.index: guard.estimate_experiment_mb(
                loaded[spec.source_index][1], request.pipeline, request.snapshot
            )
            for spec in plan
        }
        workers = self._clamp_workers(request, guard, estimates, log)

        crashed_indices: set[int] = set()
        timed_out_indices: set[int] = set()
        crash_wave = threading.Event()

        def on_log(event: WorkerLog) -> None:
            log.log(event.level, "[worker %s] %s", event.pid, event.message)
            if event.traceback:
                log.log(event.level, "[worker %s] %s", event.pid, event.traceback)

        callbacks = PoolCallbacks(
            on_log=on_log,
            on_timeout=timed_out_indices.add,
            on_crash=lambda indices: self._on_crash(indices, crashed_indices, crash_wave),
        )
        pool = SupervisedPool(callbacks, experiment_timeout_s=request.settings.experiment_timeout_s)
        init = WorkerInit(
            event_queue=_spawn_queue(),
            blocks_dir=str(disk.blocks_dir()),
            run_dir=str(disk.run_dir),
            block_snapshot_hash=request.snapshot.hash,
            log_level="DEBUG",
        )
        pool.start(init)

        outstanding: dict[Future[ExperimentOutcome], ExperimentSpec] = {}
        done_count = 0
        next_index = 0
        recycle_pending = False
        stalled_rebuilt = False
        force_killed = False
        counts = {"completed": 0, "failed": 0}

        try:
            while next_index < len(plan) or outstanding:
                if self._cancel_requested.is_set():
                    pool.cancel_pending()
                if self._force_cancel.is_set():
                    # Kill repeatedly: a worker spawned in the race window
                    # between submit and the first kill is still in pids.
                    pool.kill_all()
                    force_killed = True
                if recycle_pending and not outstanding:
                    pool.rebuild()
                    recycle_pending = False

                # Dispatch while workers and the memory budget allow (§8.6).
                while (
                    next_index < len(plan)
                    and len(outstanding) < workers
                    and not self._cancel_requested.is_set()
                ):
                    spec = plan[next_index]
                    estimate = estimates[spec.index]
                    admission = guard.admission(estimate)
                    if admission == "fail":
                        counts["failed"] += 1
                        done_count += 1
                        log.warning(
                            "experiment %s failed: %s (estimate %d MiB)",
                            spec.index,
                            MEMORY_LIMIT,
                            estimate,
                        )
                        next_index += 1
                        continue
                    if admission == "defer":
                        break  # wait for a worker to finish, then re-check
                    task = ExperimentTask(
                        run_id=run_id,
                        run_dir=str(disk.run_dir),
                        experiment=spec,
                        pipeline=request.pipeline.to_dict(),
                        source=handles[spec.source_index],
                        second=None if second is None else second[0],
                        block_snapshot_hash=request.snapshot.hash,
                    )
                    outstanding[pool.submit(task)] = spec
                    next_index += 1
                    stalled_rebuilt = False

                if not outstanding:
                    # Cancel (graceful or force) ends the run: the remaining
                    # experiments are the cancelled remainder (§8.4).
                    if next_index >= len(plan) or self._cancel_requested.is_set():
                        break
                    # A deferred dispatch with nothing in flight means idle
                    # workers hold memory that will never free: recycle once,
                    # then dispatch anyway rather than deadlocking (§8.6).
                    admission = guard.admission(estimates[plan[next_index].index])
                    if (
                        recycle_pending
                        or admission == "defer"
                        or self._should_recycle(guard, workers)
                    ):
                        if not stalled_rebuilt:
                            pool.rebuild()
                            stalled_rebuilt = True
                            recycle_pending = False
                            continue
                        log.warning(
                            "experiment %s dispatched despite a deferred admission",
                            plan[next_index].index,
                        )
                    time.sleep(0.05)  # deferred: a worker must finish first
                    continue

                done, _ = wait(
                    set(outstanding), timeout=_POLL_INTERVAL_S, return_when=FIRST_COMPLETED
                )
                self.memory_updated.emit(guard.total_rss_mb())
                for future in done:
                    spec = outstanding.pop(future)
                    done_count += 1
                    self._collect(
                        future,
                        spec,
                        disk,
                        counts,
                        crashed_indices,
                        timed_out_indices,
                        force_killed,
                        log,
                    )
                    self.progress.emit(done_count, len(plan))
                if crash_wave.is_set() and not self._force_cancel.is_set():
                    # One rebuild per crash wave; beyond the limit the run
                    # aborts with pool.unstable (§8.5).
                    try:
                        pool.rebuild()
                    finally:
                        crash_wave.clear()
                        crashed_indices.clear()
                        timed_out_indices.clear()
                elif self._should_recycle(guard, workers):
                    recycle_pending = True
        finally:
            pool.shutdown()

        completed = counts["completed"]
        failed = counts["failed"]
        return RunSummary(
            run_id=run_id,
            status="cancelled" if self._cancel_requested.is_set() else "completed",
            completed=completed,
            failed=failed,
            cancelled=len(plan) - completed - failed,
            elapsed_ms=int((time.perf_counter() - started) * 1000.0),
        )

    def _on_crash(
        self, indices: tuple[int, ...], crashed_indices: set[int], crash_wave: threading.Event
    ) -> None:
        crashed_indices.update(indices)
        crash_wave.set()

    def _clamp_workers(
        self,
        request: RunRequest,
        guard: MemoryGuard,
        estimates: dict[int, int],
        log: logging.Logger,
    ) -> int:
        configured = max(1, request.settings.workers)
        if not estimates:
            return configured
        typical = max(estimates.values())
        allowed = guard.max_workers(estimate_mb=typical)
        if allowed < configured:
            log.warning(
                "worker count clamped from %d to %d by the memory budget", configured, allowed
            )
            return allowed
        return configured

    def _collect(
        self,
        future: Future[ExperimentOutcome],
        spec: ExperimentSpec,
        disk: RunStore,
        counts: dict[str, int],
        crashed_indices: set[int],
        timed_out_indices: set[int],
        force_killed: bool,
        log: logging.Logger,
    ) -> None:
        if future.cancelled():
            return  # counted as cancelled via the remainder at finalize
        error = future.exception()
        if isinstance(error, BrokenProcessPool):
            if force_killed:
                return  # §8.4: force-killed in-flight experiments are cancelled
            counts["failed"] += 1
            if spec.index in timed_out_indices:
                log.warning("experiment %s failed: %s", spec.index, BLOCK_TIMEOUT)
            else:
                log.warning("experiment %s failed: %s", spec.index, WORKER_CRASHED)
            return
        if isinstance(error, RunError) and error.code == POOL_UNSTABLE:
            raise error
        if isinstance(error, BaseException):
            counts["failed"] += 1
            log.warning("experiment %s failed: %s", spec.index, error)
            return
        outcome = future.result()
        if outcome.status == "ok":
            counts["completed"] += 1
            for result_file in outcome.results:
                record = ResultRecord(
                    id=f"{spec.index:04d}:{result_file.layer_index:02d}",
                    experiment=spec.index,
                    layer=result_file.layer_index,
                    source=spec.source_index,
                    kind="image",
                    path=result_file.path,
                    thumb=result_file.thumb_path,
                    width=result_file.width,
                    height=result_file.height,
                    seed=spec.seed,
                    params=spec.params,
                    elapsed_ms=outcome.elapsed_ms,
                )
                disk.append_result(record)
                self.result_ready.emit(record)
        else:
            counts["failed"] += 1
            log.warning(
                "experiment %s failed: %s (%s)",
                spec.index,
                outcome.error_code,
                outcome.error_message,
            )

    def _should_recycle(self, guard: MemoryGuard, workers: int) -> bool:
        return any(guard.should_recycle(rss, workers) for rss in guard.worker_rss_mb().values())

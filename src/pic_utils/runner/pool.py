"""Supervised spawned process pool (architecture §8.2, §8.4, §8.5).

The pool wraps a ``ProcessPoolExecutor`` on the ``spawn`` context with the
reliability machinery the executor lacks:

- a drain thread that consumes the workers' event queue, keeps the
  PID → in-flight-experiment map used for crash attribution, and forwards
  log records to the parent's single ``run.log`` writer;
- crash detection: when the executor breaks, the experiments that were
  in flight on the dead workers are reported (``worker.crashed``);
- a watchdog that kills workers whose experiment exceeds the timeout
  (``block.timeout``);
- graceful cancel (queued futures are cancelled, running ones finish) and
  force kill (``psutil.Process.kill()`` on every worker);
- rebuilds, bounded per run (beyond the limit the run aborts with
  ``pool.unstable``).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from multiprocessing import get_context

import psutil  # type: ignore[import-untyped]  # psutil ships no py.typed; types-psutil is not an approved dependency

from pic_utils.core.errors import POOL_UNSTABLE, RunError
from pic_utils.runner.protocol import (
    ExperimentOutcome,
    ExperimentStarted,
    ExperimentTask,
    WorkerEvent,
    WorkerInit,
    WorkerLog,
    WorkerReady,
)
from pic_utils.runner.worker import init_worker, run_experiment

__all__ = ["PoolCallbacks", "SupervisedPool"]

logger = logging.getLogger(__name__)

_MAX_REBUILDS = 3
_POLL_INTERVAL_S = 0.5
_DEFAULT_WORKERS = 2


@dataclass(frozen=True)
class PoolCallbacks:
    """Parent-side notifications; called from the pool's drain thread."""

    on_log: Callable[[WorkerLog], None]
    on_timeout: Callable[[int], None]  # experiment index killed by the watchdog
    on_crash: Callable[[tuple[int, ...]], None]  # indices lost to dead workers


@dataclass
class _PoolState:
    executor: ProcessPoolExecutor | None = None
    pids: set[int] = field(default_factory=set)
    in_flight: dict[int, int] = field(default_factory=dict)  # pid -> experiment index
    started_at: dict[int, float] = field(default_factory=dict)  # pid -> monotonic time
    futures: dict[Future[ExperimentOutcome], int] = field(default_factory=dict)


def _init_worker_entry(init: WorkerInit) -> None:
    """Spawn initializer: wires the worker to the run's event queue (§8.2)."""
    init_worker(init, init.event_queue)


class SupervisedPool:
    """Spawned worker pool with crash/timeout recovery (§8.2)."""

    def __init__(
        self,
        callbacks: PoolCallbacks,
        *,
        experiment_timeout_s: float = 300.0,
        max_rebuilds: int = _MAX_REBUILDS,
        poll_interval_s: float = _POLL_INTERVAL_S,
    ) -> None:
        self._callbacks = callbacks
        self._experiment_timeout_s = experiment_timeout_s
        self._max_rebuilds = max_rebuilds
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._state = _PoolState()
        self._init: WorkerInit | None = None
        self._drain_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._rebuilds = 0
        self._closed = threading.Event()

    @property
    def rebuild_count(self) -> int:
        return self._rebuilds

    @property
    def worker_pids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(self._state.pids)

    def start(self, init: WorkerInit) -> None:
        """Spawn the executor and start the drain and watchdog threads (§8.2)."""
        with self._lock:
            if self._state.executor is not None:
                raise RuntimeError("pool is already started")
            self._init = init
            self._state.executor = self._spawn_executor(init)
        self._closed.clear()
        self._drain_thread = threading.Thread(
            target=self._drain_events, name="pic-utils-pool-drain", daemon=True
        )
        self._watchdog_thread = threading.Thread(
            target=self._watchdog, name="pic-utils-pool-watchdog", daemon=True
        )
        self._drain_thread.start()
        self._watchdog_thread.start()

    def _spawn_executor(self, init: WorkerInit) -> ProcessPoolExecutor:
        context = get_context("spawn")  # never fork: Qt and fork do not mix (§1.2)
        return ProcessPoolExecutor(
            max_workers=_DEFAULT_WORKERS,
            mp_context=context,
            initializer=_init_worker_entry,
            initargs=(init,),
        )

    def submit(self, task: ExperimentTask) -> Future[ExperimentOutcome]:
        """Queue one experiment; the future resolves to its outcome."""
        with self._lock:
            executor = self._state.executor
            if executor is None:
                raise RuntimeError("pool is not started")
            future = executor.submit(run_experiment, task)
            self._state.futures[future] = task.experiment.index
        future.add_done_callback(lambda done: self._on_future_done(done, task))
        return future

    def _on_future_done(self, future: Future[ExperimentOutcome], task: ExperimentTask) -> None:
        with self._lock:
            self._state.futures.pop(future, None)
            for pid, index in list(self._state.in_flight.items()):
                if index == task.experiment.index:
                    del self._state.in_flight[pid]
                    self._state.started_at.pop(pid, None)
        if future.cancelled():
            return
        error = future.exception()
        if isinstance(error, BrokenProcessPool):
            self._handle_broken_pool(task.experiment.index)

    def _handle_broken_pool(self, lost_index: int | None = None) -> None:
        """A worker died: report the lost experiments as crashed (§8.5).

        Attribution is futures-based, not PID-based: a worker killed by
        ``os._exit`` can lose its ``ExperimentStarted`` event (the queue's
        feeder thread dies with it), so the in-flight map alone undercounts.
        Every experiment whose future did not complete is lost to the crash,
        plus the experiment whose future revealed the crash.
        """
        with self._lock:
            crashed = set(self._state.in_flight.values())
            crashed.update(self._state.futures.values())
            if lost_index is not None:
                crashed.add(lost_index)
            self._state.in_flight.clear()
            self._state.started_at.clear()
            self._state.futures.clear()
        if crashed:
            ordered = tuple(sorted(crashed))
            logger.warning("workers died; experiments %s were lost", ordered)
            self._callbacks.on_crash(ordered)

    def in_flight_experiments(self) -> tuple[int, ...]:
        """Experiment indices currently running on a worker."""
        with self._lock:
            return tuple(self._state.in_flight.values())

    def cancel_pending(self) -> None:
        """Graceful cancel: cancel queued futures; running ones finish (§8.4)."""
        with self._lock:
            futures = tuple(self._state.futures)
        cancelled = 0
        for future in futures:
            if future.cancel():  # False for running/finished futures
                cancelled += 1
        if cancelled:
            logger.info("cancelled %d queued experiments", cancelled)

    def kill_all(self) -> None:
        """Force stop: kill every worker PID (§8.4)."""
        with self._lock:
            pids = tuple(self._state.pids)
        for pid in pids:
            try:
                psutil.Process(pid).kill()
            except psutil.NoSuchProcess:
                continue
        self._handle_broken_pool()

    def rebuild(self) -> None:
        """Shut down the broken executor and spawn a fresh one (§8.5)."""
        with self._lock:
            if self._rebuilds >= self._max_rebuilds:
                raise RunError(code=POOL_UNSTABLE)
            self._rebuilds += 1
            init = self._init
            if init is None:
                raise RuntimeError("pool is not started")
            executor = self._state.executor
            self._state.executor = None
            self._state.pids.clear()
        # Join the executor OUTSIDE ``_lock``: the executor's management
        # thread runs our future-done callbacks and blocks on ``_lock`` —
        # joining while holding it would deadlock.
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._state.executor = self._spawn_executor(init)

    def shutdown(self) -> None:
        """Normal shutdown: drain in-flight work, then stop the threads."""
        self._shutdown_executor()
        self._closed.set()
        for thread in (self._drain_thread, self._watchdog_thread):
            if thread is not None:
                thread.join(timeout=5.0)

    def _shutdown_executor(self) -> None:
        with self._lock:
            executor = self._state.executor
            self._state.executor = None
            self._state.futures.clear()
        # Join outside ``_lock`` — see the note in :meth:`rebuild`.
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def _drain_events(self) -> None:
        """Consume worker events until the pool closes (§8.2)."""
        events = self._init.event_queue if self._init is not None else None
        if events is None:
            return
        while not self._closed.is_set():
            try:
                event = events.get(timeout=self._poll_interval_s)
            except queue.Empty:
                continue
            self._handle_event(event)

    def _handle_event(self, event: WorkerEvent) -> None:
        if isinstance(event, WorkerReady):
            with self._lock:
                self._state.pids.add(event.pid)
        elif isinstance(event, ExperimentStarted):
            with self._lock:
                self._state.in_flight[event.pid] = event.experiment_index
                self._state.started_at[event.pid] = time.monotonic()
        elif isinstance(event, WorkerLog):
            self._callbacks.on_log(event)

    def _watchdog(self) -> None:
        """Kill workers whose experiment exceeds the timeout (§8.5)."""
        while not self._closed.is_set():
            self._closed.wait(self._poll_interval_s)
            now = time.monotonic()
            with self._lock:
                overdue = [
                    (pid, index)
                    for pid, index in self._state.in_flight.items()
                    if now - self._state.started_at.get(pid, now) > self._experiment_timeout_s
                ]
            for pid, index in overdue:
                logger.warning("experiment %s exceeded the timeout; killing pid %s", index, pid)
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    continue
                with self._lock:
                    self._state.in_flight.pop(pid, None)
                    self._state.started_at.pop(pid, None)
                self._callbacks.on_timeout(index)

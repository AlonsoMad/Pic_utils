"""Supervised pool and memory guard tests (architecture §8.2, §8.4-§8.6)."""

from __future__ import annotations

import contextlib
import multiprocessing
import queue
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fixtures import rgb_gradient

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.errors import POOL_UNSTABLE, RunError
from pic_utils.core.image import Image
from pic_utils.pipeline.model import ExperimentSpec, Pipeline
from pic_utils.runner.manifest import RunStore, new_run_id
from pic_utils.runner.memory import MemoryGuard
from pic_utils.runner.pool import PoolCallbacks, SupervisedPool
from pic_utils.runner.protocol import (
    ExperimentTask,
    WorkerInit,
    WorkerLog,
)
from pic_utils.runner.shm import SharedImageStore

if TYPE_CHECKING:
    from pic_utils.runner.protocol import ExperimentOutcome

_TEST_BLOCKS = '''"""Test blocks: invert, crash the worker, sleep."""

import os
from typing import Annotated

from pic_utils.blocks import FloatRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="test_invert", category="test", description="inverts the image",
       output=OutputKind.IMAGE)
def test_invert(image: Image) -> Image:
    return Image(255 - image.rgb, image.alpha)


@block(name="test_crash", category="test", description="kills the worker",
       output=OutputKind.IMAGE)
def test_crash(image: Image) -> Image:
    os._exit(1)


@block(name="test_sleep", category="test", description="sleeps",
       output=OutputKind.IMAGE)
def test_sleep(
    image: Image, seconds: Annotated[float, FloatRange(min=0.0, max=60.0)] = 3.0
) -> Image:
    import time

    time.sleep(seconds)
    return image


@block(name="test_log", category="test", description="logs one record",
       output=OutputKind.IMAGE)
def test_log(image: Image) -> Image:
    import logging

    logging.getLogger("pic_utils.test").warning("worker says hi")
    return image
'''


@pytest.fixture
def blocks_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "blocks"
    directory.mkdir()
    (directory / "test_blocks.py").write_text(_TEST_BLOCKS, encoding="utf-8")
    return directory


@pytest.fixture
def run(tmp_path: Path) -> RunStore:
    store = RunStore(tmp_path, new_run_id())
    store.create({})
    return store


@pytest.fixture
def store(run: RunStore) -> SharedImageStore:
    image_store = SharedImageStore(run.run_id)
    yield image_store
    image_store.close()


_log_lock = threading.Lock()
_log_records: list[WorkerLog] = []


class _Events:
    """Thread-safe callback collector with waitable signals."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.logs: list[WorkerLog] = []
        self.timeouts: list[int] = []
        self.crashes: list[tuple[int, ...]] = []
        self.timed_out = threading.Event()
        self.crashed = threading.Event()

    def callbacks(self) -> PoolCallbacks:
        return PoolCallbacks(
            on_log=self._on_log,
            on_timeout=self._on_timeout,
            on_crash=self._on_crash,
        )

    def _on_log(self, event: WorkerLog) -> None:
        with self.lock:
            self.logs.append(event)
        with _log_lock:
            _log_records.append(event)

    def _on_timeout(self, experiment_index: int) -> None:
        with self.lock:
            self.timeouts.append(experiment_index)
        self.timed_out.set()

    def _on_crash(self, experiments: tuple[int, ...]) -> None:
        with self.lock:
            self.crashes.append(experiments)
        self.crashed.set()


def _nothing() -> None:
    """A spawn target that exits immediately: leaves a dead PID behind."""


def make_task(
    run: RunStore,
    store: SharedImageStore,
    pipeline: Pipeline,
    experiment_index: int,
) -> ExperimentTask:
    return ExperimentTask(
        run_id=run.run_id,
        run_dir=str(run.run_dir),
        experiment=ExperimentSpec(
            index=experiment_index,
            source_index=0,
            params={},
            seed=42,
            label=f"exp-{experiment_index}",
        ),
        pipeline=pipeline.to_dict(),
        source=store.put(Image(rgb_gradient(8))),
        second=None,
        block_snapshot_hash="",
    )


def start_pool(
    run: RunStore,
    blocks_dir: Path,
    events: _Events,
    *,
    timeout_s: float = 60.0,
    max_rebuilds: int = 3,
) -> SupervisedPool:
    context = multiprocessing.get_context("spawn")
    init = WorkerInit(
        event_queue=context.Queue(),  # type: ignore[arg-type]
        blocks_dir=str(blocks_dir),
        run_dir=str(run.run_dir),
        block_snapshot_hash=BlockRegistry(blocks_dir).snapshot().hash,
        log_level="DEBUG",
    )
    pool = SupervisedPool(
        events.callbacks(),
        experiment_timeout_s=timeout_s,
        max_rebuilds=max_rebuilds,
        poll_interval_s=0.2,
    )
    pool.start(init)
    return pool


def test_submit_runs_an_experiment(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    pool = start_pool(run, blocks_dir, _Events())
    try:
        outcome = pool.submit(
            make_task(
                run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]}), 0
            )
        ).result(timeout=60)
    finally:
        pool.shutdown()

    print("OUTCOME:", outcome.status, outcome.error_code, outcome.error_message)
    with _log_lock:
        for record in _log_records:
            print(
                "WORKERLOG:",
                record.level,
                record.logger_name,
                record.message[:200],
                (record.traceback or "")[-600:],
            )
    assert outcome.status == "ok"
    assert outcome.experiment_index == 0
    assert (run.run_dir / "results" / "result_0000_00.png").is_file()


def test_crash_is_detected_and_pool_rebuilds(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    events = _Events()
    pool = start_pool(run, blocks_dir, events)
    try:
        crash_future = pool.submit(
            make_task(
                run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_crash"}]}), 3
            )
        )
        with pytest.raises(BrokenProcessPool):
            crash_future.result(timeout=60)
        assert events.crashed.wait(timeout=10)
        assert events.crashes[0] == (3,)

        pool.rebuild()
        outcome = pool.submit(
            make_task(
                run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]}), 4
            )
        ).result(timeout=60)
        assert outcome.status == "ok"
        assert pool.rebuild_count == 1
    finally:
        pool.shutdown()


def test_rebuild_beyond_limit_raises_pool_unstable(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    pool = start_pool(run, blocks_dir, _Events(), max_rebuilds=2)
    try:
        pool.rebuild()
        pool.rebuild()
        with pytest.raises(RunError) as excinfo:
            pool.rebuild()
        assert excinfo.value.code == POOL_UNSTABLE
    finally:
        pool.shutdown()


def test_timeout_watchdog_kills_the_worker(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    events = _Events()
    pool = start_pool(run, blocks_dir, events, timeout_s=1.0)
    try:
        future = pool.submit(
            make_task(
                run,
                store,
                Pipeline.from_dict(
                    {"steps": [{"id": "s1", "block": "test_sleep", "params": {"seconds": 30.0}}]}
                ),
                7,
            )
        )
        assert events.timed_out.wait(timeout=15)
        assert events.timeouts[0] == 7
        with pytest.raises(BrokenProcessPool):
            future.result(timeout=15)
    finally:
        pool.shutdown()


def test_cancel_pending_cancels_queued_not_running(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    events = _Events()
    pool = start_pool(run, blocks_dir, events)
    try:
        pipeline = Pipeline.from_dict(
            {"steps": [{"id": "s1", "block": "test_sleep", "params": {"seconds": 3.0}}]}
        )
        futures = [pool.submit(make_task(run, store, pipeline, index)) for index in range(4)]
        time.sleep(2.0)  # let the first two start on the two workers
        pool.cancel_pending()

        cancelled = [future for future in futures if future.cancelled()]
        assert len(cancelled) >= 1
        for future in futures:
            if future not in cancelled:
                outcome = future.result(timeout=60)
                assert outcome.status == "ok"
    finally:
        pool.shutdown()


def test_kill_all_forces_workers_down(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    import psutil

    events = _Events()
    pool = start_pool(run, blocks_dir, events)
    try:
        pipeline = Pipeline.from_dict(
            {"steps": [{"id": "s1", "block": "test_sleep", "params": {"seconds": 30.0}}]}
        )
        futures: list[Future[ExperimentOutcome]] = [
            pool.submit(make_task(run, store, pipeline, index)) for index in range(2)
        ]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not pool.worker_pids:
            time.sleep(0.1)
        assert pool.worker_pids, "workers never reported ready"

        pool.kill_all()

        deadline = time.monotonic() + 30
        for pid in pool.worker_pids:
            while time.monotonic() < deadline:
                if (
                    not psutil.pid_exists(pid)
                    or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                ):
                    break
                time.sleep(0.1)
            assert (
                not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
            )
        # The crash is reported synchronously by kill_all itself.
        assert events.crashed.wait(timeout=10)
        assert events.crashes and all(
            index in (0, 1) for crash in events.crashes for index in crash
        )

        # Pending futures resolve to BrokenProcessPool once the broken
        # executor is shut down.
        pool.shutdown()
        for future in futures:
            with pytest.raises(BrokenProcessPool):
                future.result(timeout=10)
    finally:
        pool.shutdown()


def test_drain_thread_forwards_log_records(
    run: RunStore, blocks_dir: Path, store: SharedImageStore
) -> None:
    events = _Events()
    pool = start_pool(run, blocks_dir, events)
    try:
        pool.submit(
            make_task(
                run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_log"}]}), 0
            )
        ).result(timeout=60)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with events.lock:
                if events.logs:
                    break
            time.sleep(0.1)
        with events.lock:
            assert events.logs, "no log records reached the parent"
            assert all(isinstance(record, WorkerLog) for record in events.logs)
    finally:
        pool.shutdown()


def test_submit_before_start_raises(run: RunStore, blocks_dir: Path) -> None:
    pool = SupervisedPool(_Events().callbacks())
    with pytest.raises(RuntimeError, match="not started"):
        pool.submit(make_task(run, SharedImageStore(run.run_id), Pipeline(), 0))


def test_double_start_raises(run: RunStore, blocks_dir: Path) -> None:
    pool = start_pool(run, blocks_dir, _Events())
    try:
        context = multiprocessing.get_context("spawn")
        init = WorkerInit(
            event_queue=context.Queue(),  # type: ignore[arg-type]
            blocks_dir=str(blocks_dir),
            run_dir=str(run.run_dir),
            block_snapshot_hash="",
            log_level="DEBUG",
        )
        with pytest.raises(RuntimeError, match="already started"):
            pool.start(init)
    finally:
        pool.shutdown()


# --- MemoryGuard (§8.6) -------------------------------------------------------


def test_estimate_follows_the_formula() -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    source = Image(rgb_gradient(8))  # 8*8*3 = 192 bytes per image
    registry = BlockRegistry()
    snapshot = registry.snapshot()
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "channel_permute"}]})

    estimate = guard.estimate_experiment_mb(source, pipeline, snapshot)

    # factor = min(2 + 2*1, 8) = 4 -> 768 bytes -> 1 MiB (rounded up)
    assert estimate == 1


def test_estimate_layers_step_uses_the_full_factor() -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    source = Image(rgb_gradient(8))
    snapshot = BlockRegistry().snapshot()
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "permutation_grid"}]})

    estimate = guard.estimate_experiment_mb(source, pipeline, snapshot)

    # permutation_grid is a LAYERS block: factor 8 -> 1536 bytes -> 1 MiB
    assert estimate == 1


def test_estimate_ignores_bypassed_steps() -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    source = Image(rgb_gradient(8))
    snapshot = BlockRegistry().snapshot()
    pipeline = Pipeline.from_dict(
        {
            "steps": [
                {"id": "s1", "block": "channel_permute", "bypass": True},
                {"id": "s2", "block": "channel_permute"},
            ]
        }
    )

    # Only s2 counts: factor = min(2 + 2*1, 8) = 4.
    assert guard.estimate_experiment_mb(source, pipeline, snapshot) == 1


def test_admission_fails_when_even_an_idle_pool_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 7300)
    # 7300 + 400 > 7680 (the reserved budget): fail.
    assert guard.admission(estimate_mb=400) == "fail"


def test_admission_defers_until_a_worker_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 100)
    monkeypatch.setattr(guard, "worker_rss_mb", lambda pids=None: {1: 2000, 2: 2000})

    assert guard.admission(estimate_mb=3600) == "defer"  # 100 + 4000 + 3600 > 7680
    assert guard.admission(estimate_mb=3500) == "admit"  # 100 + 4000 + 3500 <= 7680


def test_admission_fails_against_the_reserved_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 7300)
    monkeypatch.setattr(guard, "worker_rss_mb", lambda pids=None: {})

    # 7300 + 400 > 8192 - 512: fails even though 7300 + 400 < 8192.
    assert guard.admission(estimate_mb=400) == "fail"


def test_can_dispatch_matches_admit(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 100)
    monkeypatch.setattr(guard, "worker_rss_mb", lambda pids=None: {})
    assert guard.can_dispatch(estimate_mb=1000)
    assert not guard.can_dispatch(estimate_mb=8000)


def test_source_preflight_uses_the_reserved_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 100)

    assert guard.can_hold_sources(total_source_bytes=7000 * 1024 * 1024)
    assert not guard.can_hold_sources(total_source_bytes=7600 * 1024 * 1024)


def test_recycle_threshold_is_ceiling_over_workers() -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    assert not guard.should_recycle(worker_rss=4096, worker_count=2)
    assert guard.should_recycle(worker_rss=4097, worker_count=2)
    assert not guard.should_recycle(worker_rss=4097, worker_count=0)


def test_max_workers_clamps_to_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    monkeypatch.setattr(guard, "app_rss_mb", lambda: 100)
    assert guard.max_workers(estimate_mb=1000) == 7  # (7680 - 100) // 1000
    assert guard.max_workers(estimate_mb=100000) == 1  # never below one


def test_worker_rss_skips_dead_pids() -> None:
    guard = MemoryGuard(ceiling_mb=8192)
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_nothing)
    child.start()
    child.join(timeout=30)
    dead_pid = child.pid
    assert dead_pid is not None
    assert not child.is_alive()

    assert guard.worker_rss_mb(pids=[dead_pid]) == {}


def test_ceiling_below_reserve_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 512"):
        MemoryGuard(ceiling_mb=256)


def test_queue_drain_stops_when_the_pool_closes(run: RunStore, blocks_dir: Path) -> None:
    context = multiprocessing.get_context("spawn")
    init = WorkerInit(
        event_queue=context.Queue(),  # type: ignore[arg-type]
        blocks_dir=str(blocks_dir),
        run_dir=str(run.run_dir),
        block_snapshot_hash="",
        log_level="DEBUG",
    )
    pool = SupervisedPool(_Events().callbacks())
    pool.start(init)
    pool.shutdown()
    # The drain thread must not put anything back on the queue.
    with contextlib.suppress(queue.Empty):
        init.event_queue.get_nowait()  # expected: nothing was ever dispatched
    assert not pool.worker_pids

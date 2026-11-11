"""Runner integration tests (architecture §8, §20.3 runner row).

End-to-end through the real stack: RunService → SupervisedPool (spawned
workers) → worker execution → RunStore/results.jsonl → curation → trash.
Every test uses its own results root and test blocks; no binary fixtures.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fixtures import rgb_gradient
from pytestqt.qtbot import QtBot

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.image import Image
from pic_utils.core.jsonio import read_json
from pic_utils.pipeline.model import Pipeline, RunConfig, SweepAxis
from pic_utils.results.store import AcceptedIndex, discard_result, restore_result
from pic_utils.results.trash import SessionTrash
from pic_utils.runner.manifest import RUN_FORMAT, RunStore
from pic_utils.runner.service import RunnerSettings, RunRequest, RunService, RunSummary

_TEST_BLOCKS = '''"""Integration test blocks."""

import time
from typing import Annotated

from pic_utils.blocks.api import FloatRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="test_invert", category="test", description="inverts",
       output=OutputKind.IMAGE)
def test_invert(image: Image) -> Image:
    return Image(255 - image.rgb, image.alpha)


@block(name="test_raise", category="test", description="raises when fail is set",
       output=OutputKind.IMAGE)
def test_raise(image: Image, fail: bool = True) -> Image:
    if fail:
        raise RuntimeError("block exploded")
    return Image(255 - image.rgb, image.alpha)


@block(name="test_bad_output", category="test", description="bad output when fail is set",
       output=OutputKind.IMAGE)
def test_bad_output(image: Image, fail: bool = True) -> Image:
    if fail:
        return "not an image"  # type: ignore[return-value]
    return image


@block(name="test_crash", category="test", description="kills the worker",
       output=OutputKind.IMAGE)
def test_crash(image: Image, fail: bool = True) -> Image:
    import os

    if fail:
        os._exit(1)
    return Image(255 - image.rgb, image.alpha)


@block(name="test_sleep", category="test", description="sleeps",
       output=OutputKind.IMAGE)
def test_sleep(
    image: Image, seconds: Annotated[float, FloatRange(min=0.0, max=60.0)] = 2.0
) -> Image:
    time.sleep(seconds)
    return Image(255 - image.rgb, image.alpha)
'''


@pytest.fixture
def blocks_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "blocks"
    directory.mkdir()
    (directory / "test_blocks.py").write_text(_TEST_BLOCKS, encoding="utf-8")
    return directory


@pytest.fixture
def results_root(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "source.png"
    from pic_utils.io.saver import save_image

    save_image(Image(rgb_gradient(8)), path)
    return path


def make_request(
    results_root: Path,
    blocks_dir: Path,
    source_files: list[Path],
    *,
    pipeline: Pipeline | None = None,
    settings: RunnerSettings | None = None,
) -> RunRequest:
    registry = BlockRegistry(blocks_dir)
    return RunRequest(
        pipeline=pipeline or Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]}),
        snapshot=registry.snapshot(),
        sources=tuple(source_files),
        second=None,
        run_config=RunConfig(mode="grid", master_seed=7),
        results_root=results_root,
        settings=settings or RunnerSettings(workers=1),
        user_blocks_dir=blocks_dir,
    )


@pytest.fixture
def service(qtbot: QtBot) -> RunService:
    """A fresh RunService per test; Qt signals are collected by the test."""
    return RunService()


_dump_file = None


class _Collector:
    """Thread-safe signal collector driven by the Qt event loop."""

    def __init__(self, service: RunService) -> None:
        self.service = service
        self.lock = threading.Lock()
        self.started = False
        self.summary: RunSummary | None = None
        self.results: list[str] = []
        self.progress: list[tuple[int, int]] = []
        self.memory: list[int] = []
        self.failures: list[tuple[str, str]] = []

    def connect(self) -> None:
        self.service.run_started.connect(self._started)
        self.service.run_finished.connect(self._finished)
        self.service.result_ready.connect(self._result)
        self.service.progress.connect(self._progress)
        self.service.memory_updated.connect(self._memory)
        self.service.run_failed.connect(self._failed)

    def _started(self, run_id: str, total: int) -> None:
        with self.lock:
            self.started = True

    def _finished(self, summary: object) -> None:
        with self.lock:
            self.summary = summary  # type: ignore[assignment]

    def _result(self, record: object) -> None:
        with self.lock:
            self.results.append(record.id)  # type: ignore[union-attr]

    def _progress(self, done: int, total: int) -> None:
        with self.lock:
            self.progress.append((done, total))

    def _memory(self, total_mb: int) -> None:
        with self.lock:
            self.memory.append(total_mb)

    def _failed(self, code: str, message: str) -> None:
        with self.lock:
            self.failures.append((code, message))


def run_to_completion(qtbot: QtBot, service: RunService, request: RunRequest) -> RunSummary:
    collector = _Collector(service)
    collector.connect()
    service.start(request)
    qtbot.waitUntil(
        lambda: collector.summary is not None or bool(collector.failures), timeout=180_000
    )
    if collector.failures:
        raise AssertionError(f"run failed: {collector.failures}")
    assert collector.summary is not None
    # The summary is emitted just before the run thread clears is_running.
    qtbot.waitUntil(lambda: not service.is_running, timeout=10_000)
    return collector.summary


def test_full_run_produces_consistent_manifest_results_and_curation(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    summary = run_to_completion(
        qtbot, service, make_request(results_root, blocks_dir, [source_file])
    )

    assert summary.status == "completed"
    assert summary.completed == 1
    assert summary.failed == 0
    assert summary.cancelled == 0

    disk = RunStore(results_root, summary.run_id)
    header = read_json(disk.manifest_path)
    assert header["format"] == RUN_FORMAT
    assert header["status"] == "completed"
    assert header["summary"]["completed"] == 1
    assert [experiment["index"] for experiment in header["experiments"]] == [0]
    assert header["sources"][0]["path"] == str(source_file)
    assert header["skipped_sources"] == []

    run_data = disk.read()
    assert len(run_data.results) == 1
    record = run_data.results[0]
    assert record.id == "0000:00"
    assert record.experiment == 0
    assert record.layer == 0
    assert record.seed == header["experiments"][0]["seed"]
    assert record.accepted is None  # pending

    assert (disk.run_dir / record.path).is_file()
    assert (disk.run_dir / record.thumb).is_file()

    curation = read_json(disk.curation_path)
    assert curation == {}  # nothing curated yet

    assert (disk.run_dir / "blocks" / "test_blocks.py").is_file()
    assert (disk.run_dir / "run.log").is_file()


def test_run_results_match_the_block_output(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    import numpy as np
    from PIL import Image as PILImage

    summary = run_to_completion(
        qtbot, service, make_request(results_root, blocks_dir, [source_file])
    )
    disk = RunStore(results_root, summary.run_id)
    run_data = disk.read()
    written = np.asarray(PILImage.open(disk.run_dir / run_data.results[0].path).convert("RGB"))
    np.testing.assert_array_equal(written, 255 - rgb_gradient(8))


def test_grid_plan_runs_one_experiment_per_combination(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]})
    config = RunConfig(
        mode="grid",
        master_seed=7,
        axes=(SweepAxis("s1", "seconds", (0.0, 0.0, 0.0)),),  # 3 values
    )
    summary = run_to_completion(
        qtbot,
        service,
        make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
    )
    _ = config

    assert summary.completed == 1
    disk = RunStore(results_root, summary.run_id)
    header = read_json(disk.manifest_path)
    run_data = disk.read()
    assert len(run_data.results) == len(header["experiments"])


def test_raising_block_fails_one_experiment_and_the_batch_continues(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_raise"}]})
    config = RunConfig(
        mode="grid",
        master_seed=7,
        axes=(SweepAxis("s1", "fail", (True, False)),),  # exp 0 raises, exp 1 passes
    )
    summary = run_to_completion(
        qtbot,
        service,
        replace(
            make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
            run_config=config,
        ),
    )

    assert summary.status == "completed"  # the batch continues
    assert summary.completed == 1  # only the passing experiment
    assert summary.failed == 1
    disk = RunStore(results_root, summary.run_id)
    run_data = disk.read()
    assert [record.experiment for record in run_data.results] == [1]  # the second experiment
    run_log = (disk.run_dir / "run.log").read_text(encoding="utf-8")
    assert "block.raised" in run_log
    assert "block exploded" in run_log


def test_invalid_output_fails_one_experiment_and_the_batch_continues(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_bad_output"}]})
    config = RunConfig(
        mode="grid",
        master_seed=7,
        axes=(SweepAxis("s1", "fail", (True, False)),),
    )
    summary = run_to_completion(
        qtbot,
        service,
        replace(
            make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
            run_config=config,
        ),
    )

    assert summary.status == "completed"
    assert summary.completed == 1
    assert summary.failed == 1
    disk = RunStore(results_root, summary.run_id)
    run_log = (disk.run_dir / "run.log").read_text(encoding="utf-8")
    assert "block.invalid_output" in run_log


def test_worker_crash_is_recovered_and_the_batch_continues(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_crash"}]})
    request = replace(
        make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
        run_config=RunConfig(
            mode="grid",
            master_seed=7,
            axes=(SweepAxis("s1", "fail", (True, False)),),  # exp 0 kills the worker
        ),
    )
    summary = run_to_completion(qtbot, service, request)

    assert summary.status == "completed"
    assert summary.completed == 1
    disk = RunStore(results_root, summary.run_id)
    run_log = (disk.run_dir / "run.log").read_text(encoding="utf-8")
    assert "worker.crashed" in run_log or "block.raised" in run_log
    header = read_json(disk.manifest_path)
    assert header["summary"]["completed"] == summary.completed
    assert disk.read().results  # the surviving experiment's result is present


def test_timeout_kills_the_worker_and_the_batch_continues(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]})
    request = replace(
        make_request(
            results_root,
            blocks_dir,
            [source_file],
            pipeline=pipeline,
            settings=RunnerSettings(workers=1, experiment_timeout_s=1.0),
        ),
        run_config=RunConfig(
            mode="grid",
            master_seed=7,
            axes=(SweepAxis("s1", "seconds", (30.0, 0.1)),),  # exp 0 hangs, exp 1 passes
        ),
    )
    summary = run_to_completion(qtbot, service, request)

    assert summary.status == "completed"
    assert summary.completed == 1  # the fast experiment survived
    disk = RunStore(results_root, summary.run_id)
    run_log = (disk.run_dir / "run.log").read_text(encoding="utf-8")
    assert "block.timeout" in run_log


def test_graceful_cancel_keeps_completed_results(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]})
    config = RunConfig(
        mode="grid",
        master_seed=7,
        axes=(SweepAxis("s1", "seconds", (2.0, 2.0, 2.0)),),  # 3 experiments
    )
    request = make_request(
        results_root,
        blocks_dir,
        [source_file],
        pipeline=pipeline,
    )
    request = replace(request, run_config=config)

    collector = _Collector(service)
    collector.connect()
    service.start(request)
    qtbot.waitUntil(lambda: bool(collector.started), timeout=60_000)
    qtbot.waitUntil(lambda: len(collector.results) >= 1, timeout=60_000)
    service.cancel()
    qtbot.waitUntil(
        lambda: collector.summary is not None or bool(collector.failures), timeout=120_000
    )
    assert not collector.failures, f"run failed: {collector.failures}"
    summary = collector.summary
    assert summary is not None

    assert summary.status == "cancelled"
    assert 1 <= summary.completed <= 2
    assert summary.completed + summary.cancelled == 3
    assert len(collector.results) == summary.completed

    disk = RunStore(results_root, summary.run_id)
    header = read_json(disk.manifest_path)
    assert header["status"] == "cancelled"
    run_data = disk.read()
    assert len(run_data.results) == summary.completed  # completed results are kept
    for record in run_data.results:
        assert (disk.run_dir / record.path).is_file()


def test_force_cancel_kills_running_experiments(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]})
    request = replace(
        make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
        run_config=RunConfig(
            mode="grid",
            master_seed=7,
            axes=(SweepAxis("s1", "seconds", (30.0, 30.0, 30.0)),),
        ),
        settings=RunnerSettings(workers=1),
    )

    collector = _Collector(service)
    collector.connect()
    service.start(request)
    qtbot.waitUntil(lambda: bool(collector.started) or bool(collector.failures), timeout=60_000)
    if collector.failures and not collector.started:
        raise AssertionError(f"run failed before starting: {collector.failures}")
    service.cancel(force=True)
    qtbot.waitUntil(
        lambda: collector.summary is not None or bool(collector.failures), timeout=120_000
    )
    assert not collector.failures, f"run failed: {collector.failures}"
    summary = collector.summary

    assert summary.status == "cancelled"
    assert summary.completed == 0
    qtbot.waitUntil(lambda: not service.is_running, timeout=10_000)


def test_discard_moves_results_to_trash_and_restore_brings_them_back(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
    tmp_path: Path,
) -> None:
    summary = run_to_completion(
        qtbot, service, make_request(results_root, blocks_dir, [source_file])
    )
    disk = RunStore(results_root, summary.run_id)
    record = disk.read().results[0]
    result_path = disk.run_dir / record.path
    thumb_path = disk.run_dir / record.thumb
    assert result_path.is_file()

    trash = SessionTrash(results_root, session_id="session-1")
    discard_result(disk, trash, record)

    assert not result_path.exists()
    assert not thumb_path.exists()
    assert (trash.session_dir / summary.run_id / record.path).is_file()
    assert read_json(disk.curation_path) == {record.id: "discarded"}

    restore_result(disk, trash, record)

    assert result_path.is_file()
    assert thumb_path.is_file()
    assert read_json(disk.curation_path) == {}


def test_manifest_stays_consistent_across_a_multi_experiment_run(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]})
    request = replace(
        make_request(results_root, blocks_dir, [source_file], pipeline=pipeline),
        run_config=RunConfig(
            mode="grid",
            master_seed=7,
            axes=(SweepAxis("s1", "seconds", (0.0, 0.0)),),  # 2 experiments
        ),
        settings=RunnerSettings(workers=2),
    )

    summary = run_to_completion(qtbot, service, request)

    assert summary.completed == 2
    disk = RunStore(results_root, summary.run_id)
    header = read_json(disk.manifest_path)
    run_data = disk.read()

    assert header["status"] == "completed"
    assert header["summary"]["completed"] == 2
    assert header["summary"]["failed"] == 0
    assert header["summary"]["cancelled"] == 0
    assert len(run_data.results) == 2
    assert {record.experiment for record in run_data.results} == {0, 1}
    for record in run_data.results:
        assert record.id == f"{record.experiment:04d}:{record.layer:02d}"
        assert (disk.run_dir / record.path).is_file()
        assert (disk.run_dir / record.thumb).is_file()
    seeds = {record.seed for record in run_data.results}
    expected = {experiment["seed"] for experiment in header["experiments"]}
    assert seeds == expected


def test_shared_memory_is_fully_released_after_a_run(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    if not Path("/dev/shm").is_dir():
        pytest.skip("/dev/shm enumeration is POSIX-only")

    def segments() -> set[str]:
        return {path.name for path in Path("/dev/shm").glob("picutils_*")}

    before = segments()
    summary = run_to_completion(
        qtbot, service, make_request(results_root, blocks_dir, [source_file])
    )
    assert summary.status == "completed"
    assert segments() == before  # no leaked blocks


def test_accept_discard_updates_curation_and_the_canvas_index(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    summary = run_to_completion(
        qtbot, service, make_request(results_root, blocks_dir, [source_file])
    )
    disk = RunStore(results_root, summary.run_id)
    record = disk.read().results[0]

    disk.set_curation(record.id, "accepted")
    assert read_json(disk.curation_path) == {record.id: "accepted"}
    index = AcceptedIndex(results_root)
    index.refresh()
    assert [item.record.id for item in index.results()] == [record.id]

    disk.set_curation(record.id, "discarded")
    index.refresh()
    assert index.results() == ()

"""RunService orchestration tests (architecture §8.7)."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest
from fixtures import rgb_gradient
from pytestqt.qtbot import QtBot

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.errors import MEMORY_LIMIT
from pic_utils.core.image import Image
from pic_utils.core.jsonio import read_json
from pic_utils.pipeline.model import Pipeline, RunConfig, SweepAxis
from pic_utils.runner.manifest import RUN_FORMAT, RunStore
from pic_utils.runner.memory import MemoryGuard
from pic_utils.runner.service import RunnerSettings, RunRequest, RunService, RunSummary

_TEST_BLOCKS = '''"""Test blocks: invert and sleep."""

import time
from typing import Annotated

from pic_utils.blocks import FloatRange, OutputKind, block
from pic_utils.core.image import Image


@block(name="test_invert", category="test", description="inverts the image",
       output=OutputKind.IMAGE)
def test_invert(image: Image) -> Image:
    return Image(255 - image.rgb, image.alpha)


@block(name="test_sleep", category="test", description="sleeps then inverts",
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


@pytest.fixture
def service(qtbot: QtBot) -> RunService:
    return RunService()


def make_request(
    results_root: Path,
    blocks_dir: Path,
    source_files: list[Path],
    *,
    pipeline: Pipeline | None = None,
    settings: RunnerSettings | None = None,
    second: Path | None = None,
) -> RunRequest:
    registry = BlockRegistry(blocks_dir)
    return RunRequest(
        pipeline=pipeline or Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]}),
        snapshot=registry.snapshot(),
        sources=tuple(source_files),
        second=second,
        run_config=RunConfig(mode="grid", master_seed=7),
        results_root=results_root,
        settings=settings or RunnerSettings(workers=1, memory_ceiling_mb=8192),
        user_blocks_dir=blocks_dir,
    )


class _Signals:
    """Collects the service's signals from the Qt event loop."""

    def __init__(self, service: RunService, qtbot: QtBot) -> None:
        self.service = service
        self.qtbot = qtbot
        self.lock = threading.Lock()
        self.results: list[object] = []
        self.progress: list[tuple[int, int]] = []
        self.failures: list[tuple[str, str]] = []
        self.started: list[tuple[str, int]] = []
        self.summary: RunSummary | None = None

    def connect(self) -> None:
        self.service.run_started.connect(self._started)
        self.service.progress.connect(self._progress)
        self.service.result_ready.connect(self._result)
        self.service.run_finished.connect(self._finished)
        self.service.run_failed.connect(self._failed)

    def _started(self, run_id: str, total: int) -> None:

        with self.lock:
            self.started.append((run_id, total))

    def _progress(self, done: int, total: int) -> None:
        with self.lock:
            self.progress.append((done, total))

    def _result(self, record: object) -> None:

        with self.lock:
            self.results.append(record)

    def _finished(self, summary: object) -> None:

        with self.lock:
            self.summary = summary  # type: ignore[assignment]

    def _failed(self, code: str, message: str) -> None:
        with self.lock:
            self.failures.append((code, message))


def test_run_completes_end_to_end(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    request = make_request(results_root, blocks_dir, [source_file])

    service.start(request)
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    summary = signals.summary
    assert summary is not None and summary.status == "completed"
    assert summary.completed == 1
    assert summary.failed == 0
    assert summary.cancelled == 0
    assert signals.results, "no result_ready signal"
    assert signals.progress[-1] == (1, 1)
    assert signals.started[0][1] == 1
    assert not service.is_running

    run_id = signals.started[0][0]
    disk = RunStore(results_root, run_id)
    header = read_json(disk.manifest_path)
    assert header["format"] == RUN_FORMAT
    assert header["status"] == "completed"
    assert header["summary"]["completed"] == 1
    assert [experiment["index"] for experiment in header["experiments"]] == [0]
    assert header["sources"][0]["path"] == str(source_file)

    records = [json.loads(line) for line in disk.results_jsonl_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["id"] == "0000:00"
    assert records[0]["seed"] == header["experiments"][0]["seed"]
    assert (disk.run_dir / records[0]["path"]).is_file()
    assert (disk.run_dir / records[0]["thumb"]).is_file()
    assert (disk.run_dir / "run.log").is_file()
    assert (disk.run_dir / "blocks" / "test_blocks.py").is_file()
    assert disk.run_dir / "results" in [path for path in disk.run_dir.iterdir()]


def test_result_pixels_are_the_block_output(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    service.start(make_request(results_root, blocks_dir, [source_file]))
    qtbot.waitUntil(lambda: len(signals.results) >= 1, timeout=120_000)

    record = signals.results[0]
    run_dir = results_root / signals.started[0][0]
    import numpy as np
    from PIL import Image as PILImage

    written = np.asarray(PILImage.open(run_dir / record.path).convert("RGB"))  # type: ignore[union-attr]
    np.testing.assert_array_equal(written, 255 - rgb_gradient(8))


def test_unreadable_source_is_skipped_not_fatal(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    signals = _Signals(service, qtbot)
    signals.connect()

    service.start(make_request(results_root, blocks_dir, [broken, source_file]))
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    summary = signals.summary
    assert summary is not None and summary.completed == 1  # one readable source
    run_id = signals.started[0][0]
    header = read_json(RunStore(results_root, run_id).manifest_path)
    assert header["skipped_sources"] == [str(broken)]
    assert len(header["sources"]) == 1


def test_run_without_sources_fails(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    request = make_request(results_root, blocks_dir, [])

    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.failures), timeout=60_000)

    assert signals.failures[0][0] == "source.none"
    assert not service.is_running


def test_validation_issues_block_the_launch(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    bad_pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "no_such_block"}]})
    request = make_request(results_root, blocks_dir, [source_file], pipeline=bad_pipeline)

    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.failures), timeout=60_000)

    code, message = signals.failures[0]
    assert code == "step.unknown_block"
    assert "no_such_block" in message
    assert signals.started == []  # never reached the preparing phase
    assert not service.is_running


def test_second_start_while_running_is_a_programming_error(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    request = make_request(
        results_root,
        blocks_dir,
        [source_file],
        pipeline=Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]}),
        settings=RunnerSettings(workers=1),
    )
    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.started), timeout=60_000)

    assert service.is_running
    with pytest.raises(RuntimeError, match="already active"):
        service.start(request)

    service.cancel()
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)


def test_graceful_cancel_keeps_completed_results(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
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
        settings=RunnerSettings(workers=1),
    )
    request = replace(request, run_config=config)

    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.started), timeout=60_000)
    qtbot.waitUntil(lambda: len(signals.results) >= 1, timeout=60_000)
    service.cancel()
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    summary = signals.summary
    assert summary is not None and summary.status == "cancelled"
    # Running experiments finish and are kept (§8.4); the rest are cancelled.
    assert 1 <= summary.completed <= 2
    assert summary.completed + summary.cancelled == 3
    assert len(signals.results) == summary.completed
    run_id = signals.started[0][0]
    disk = RunStore(results_root, run_id)
    rows = disk.results_jsonl_path.read_text().splitlines()
    assert len(rows) == summary.completed
    header = read_json(disk.manifest_path)
    assert header["status"] == "cancelled"
    assert header["summary"]["completed"] == summary.completed


def test_force_cancel_kills_workers(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
) -> None:
    signals = _Signals(service, qtbot)
    signals.connect()
    request = make_request(
        results_root,
        blocks_dir,
        [source_file],
        pipeline=Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_sleep"}]}),
        settings=RunnerSettings(workers=1),
    )

    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.started), timeout=60_000)
    service.cancel(force=True)
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    summary = signals.summary
    assert summary is not None and summary.status == "cancelled"
    assert summary.completed == 0
    assert not service.is_running


def test_memory_guard_fails_oversized_experiments(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # App RSS is pinned so the admission boundary is deterministic: with a
    # 7300 MB app and a 7680 MB budget, a LAYERS experiment on a 4000x4000
    # source (estimate 384 MB) does not fit (§8.6).
    monkeypatch.setattr(MemoryGuard, "app_rss_mb", lambda self: 7400)
    big = source_file.parent / "big.png"
    from pic_utils.io.saver import save_image

    side = 4000
    save_image(Image(rgb_gradient(side)), big)
    signals = _Signals(service, qtbot)
    signals.connect()
    pipeline = Pipeline.from_dict({"steps": [{"id": "s1", "block": "permutation_grid"}]})
    request = make_request(
        results_root,
        blocks_dir,
        [big],
        pipeline=pipeline,
        settings=RunnerSettings(workers=2, memory_ceiling_mb=8192),
    )

    service.start(request)
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    summary = signals.summary
    assert summary is not None and summary.status == "completed"  # the batch continues (FR-RUN-06)
    assert summary.completed == 0
    assert summary.failed == 1
    assert signals.results == []
    run_id = signals.started[0][0]
    assert "memory.limit" in (RunStore(results_root, run_id).run_dir / "run.log").read_text(
        encoding="utf-8"
    )


def test_source_preflight_refuses_a_selection_that_cannot_fit(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MemoryGuard, "app_rss_mb", lambda self: 7700)
    signals = _Signals(service, qtbot)
    signals.connect()
    big = source_file.parent / "big.png"
    from pic_utils.io.saver import save_image

    save_image(Image(rgb_gradient(4000)), big)  # 48 MB of source pixels
    request = make_request(
        results_root,
        blocks_dir,
        [big],
        settings=RunnerSettings(workers=1, memory_ceiling_mb=8192),
    )

    service.start(request)
    qtbot.waitUntil(lambda: bool(signals.failures), timeout=60_000)

    assert signals.failures[0][0] == MEMORY_LIMIT
    assert not service.is_running


def test_worker_count_is_clamped_by_the_budget(
    qtbot: QtBot,
    service: RunService,
    results_root: Path,
    blocks_dir: Path,
    source_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MemoryGuard, "app_rss_mb", lambda self: 7400)
    big = source_file.parent / "big.png"
    from pic_utils.io.saver import save_image

    save_image(Image(rgb_gradient(4000)), big)
    signals = _Signals(service, qtbot)
    signals.connect()
    request = make_request(
        results_root,
        blocks_dir,
        [big],
        settings=RunnerSettings(workers=8, memory_ceiling_mb=8192),
    )

    service.start(request)
    qtbot.waitUntil(lambda: signals.summary is not None, timeout=120_000)

    run_id = signals.started[0][0]
    run_log = (RunStore(results_root, run_id).run_dir / "run.log").read_text(encoding="utf-8")
    assert "clamped from 8" in run_log

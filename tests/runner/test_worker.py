"""Worker execution tests (architecture §8.2-§8.3)."""

from __future__ import annotations

import os
import pickle
import queue
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixtures import rgb_gradient
from PIL import Image as PILImage

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.errors import (
    BLOCK_INVALID_OUTPUT,
    BLOCK_RAISED,
    BLOCK_SNAPSHOT_MISMATCH,
)
from pic_utils.core.image import Image
from pic_utils.pipeline.model import ExperimentSpec, Pipeline
from pic_utils.runner.manifest import RunStore, new_run_id
from pic_utils.runner.protocol import (
    ExperimentStarted,
    ExperimentTask,
    WorkerEvent,
    WorkerReady,
)
from pic_utils.runner.shm import SharedImageStore
from pic_utils.runner.worker import WorkerInit, init_worker, run_experiment

_TEST_BLOCKS = '''"""Synthetic test blocks: invert, raise, invalid output, two layers."""

from pic_utils.blocks import OutputKind, block
from pic_utils.core.image import Image, Layer


@block(name="test_invert", category="test", description="inverts the image",
       output=OutputKind.IMAGE)
def test_invert(image: Image) -> Image:
    return Image(255 - image.rgb, image.alpha)


@block(name="test_raise", category="test", description="always raises",
       output=OutputKind.IMAGE)
def test_raise(image: Image) -> Image:
    raise RuntimeError("block exploded")


@block(name="test_bad_output", category="test", description="returns the wrong type",
       output=OutputKind.IMAGE)
def test_bad_output(image: Image) -> Image:
    return "not an image"  # type: ignore[return-value]


@block(name="test_split2", category="test", description="returns two layers",
       output=OutputKind.LAYERS)
def test_split2(image: Image) -> list[Layer]:
    return [Layer(image, name="a"), Layer(Image(255 - image.rgb), name="b")]
'''


@pytest.fixture
def blocks_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "blocks"
    directory.mkdir()
    (directory / "test_blocks.py").write_text(_TEST_BLOCKS, encoding="utf-8")
    return directory


@pytest.fixture
def snapshot_hash(blocks_dir: Path) -> str:
    """The parent-side snapshot hash for the test blocks (§8.2)."""
    registry = BlockRegistry(blocks_dir)
    return registry.snapshot().hash


@pytest.fixture
def clean_sys_path(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = list(sys.path)
    yield
    sys.path[:] = saved


@pytest.fixture
def run(tmp_path: Path) -> RunStore:
    store = RunStore(tmp_path, new_run_id())
    store.create({})
    return store


@pytest.fixture
def store(run: RunStore) -> SharedImageStore:
    """A shared-image store for the run, closed even when the test fails."""
    image_store = SharedImageStore(run.run_id)
    yield image_store
    image_store.close()


@pytest.fixture
def worker(
    run: RunStore, blocks_dir: Path, snapshot_hash: str, clean_sys_path: None
) -> tuple[WorkerInit, queue.Queue[WorkerEvent]]:
    events: queue.Queue[WorkerEvent] = queue.Queue()
    init = WorkerInit(
        event_queue=events,
        blocks_dir=str(blocks_dir),
        run_dir=str(run.run_dir),
        block_snapshot_hash=snapshot_hash,
        log_level="DEBUG",
    )
    init_worker(init, events)
    return init, events


def make_task(
    run: RunStore,
    store: SharedImageStore,
    pipeline: Pipeline,
    experiment_index: int = 0,
    image: Image | None = None,
) -> ExperimentTask:
    image = image if image is not None else Image(rgb_gradient(8))
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
        source=store.put(image),
        second=None,
        block_snapshot_hash="",  # unused by run_experiment; init_worker verifies
    )


def test_init_worker_reports_ready(worker: tuple[WorkerInit, queue.Queue[WorkerEvent]]) -> None:
    _, events = worker
    event = events.get_nowait()
    assert isinstance(event, WorkerReady)
    assert event.pid == os.getpid()


def test_run_experiment_executes_and_writes_results(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]})
    )

    outcome = run_experiment(task)

    assert outcome.status == "ok"
    assert outcome.error_code is None
    assert len(outcome.results) == 1
    result = outcome.results[0]
    assert result.layer_index == 0
    assert result.path == "results/result_0000_00.png"
    assert result.thumb_path == "thumbs/result_0000_00.png"
    assert (result.width, result.height) == (8, 8)

    written = PILImage.open(run.run_dir / result.path)
    np.testing.assert_array_equal(np.asarray(written.convert("RGB")), 255 - rgb_gradient(8))
    thumb = PILImage.open(run.run_dir / result.thumb_path)
    assert thumb.format == "PNG"

    # The started event reached the queue before the outcome.
    _, events = worker
    assert isinstance(events.get_nowait(), WorkerReady)
    started = events.get_nowait()
    assert isinstance(started, ExperimentStarted)
    assert started.experiment_index == 0


def test_run_experiment_writes_one_result_per_layer(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_split2"}]})
    )

    outcome = run_experiment(task)

    assert outcome.status == "ok"
    assert [result.layer_index for result in outcome.results] == [0, 1]
    assert (run.run_dir / "results" / "result_0000_00.png").is_file()
    assert (run.run_dir / "results" / "result_0000_01.png").is_file()
    assert (run.run_dir / "thumbs" / "result_0000_00.png").is_file()
    assert (run.run_dir / "thumbs" / "result_0000_01.png").is_file()


def test_thumbnail_is_bounded_to_256(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    # Make the source large enough that the thumbnail must shrink.
    task = make_task(
        run,
        store,
        Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]}),
        image=Image(rgb_gradient(300)),
    )

    outcome = run_experiment(task)

    assert outcome.status == "ok"
    thumb = PILImage.open(run.run_dir / outcome.results[0].thumb_path)
    assert max(thumb.size) == 256


def test_raising_block_fails_the_experiment(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_raise"}]})
    )

    outcome = run_experiment(task)

    assert outcome.status == "failed"
    assert outcome.error_code == BLOCK_RAISED
    assert "block exploded" in (outcome.error_message or "")
    assert outcome.results == ()


def test_invalid_output_fails_the_experiment(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_bad_output"}]})
    )

    outcome = run_experiment(task)

    assert outcome.status == "failed"
    assert outcome.error_code == BLOCK_INVALID_OUTPUT


def test_snapshot_mismatch_fails_every_task(
    run: RunStore,
    store: SharedImageStore,
    blocks_dir: Path,
    clean_sys_path: None,
) -> None:
    events: queue.Queue[WorkerEvent] = queue.Queue()
    init = WorkerInit(
        event_queue=events,
        blocks_dir=str(blocks_dir),
        run_dir=str(run.run_dir),
        block_snapshot_hash="0" * 64,  # not the hash the worker computes
        log_level="DEBUG",
    )
    init_worker(init, events)

    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]})
    )

    outcome = run_experiment(task)

    assert outcome.status == "failed"
    assert outcome.error_code == BLOCK_SNAPSHOT_MISMATCH
    assert "does not match" in (outcome.error_message or "")


def test_worker_imports_no_qt() -> None:
    """The worker module must be importable without PySide6 (§1.3)."""
    project_root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pic_utils.runner.worker; "
            "assert 'PySide6' not in sys.modules, 'worker imported Qt'",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
        cwd=project_root,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_worker_outcome_has_no_image_data(
    run: RunStore,
    store: SharedImageStore,
    worker: tuple[WorkerInit, queue.Queue[WorkerEvent]],
) -> None:
    """The pickled outcome stays tiny: no image data crosses back (§8.3)."""
    task = make_task(
        run, store, Pipeline.from_dict({"steps": [{"id": "s1", "block": "test_invert"}]})
    )

    outcome = run_experiment(task)

    assert len(pickle.dumps(outcome)) < 1024

"""Shared helpers for the slow performance measurements (P7-03).

These helpers are imported by ``tests/perf/test_perf.py`` and by the
end-to-end acceptance suite. They build deterministic 48 MP sources, measure
the process-tree RSS, and drive a real :class:`RunService` batch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import psutil

from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.image import Image
from pic_utils.io.saver import save_image
from pic_utils.pipeline.model import Pipeline, RunConfig
from pic_utils.runner.service import RunnerSettings, RunRequest, RunService, RunSummary

__all__ = [
    "MIB",
    "RssSampler",
    "build_request",
    "make_48mp_source",
    "process_tree_rss_mb",
    "run_to_completion",
]

MIB = 1024 * 1024
DEFAULT_WIDTH = 8000
DEFAULT_HEIGHT = 6000  # exactly 48,000,000 pixels (the loader's ceiling)


def make_48mp_source(path: Path, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> Path:
    """Write a smooth 48 MP RGB gradient PNG (compresses well, no randomness)."""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = x.astype(np.uint8)[None, :]
    rgb[..., 1] = y.astype(np.uint8)[:, None]
    rgb[..., 2] = ((x[None, :] + y[:, None]) / 2.0).astype(np.uint8)
    save_image(Image.from_rgb(rgb), path)
    del rgb
    return path


def process_tree_rss_mb(pid: int | None = None) -> int:
    """Resident memory of a process and all its descendants, in MiB."""
    root = psutil.Process(pid)
    total = 0
    for process in [root, *root.children(recursive=True)]:
        try:
            total += process.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total // MIB


class RssSampler:
    """Sample the process-tree RSS on a background thread."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb = 0

    def __enter__(self) -> RssSampler:
        self._thread = threading.Thread(target=self._run, name="rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, process_tree_rss_mb())
            time.sleep(self._interval_s)


def build_request(
    results_root: Path,
    source: Path,
    *,
    experiments: int = 32,
    workers: int = 2,
    ceiling_mb: int = 8192,
) -> RunRequest:
    """A 32-experiment, one-source request for the default memory budget."""
    registry = BlockRegistry(None)
    pipeline = Pipeline.from_dict(
        {"steps": [{"id": "s1", "block": "channel_permute", "params": {"order": "bgr"}}]}
    )
    return RunRequest(
        pipeline=pipeline,
        snapshot=registry.snapshot(),
        sources=(source,),
        second=None,
        run_config=RunConfig(
            mode="random",
            batch_size=experiments,
            master_seed=11,
            sampled=(("s1", "order"),),
        ),
        results_root=results_root,
        settings=RunnerSettings(workers=workers, memory_ceiling_mb=ceiling_mb),
        user_blocks_dir=None,
    )


class _Collector:
    def __init__(self, service: RunService) -> None:
        self.summary: RunSummary | None = None
        self.failures: list[tuple[str, str]] = []
        service.run_finished.connect(self._finished)
        service.run_failed.connect(self._failed)

    def _finished(self, summary: object) -> None:
        self.summary = summary  # type: ignore[assignment]

    def _failed(self, code: str, message: str) -> None:
        self.failures.append((code, message))


def run_to_completion(
    service: RunService,
    request: RunRequest,
    wait: Callable[..., None],
    *,
    timeout_ms: int = 600_000,
) -> RunSummary:
    """Start ``request`` and wait for the summary (``wait`` is qtbot.waitUntil)."""
    collector = _Collector(service)
    service.start(request)
    wait(
        lambda: collector.summary is not None or bool(collector.failures),
        timeout=timeout_ms,
    )
    if collector.failures:
        raise AssertionError(f"run failed: {collector.failures}")
    assert collector.summary is not None
    wait(lambda: not service.is_running, timeout=10_000)
    return collector.summary

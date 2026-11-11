"""Worker protocol: task, outcome, and event-queue value types (§8.3, §8.2).

Everything here is picklable and Qt-free: these objects cross the process
boundary between :mod:`pic_utils.runner.pool` and the spawned workers. The
outcome carries small metadata only — no image data ever crosses back
(§1.2). An outcome status is ``"ok"``, ``"failed"`` (with a stable error
code from :mod:`pic_utils.core.errors`, e.g. ``block.raised`` or
``block.invalid_output``), or ``"cancelled"`` (force stop, §8.4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pic_utils.pipeline.model import ExperimentSpec

__all__ = [
    "ExperimentOutcome",
    "ExperimentStarted",
    "ExperimentTask",
    "ImageHandle",
    "ResultFile",
    "WorkerEvent",
    "WorkerLog",
    "WorkerReady",
]


@dataclass(frozen=True, slots=True)
class ImageHandle:
    """A reference to one image placed in shared memory (§8.8).

    ``name`` is the shared-memory block holding the ``(H, W, 3)`` RGB plane;
    ``alpha_name`` optionally holds the separate ``(H, W)`` alpha plane.
    """

    name: str
    shape: tuple[int, int, int]
    dtype: str = "uint8"
    alpha_name: str | None = None
    alpha_shape: tuple[int, int] | None = None

    @property
    def nbytes(self) -> int:
        """Total shared bytes backing this image (memory accounting, §8.6)."""
        rgb_bytes = math.prod(self.shape)
        alpha_bytes = 0 if self.alpha_shape is None else math.prod(self.alpha_shape)
        return rgb_bytes + alpha_bytes


@dataclass(frozen=True, slots=True)
class ExperimentTask:
    """One experiment dispatched to a worker."""

    run_dir: str
    experiment: ExperimentSpec
    pipeline: dict  # serialized Pipeline (Pipeline.to_dict())
    source: ImageHandle
    second: ImageHandle | None
    block_snapshot_hash: str


@dataclass(frozen=True, slots=True)
class ResultFile:
    """One layer written by the worker, relative to ``run_dir``."""

    layer_index: int
    path: str  # relative to run_dir
    thumb_path: str  # relative to run_dir
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """The small result of one experiment; no image data (§8.3)."""

    experiment_index: int
    status: Literal["ok", "failed", "cancelled"]
    error_code: str | None
    error_message: str | None
    elapsed_ms: int
    results: tuple[ResultFile, ...]


# --- Event-queue records (§8.2) ----------------------------------------------
#
# Workers report readiness and experiment starts through a
# ``multiprocessing.Queue`` so the parent can attribute in-flight
# experiments to PIDs for crash recovery (§8.5) and route log records into
# ``run.log`` (§18.3).


@dataclass(frozen=True, slots=True)
class WorkerReady:
    """A worker process started and finished its initialization."""

    pid: int


@dataclass(frozen=True, slots=True)
class ExperimentStarted:
    """A worker began executing one experiment."""

    pid: int
    experiment_index: int


@dataclass(frozen=True, slots=True)
class WorkerLog:
    """A log record routed through the event queue; the parent writes it."""

    pid: int
    level: int
    logger_name: str
    message: str


WorkerEvent = WorkerReady | ExperimentStarted | WorkerLog

"""Memory guard: budget-aware admission control (architecture §8.6, AD-08).

The guard compares the app's RSS, the pool workers' RSS, and the estimated
per-experiment footprint against the configured ceiling minus a 512 MB
reserve (preview process, Qt caches, allocator overhead). Admission has
three outcomes: ``admit`` (dispatch), ``defer`` (a worker must finish
first), or ``fail`` (even an idle pool cannot hold the experiment — the
experiment fails with ``memory.limit``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import psutil  # type: ignore[import-untyped]  # psutil ships no py.typed; types-psutil is not an approved dependency

from pic_utils.blocks.api import OutputKind
from pic_utils.core.image import Image
from pic_utils.pipeline.model import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pic_utils.blocks.registry import BlockSnapshot

__all__ = ["RESERVE_MB", "MemoryGuard"]

RESERVE_MB = 512

_MIB = 1024 * 1024
_MAX_IMAGE_FACTOR = 8
_LAYERS_IMAGE_FACTOR = 8
_BASE_IMAGE_FACTOR = 2
_PER_STEP_IMAGE_FACTOR = 2


class MemoryGuard:
    """Budget-aware admission control against the memory ceiling (§8.6)."""

    def __init__(self, ceiling_mb: int = 8192) -> None:
        if ceiling_mb < RESERVE_MB:
            raise ValueError(f"memory ceiling must be at least {RESERVE_MB} MB, got {ceiling_mb}")
        self.ceiling_mb = ceiling_mb

    @property
    def budget_mb(self) -> int:
        """The ceiling minus the 512 MB reserve (§8.6)."""
        return self.ceiling_mb - RESERVE_MB

    def app_rss_mb(self) -> int:
        """The current process's RSS in MiB."""
        return _rss_mb(psutil.Process())

    def worker_rss_mb(self, pids: Iterable[int] | None = None) -> dict[int, int]:
        """RSS in MiB per live worker PID.

        Without ``pids``, every direct child of the current process is
        polled — the spawned pool workers (§8.6: psutil, 500 ms cadence).
        Dead PIDs are simply absent from the result.
        """
        if pids is None:
            children = psutil.Process().children(recursive=False)
            return {child.pid: _rss_mb(child) for child in children}
        result: dict[int, int] = {}
        for pid in pids:
            try:
                result[pid] = _rss_mb(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        return result

    def total_rss_mb(self, pids: Iterable[int] | None = None) -> int:
        """App RSS plus the given (or all child) workers' RSS, in MiB."""
        workers = self.worker_rss_mb(pids)
        return self.app_rss_mb() + sum(workers.values())

    def estimate_experiment_mb(
        self, source: Image, pipeline: Pipeline, snapshot: BlockSnapshot
    ) -> int:
        """Estimated peak footprint of one experiment, in MiB (§8.6).

        ``bytes_per_image = H*W*3``; the factor is ``min(2 + 2*image_steps,
        8)`` — or exactly 8 when the pipeline terminates in a ``LAYERS``
        step, which materializes every layer at once.
        """
        bytes_per_image = source.rgb.shape[0] * source.rgb.shape[1] * 3
        image_steps = 0
        has_layers_step = False
        for step in pipeline.steps:
            if step.bypass or step.block not in snapshot:
                continue
            definition = snapshot.get(step.block)

            if definition.output is OutputKind.LAYERS:
                has_layers_step = True
            elif definition.output is OutputKind.IMAGE:
                image_steps += 1
        factor = (
            _LAYERS_IMAGE_FACTOR
            if has_layers_step
            else min(_BASE_IMAGE_FACTOR + _PER_STEP_IMAGE_FACTOR * image_steps, _MAX_IMAGE_FACTOR)
        )
        return _ceil_to_mb(bytes_per_image * factor)

    def admission(self, estimate_mb: int) -> Literal["admit", "defer", "fail"]:
        """Decide whether an experiment may be dispatched now (§8.6).

        - ``fail``: even with an idle pool the experiment cannot fit
          (``app_rss + est > ceiling - reserve``) — the caller fails the
          experiment with ``memory.limit``.
        - ``defer``: it fits only if workers finish first — the caller waits
          for a worker instead of dispatching.
        - ``admit``: dispatch.
        """
        app = self.app_rss_mb()
        if app + estimate_mb > self.budget_mb:
            return "fail"
        workers = sum(self.worker_rss_mb().values())
        if app + workers + estimate_mb > self.budget_mb:
            return "defer"
        return "admit"

    def can_dispatch(self, estimate_mb: int) -> bool:
        """``True`` when the experiment may be dispatched immediately."""
        return self.admission(estimate_mb) == "admit"

    def can_hold_sources(self, total_source_bytes: int) -> bool:
        """Preflight: the selection's shared sources fit beside the app (§8.6)."""
        return self.app_rss_mb() + _ceil_to_mb(total_source_bytes) <= self.budget_mb

    def should_recycle(self, worker_rss: int, worker_count: int) -> bool:
        """``True`` when a worker's RSS exceeds ``ceiling / workers`` (§8.6)."""
        if worker_count < 1:
            return False
        return worker_rss > self.ceiling_mb // worker_count

    def max_workers(self, estimate_mb: int) -> int:
        """The largest worker count that fits the budget (§8.6 clamping)."""
        if estimate_mb < 1:
            raise ValueError(f"estimate must be positive, got {estimate_mb}")
        app = self.app_rss_mb()
        return max(1, (self.budget_mb - app) // estimate_mb)


def _rss_mb(process: psutil.Process) -> int:
    return process.memory_info().rss // _MIB


def _ceil_to_mb(bytes_value: int) -> int:
    return -(-bytes_value // _MIB)

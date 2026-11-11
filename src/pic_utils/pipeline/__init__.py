"""Pure pipeline package: model, validation, execution, planning.

Nothing in this package may import Qt or :mod:`pic_utils.io`; the pipeline is
a pure value layer that the UI and the runner both drive.
"""

from __future__ import annotations

from pic_utils.pipeline.model import (
    ExperimentSpec,
    Pipeline,
    PipelineStep,
    RunConfig,
    SweepAxis,
    derive_seed,
)

__all__ = [
    "ExperimentSpec",
    "Pipeline",
    "PipelineStep",
    "RunConfig",
    "SweepAxis",
    "derive_seed",
]

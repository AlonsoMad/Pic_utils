"""Experiment planning (architecture §7.2).

``build_plan`` turns a :class:`RunConfig` into the concrete experiment list
recorded in the run manifest. Planning is pure and deterministic: the same
config, pipeline, snapshot, and source count always produce the same specs,
and every experiment seed is derived from the master seed and its index.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from pic_utils.blocks.api import Choice, FloatRange, IntRange, ParamDef, ParamKind
from pic_utils.core.errors import ValidationError
from pic_utils.pipeline.model import (
    ExperimentSpec,
    Pipeline,
    RunConfig,
    SweepAxis,
    derive_seed,
)
from pic_utils.pipeline.validate import Issue

if TYPE_CHECKING:
    from pic_utils.blocks.registry import BlockSnapshot

__all__ = [
    "ESTIMATED_RESULT_BYTES",
    "MAX_EXPERIMENTS",
    "build_plan",
    "disk_preflight",
    "estimate_run_bytes",
    "sample_parameter",
]

MAX_EXPERIMENTS = 10_000
ESTIMATED_RESULT_BYTES = 30 * 1024 * 1024  # 48 MP PNG, §7.2


def build_plan(
    config: RunConfig,
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    source_count: int,
) -> list[ExperimentSpec]:
    """Generate the experiment list for ``config``.

    Raises:
        ValidationError: when the plan would exceed :data:`MAX_EXPERIMENTS`.
        ValueError: for an unknown run mode.
    """
    if source_count < 1:
        return []
    combos = _combinations(config.axes)
    total = _expected_count(config, combos, source_count)
    if total > MAX_EXPERIMENTS:
        raise ValidationError(
            [
                Issue(
                    severity="error",
                    code="run.too_many_experiments",
                    message=(f"the plan has {total} experiments; the limit is {MAX_EXPERIMENTS}"),
                )
            ]
        )
    if config.mode == "grid":
        return _grid_plan(config, combos, source_count)
    if config.mode == "random":
        return _random_plan(config, pipeline, snapshot, source_count)
    if config.mode == "hybrid":
        return _hybrid_plan(config, pipeline, snapshot, combos, source_count)
    raise ValueError(f"unknown run mode {config.mode!r}")


def _grid_plan(
    config: RunConfig, combos: list[tuple[Any, ...]], source_count: int
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    index = 0
    for source_index in range(source_count):
        for combination in combos:
            params = _axis_values(config.axes, combination)
            specs.append(_experiment(index, source_index, params, config))
            index += 1
    return specs


def _random_plan(
    config: RunConfig,
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    source_count: int,
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for index in range(config.batch_size):
        params = _sampled_values(config, pipeline, snapshot, index)
        specs.append(_experiment(index, index % source_count, params, config))
    return specs


def _hybrid_plan(
    config: RunConfig,
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    combos: list[tuple[Any, ...]],
    source_count: int,
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    index = 0
    for source_index in range(source_count):
        for combination in combos:
            base = _axis_values(config.axes, combination)
            for _ in range(config.random_samples):
                params = _sampled_values(config, pipeline, snapshot, index, base)
                specs.append(_experiment(index, source_index, params, config))
                index += 1
    return specs


def _combinations(axes: tuple[SweepAxis, ...]) -> list[tuple[Any, ...]]:
    if not axes:
        return [()]
    return list(itertools.product(*(axis.values for axis in axes)))


def _axis_values(
    axes: tuple[SweepAxis, ...], combination: tuple[Any, ...]
) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for axis, value in zip(axes, combination, strict=True):
        params.setdefault(axis.step_id, {})[axis.param] = value
    return params


def _sampled_values(
    config: RunConfig,
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    index: int,
    base: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    params = {step_id: dict(values) for step_id, values in (base or {}).items()}
    rng = np.random.default_rng(derive_seed(config.master_seed, "sample", index))
    for step_id, param_name in config.sampled:
        param = _param_def(pipeline, snapshot, step_id, param_name)
        if param is None:
            continue
        params.setdefault(step_id, {})[param_name] = sample_parameter(param, rng)
    return params


def _param_def(
    pipeline: Pipeline, snapshot: BlockSnapshot, step_id: str, param_name: str
) -> ParamDef | None:
    step = next((candidate for candidate in pipeline.steps if candidate.id == step_id), None)
    if step is None or step.block not in snapshot:
        return None
    definition = snapshot.get(step.block)
    return next((param for param in definition.params if param.name == param_name), None)


def _experiment(
    index: int,
    source_index: int,
    params: dict[str, dict[str, Any]],
    config: RunConfig,
) -> ExperimentSpec:
    return ExperimentSpec(
        index=index,
        source_index=source_index,
        params=params,
        seed=derive_seed(config.master_seed, "experiment", index),
        label=f"exp-{index:04d}",
    )


def _expected_count(config: RunConfig, combos: list[tuple[Any, ...]], source_count: int) -> int:
    """Total experiments for ``config``; ``combos`` is the enumerated axis product."""
    product = len(combos)
    if config.mode == "grid":
        return product * source_count
    if config.mode == "random":
        return config.batch_size
    if config.mode == "hybrid":
        return product * config.random_samples * source_count
    return 0


def sample_parameter(param: ParamDef, rng: np.random.Generator) -> Any:
    """Draw one value for ``param`` from its declared distribution (§7.2)."""
    if param.kind is ParamKind.INT:
        int_spec = param.spec if isinstance(param.spec, IntRange) else IntRange(0, 255)
        value = int(rng.integers(int_spec.min, int_spec.max + 1))
        if int_spec.step > 1:
            value = int_spec.min + ((value - int_spec.min) // int_spec.step) * int_spec.step
        return value
    if param.kind is ParamKind.FLOAT:
        float_spec = param.spec if isinstance(param.spec, FloatRange) else FloatRange(0.0, 1.0)
        if float_spec.log and float_spec.min > 0:
            sample = float(np.exp(rng.uniform(np.log(float_spec.min), np.log(float_spec.max))))
        else:
            sample = float(rng.uniform(float_spec.min, float_spec.max))
        if float_spec.step > 0:
            sample = float_spec.min + round((sample - float_spec.min) / float_spec.step) * (
                float_spec.step
            )
            sample = min(max(sample, float_spec.min), float_spec.max)
        return sample
    if param.kind is ParamKind.CHOICE:
        options = param.spec.options if isinstance(param.spec, Choice) else ()
        if not options:
            return param.default
        return options[int(rng.integers(0, len(options)))]
    if param.kind is ParamKind.BOOL:
        return bool(rng.integers(0, 2))
    if param.kind is ParamKind.COLOR:
        red, green, blue = (int(channel) for channel in rng.integers(0, 256, size=3))
        return f"#{red:02x}{green:02x}{blue:02x}"
    return param.default


def estimate_run_bytes(experiment_count: int, results_per_experiment: int = 1) -> int:
    """Estimate the disk footprint of a run (§7.2: 30 MB per 48 MP PNG)."""
    return experiment_count * results_per_experiment * ESTIMATED_RESULT_BYTES


def disk_preflight(
    experiment_count: int, free_bytes: int, *, results_per_experiment: int = 1
) -> bool:
    """Return ``True`` when free space covers twice the estimated footprint."""
    return free_bytes >= 2 * estimate_run_bytes(experiment_count, results_per_experiment)

"""Pipeline and run-configuration validation (architecture §6.3).

Validation is pure: it reads the pipeline, an immutable block snapshot, and
two availability flags, and returns a list of :class:`Issue` values. The UI
renders them inline and disables launch while any ``error`` exists; the
runner refuses to start on the same condition.

The block snapshot is imported only for type checking: ``blocks.registry``
imports Qt, and this package MUST stay Qt-free (architecture §1.3).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pic_utils.blocks.api import (
    BLOCK_API_VERSION,
    BlockDef,
    Choice,
    FloatRange,
    IntRange,
    OutputKind,
    ParamDef,
    ParamKind,
)
from pic_utils.pipeline.model import Pipeline, PipelineStep, RunConfig, SweepAxis

if TYPE_CHECKING:
    from pic_utils.blocks.registry import BlockSnapshot

__all__ = [
    "Issue",
    "is_blocking",
    "unreadable_source_issue",
    "validate_pipeline",
    "validate_run_config",
]

Severity = Literal["error", "warning"]

_MAX_EXPERIMENTS = 10_000

_SCALAR_KINDS = frozenset(
    {
        ParamKind.INT,
        ParamKind.FLOAT,
        ParamKind.CHOICE,
        ParamKind.BOOL,
        ParamKind.COLOR,
        ParamKind.SEED,
    }
)

_SAMPLEABLE_KINDS = frozenset(
    {
        ParamKind.INT,
        ParamKind.FLOAT,
        ParamKind.CHOICE,
        ParamKind.BOOL,
        ParamKind.COLOR,
    }
)


@dataclass(frozen=True, slots=True)
class Issue:
    """One validation finding, rendered inline next to its step/param."""

    severity: Severity
    code: str
    message: str
    step_id: str | None = None
    param: str | None = None


def is_blocking(issues: Iterable[Issue]) -> bool:
    """Return ``True`` when at least one issue is an error."""
    return any(issue.severity == "error" for issue in issues)


def unreadable_source_issue(path: str) -> Issue:
    """Build the warning emitted when a source image fails to load."""
    return Issue(
        severity="warning",
        code="source.unreadable",
        message=f"source {path!r} could not be loaded and will be skipped",
    )


def validate_pipeline(
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    *,
    second_image_available: bool,
    sources_available: bool,
) -> list[Issue]:
    """Validate pipeline structure, parameters, and mask bindings."""
    issues: list[Issue] = []
    if not sources_available:
        issues.append(
            Issue(
                severity="error",
                code="source.none",
                message="select at least one source image",
            )
        )
    if not pipeline.steps:
        issues.append(
            Issue(severity="error", code="pipeline.empty", message="the pipeline has no steps")
        )
        return issues

    index_by_id: dict[str, int] = {}
    for index, step in enumerate(pipeline.steps):
        if step.id in index_by_id:
            issues.append(
                Issue(
                    severity="error",
                    code="step.duplicate_id",
                    message=f"step id {step.id!r} is used more than once",
                    step_id=step.id,
                )
            )
        else:
            index_by_id[step.id] = index

    for index, step in enumerate(pipeline.steps):
        definition = _definition(step, snapshot, issues)
        if definition is None:
            continue
        _validate_step(
            step,
            definition,
            index=index,
            pipeline=pipeline,
            index_by_id=index_by_id,
            snapshot=snapshot,
            second_image_available=second_image_available,
            issues=issues,
        )

    _validate_terminal(pipeline, snapshot, issues)
    return issues


def _definition(
    step: PipelineStep, snapshot: BlockSnapshot, issues: list[Issue]
) -> BlockDef | None:
    if step.block not in snapshot:
        issues.append(
            Issue(
                severity="error",
                code="step.unknown_block",
                message=f"block {step.block!r} is not available",
                step_id=step.id,
            )
        )
        return None
    definition = snapshot.get(step.block)
    if definition.api_version != BLOCK_API_VERSION:
        issues.append(
            Issue(
                severity="warning",
                code="block.api_version",
                message=(
                    f"block {step.block!r} targets API version {definition.api_version}, "
                    f"expected {BLOCK_API_VERSION}"
                ),
                step_id=step.id,
            )
        )
    return definition


def _validate_step(
    step: PipelineStep,
    definition: BlockDef,
    *,
    index: int,
    pipeline: Pipeline,
    index_by_id: dict[str, int],
    snapshot: BlockSnapshot,
    second_image_available: bool,
    issues: list[Issue],
) -> None:
    by_name = {param.name: param for param in definition.params}

    for name, value in step.params.items():
        param = by_name.get(name)
        if param is None:
            issues.append(
                Issue(
                    severity="error",
                    code="param.unknown",
                    message=f"parameter {name!r} is not in block {definition.name!r}",
                    step_id=step.id,
                    param=name,
                )
            )
        elif param.kind not in _SCALAR_KINDS:
            issues.append(
                Issue(
                    severity="error",
                    code="param.type",
                    message=f"parameter {name!r} is not a scalar parameter",
                    step_id=step.id,
                    param=name,
                )
            )
        else:
            issues.extend(_check_value(definition, param, value, step.id))

    for param in definition.params:
        if param.required and param.kind in _SCALAR_KINDS and param.name not in step.params:
            issues.append(
                Issue(
                    severity="error",
                    code="param.missing",
                    message=f"parameter {param.name!r} has no value",
                    step_id=step.id,
                    param=param.name,
                )
            )

    for name in step.mask_bindings:
        bound = by_name.get(name)
        if bound is None or bound.kind is not ParamKind.MASK:
            issues.append(
                Issue(
                    severity="error",
                    code="param.unknown",
                    message=f"mask binding {name!r} does not name a mask parameter",
                    step_id=step.id,
                    param=name,
                )
            )

    for param in definition.params:
        if param.kind is not ParamKind.MASK:
            continue
        target = step.mask_bindings.get(param.name)
        if target is None:
            if param.required:
                issues.append(
                    Issue(
                        severity="error",
                        code="mask.unbound",
                        message=f"mask parameter {param.name!r} needs a binding",
                        step_id=step.id,
                        param=param.name,
                    )
                )
            continue
        issues.extend(
            _validate_binding(
                step,
                param,
                target,
                index=index,
                pipeline=pipeline,
                index_by_id=index_by_id,
                snapshot=snapshot,
            )
        )

    if len(definition.image_inputs) > 1 and not second_image_available:
        issues.append(
            Issue(
                severity="error",
                code="image.second_missing",
                message=f"block {definition.name!r} needs a second image",
                step_id=step.id,
            )
        )


def _validate_binding(
    step: PipelineStep,
    param: ParamDef,
    target: str,
    *,
    index: int,
    pipeline: Pipeline,
    index_by_id: dict[str, int],
    snapshot: BlockSnapshot,
) -> list[Issue]:
    target_index = index_by_id.get(target)
    if target_index is None:
        return [
            Issue(
                severity="error",
                code="mask.unbound",
                message=f"mask parameter {param.name!r} binds to unknown step {target!r}",
                step_id=step.id,
                param=param.name,
            )
        ]
    if target_index >= index:
        return [
            Issue(
                severity="error",
                code="mask.forward_reference",
                message=(
                    f"mask parameter {param.name!r} must bind to an earlier step, not {target!r}"
                ),
                step_id=step.id,
                param=param.name,
            )
        ]
    target_step = pipeline.steps[target_index]
    if target_step.bypass:
        return [
            Issue(
                severity="error",
                code="mask.source_bypassed",
                message=f"mask parameter {param.name!r} binds to bypassed step {target!r}",
                step_id=step.id,
                param=param.name,
            )
        ]
    if target_step.block not in snapshot:
        return []
    if snapshot.get(target_step.block).output is not OutputKind.MASK:
        return [
            Issue(
                severity="error",
                code="mask.type_mismatch",
                message=f"mask parameter {param.name!r} binds to non-mask step {target!r}",
                step_id=step.id,
                param=param.name,
            )
        ]
    return []


def _validate_terminal(pipeline: Pipeline, snapshot: BlockSnapshot, issues: list[Issue]) -> None:
    active = [step for step in pipeline.steps if not step.bypass]
    for step in pipeline.steps:
        if step.bypass or step.block not in snapshot:
            continue
        if snapshot.get(step.block).output is OutputKind.LAYERS and step is not pipeline.steps[-1]:
            issues.append(
                Issue(
                    severity="error",
                    code="step.layers_not_last",
                    message="a LAYERS block must be the last step",
                    step_id=step.id,
                )
            )
    if not active:
        issues.append(
            Issue(
                severity="error",
                code="step.no_output",
                message="the pipeline has no image-producing step",
            )
        )
        return
    last = active[-1]
    if last.block in snapshot and snapshot.get(last.block).output is OutputKind.MASK:
        issues.append(
            Issue(
                severity="error",
                code="step.trailing_mask",
                message="the last step produces a mask; the pipeline yields no image",
                step_id=last.id,
            )
        )
    produces_image = any(
        step.block in snapshot
        and snapshot.get(step.block).output in (OutputKind.IMAGE, OutputKind.LAYERS)
        for step in active
    )
    if not produces_image:
        issues.append(
            Issue(
                severity="error",
                code="step.no_output",
                message="the pipeline has no image-producing step",
            )
        )


def _check_value(definition: BlockDef, param: ParamDef, value: object, step_id: str) -> list[Issue]:
    code: str | None = None
    message: str | None = None
    kind = param.kind
    if kind is ParamKind.INT:
        if isinstance(value, bool) or not isinstance(value, int):
            code, message = "param.type", f"parameter {param.name!r} must be an integer"
        elif isinstance(param.spec, IntRange) and not param.spec.min <= value <= param.spec.max:
            code = "param.range"
            message = (
                f"parameter {param.name!r} must be between "
                f"{param.spec.min} and {param.spec.max}, got {value}"
            )
    elif kind is ParamKind.FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            code, message = "param.type", f"parameter {param.name!r} must be a number"
        elif isinstance(param.spec, FloatRange) and not param.spec.min <= value <= param.spec.max:
            code = "param.range"
            message = (
                f"parameter {param.name!r} must be between "
                f"{param.spec.min} and {param.spec.max}, got {value}"
            )
    elif kind is ParamKind.BOOL:
        if not isinstance(value, bool):
            code, message = "param.type", f"parameter {param.name!r} must be a bool"
    elif kind is ParamKind.CHOICE:
        if isinstance(value, (dict, list, set, tuple)) or value is None:
            code, message = "param.type", f"parameter {param.name!r} must be a scalar option"
        elif isinstance(param.spec, Choice) and value not in param.spec.options:
            code = "param.choice"
            message = f"parameter {param.name!r} must be one of {param.spec.options}, got {value!r}"
    elif kind is ParamKind.COLOR:
        if not isinstance(value, str):
            code, message = "param.type", f"parameter {param.name!r} must be a color string"
    elif (
        kind is ParamKind.SEED
        and value is not None
        and (isinstance(value, bool) or not isinstance(value, int))
    ):
        code, message = "param.type", f"parameter {param.name!r} must be an int or None"
    if code is None or message is None:
        return []
    return [
        Issue(
            severity="error",
            code=code,
            message=message,
            step_id=step_id,
            param=param.name,
        )
    ]


def validate_run_config(
    config: RunConfig,
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    *,
    source_count: int = 1,
) -> list[Issue]:
    """Validate sweep/sampling configuration against the pipeline."""
    issues: list[Issue] = []
    steps_by_id = {step.id: step for step in pipeline.steps}

    if config.mode == "random":
        if not config.sampled:
            issues.append(
                Issue(
                    severity="error",
                    code="run.no_sampled",
                    message="random mode needs at least one sampled parameter",
                )
            )
        if config.batch_size < 1:
            issues.append(
                Issue(
                    severity="error",
                    code="run.no_sampled",
                    message="random mode needs a positive batch size",
                )
            )
    elif config.mode == "hybrid" and config.random_samples < 1:
        issues.append(
            Issue(
                severity="error",
                code="run.no_sampled",
                message="hybrid mode needs at least one random sample per combination",
            )
        )

    for axis in config.axes:
        issues.extend(_validate_axis(axis, steps_by_id, snapshot))
    for step_id, param_name in config.sampled:
        issues.extend(_validate_sampled(step_id, param_name, steps_by_id, snapshot))

    total = _experiment_count(config, source_count)
    if total > _MAX_EXPERIMENTS:
        issues.append(
            Issue(
                severity="error",
                code="run.too_many_experiments",
                message=(f"the plan has {total} experiments; the limit is {_MAX_EXPERIMENTS}"),
            )
        )
    return issues


def _validate_axis(
    axis: SweepAxis, steps_by_id: dict[str, PipelineStep], snapshot: BlockSnapshot
) -> list[Issue]:
    step = steps_by_id.get(axis.step_id)
    if step is None:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=f"axis references unknown step {axis.step_id!r}",
                step_id=axis.step_id,
                param=axis.param,
            )
        ]
    if step.block not in snapshot:
        return []
    definition = snapshot.get(step.block)
    param = next((p for p in definition.params if p.name == axis.param), None)
    if param is None:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=(f"axis parameter {axis.param!r} is not in block {definition.name!r}"),
                step_id=axis.step_id,
                param=axis.param,
            )
        ]
    if param.kind not in _SAMPLEABLE_KINDS:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=f"parameter {axis.param!r} cannot be swept",
                step_id=axis.step_id,
                param=axis.param,
            )
        ]
    if not axis.values:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=f"axis {axis.param!r} has no values",
                step_id=axis.step_id,
                param=axis.param,
            )
        ]
    issues: list[Issue] = []
    for value in axis.values:
        issues.extend(_check_value(definition, param, value, axis.step_id))
    return issues


def _validate_sampled(
    step_id: str,
    param_name: str,
    steps_by_id: dict[str, PipelineStep],
    snapshot: BlockSnapshot,
) -> list[Issue]:
    step = steps_by_id.get(step_id)
    if step is None:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=f"sampled parameter references unknown step {step_id!r}",
                step_id=step_id,
                param=param_name,
            )
        ]
    if step.block not in snapshot:
        return []
    definition = snapshot.get(step.block)
    param = next((p for p in definition.params if p.name == param_name), None)
    if param is None:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=(f"sampled parameter {param_name!r} is not in block {definition.name!r}"),
                step_id=step_id,
                param=param_name,
            )
        ]
    if param.kind not in _SAMPLEABLE_KINDS:
        return [
            Issue(
                severity="error",
                code="run.invalid_axis",
                message=f"parameter {param_name!r} cannot be sampled",
                step_id=step_id,
                param=param_name,
            )
        ]
    return []


def _experiment_count(config: RunConfig, source_count: int) -> int:
    product = 1
    for axis in config.axes:
        product *= len(axis.values)
    if config.mode == "grid":
        return product * source_count
    if config.mode == "random":
        return config.batch_size
    if config.mode == "hybrid":
        return product * config.random_samples * source_count
    return 0

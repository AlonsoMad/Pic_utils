"""Pipeline execution (architecture §6.2).

``execute_pipeline`` is a pure function: no Qt, no I/O. It walks the chain,
binds image/mask/seed/scalar arguments from the block definition, validates
each block's output (§5.6), propagates alpha (§3.1), and returns one
:class:`Layer` per result.

A block exception propagates unchanged; the runner catches it per experiment
and records ``block.raised``. Invalid output raises
:class:`~pic_utils.core.errors.InvalidBlockOutput` (``block.invalid_output``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from pic_utils.blocks.api import BlockDef, OutputKind, ParamKind
from pic_utils.core.errors import InvalidBlockOutput, TooManyOutputsError
from pic_utils.core.image import Image, Layer, Mask
from pic_utils.pipeline.model import ExperimentSpec, Pipeline, PipelineStep, derive_seed

if TYPE_CHECKING:
    from pic_utils.blocks.registry import BlockSnapshot

__all__ = ["PipelineOutput", "execute_pipeline"]


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """The layers a pipeline produced: exactly 1 for image pipelines."""

    layers: tuple[Layer, ...]


def execute_pipeline(
    pipeline: Pipeline,
    snapshot: BlockSnapshot,
    source: Image,
    second: Image | None,
    experiment: ExperimentSpec,
) -> PipelineOutput:
    """Run ``pipeline`` on ``source`` with the experiment's parameters."""
    current = source
    masks: dict[str, Mask] = {}
    for step in pipeline.steps:
        if step.bypass:
            continue
        definition = snapshot.get(step.block)
        args = _arguments(definition, step, experiment, current, second, masks)
        result = definition.fn(**args)
        _validate_output(definition, result, current, step)
        if definition.output is OutputKind.IMAGE:
            current = _propagate_alpha(result, current)  # type: ignore[arg-type]
        elif definition.output is OutputKind.MASK:
            masks[step.id] = result  # type: ignore[assignment]
        else:
            return PipelineOutput(layers=tuple(result))  # type: ignore[arg-type]
    return PipelineOutput(layers=(Layer(current),))


def _arguments(
    definition: BlockDef,
    step: PipelineStep,
    experiment: ExperimentSpec,
    current: Image,
    second: Image | None,
    masks: dict[str, Mask],
) -> dict[str, Any]:
    values = experiment.params.get(step.id, {})
    args: dict[str, Any] = {}
    for param in definition.params:
        if param.kind is ParamKind.IMAGE:
            args[param.name] = current if param.name == definition.image_inputs[0] else second
        elif param.kind is ParamKind.MASK:
            target = step.mask_bindings.get(param.name)
            args[param.name] = masks[target] if target is not None else None
        elif param.kind is ParamKind.SEED:
            args[param.name] = _resolve_seed(step, param.name, experiment)
        else:
            args[param.name] = values.get(param.name, step.params.get(param.name, param.default))
    return args


def _resolve_seed(step: PipelineStep, param_name: str, experiment: ExperimentSpec) -> int:
    # An explicit None here means "not overridden" (the planner never emits
    # None), so it falls through to the pipeline value and then the default.
    explicit = experiment.params.get(step.id, {}).get(param_name)
    if explicit is None:
        explicit = step.params.get(param_name)
    if explicit is not None:
        return int(explicit)
    return derive_seed(experiment.seed, "step", step.id)


def _propagate_alpha(result: Image, chain_input: Image) -> Image:
    """Reattach the input alpha when a same-size block dropped it (§3.1)."""
    if (
        result.alpha is None
        and chain_input.alpha is not None
        and result.rgb.shape[:2] == chain_input.rgb.shape[:2]
    ):
        return result.with_alpha(chain_input.alpha)
    return result


def _validate_output(
    definition: BlockDef, result: object, chain_input: Image, step: PipelineStep
) -> None:
    if definition.output is OutputKind.IMAGE:
        if not isinstance(result, Image):
            _raise_invalid(definition, step, "expected an Image")
        return
    if definition.output is OutputKind.MASK:
        if not isinstance(result, Mask):
            _raise_invalid(definition, step, "expected a Mask")
        if result.data.shape != chain_input.rgb.shape[:2]:
            _raise_invalid(
                definition,
                step,
                f"mask shape {result.data.shape} does not match "
                f"image shape {chain_input.rgb.shape[:2]}",
            )
        if not result.soft and not np.all((result.data == 0) | (result.data == 255)):
            _raise_invalid(definition, step, "binary mask contains values other than 0 and 255")
        return
    if not isinstance(result, (list, tuple)):
        _raise_invalid(definition, step, "expected a list or tuple of Layer")
    if len(result) > 16:
        # §18.2: block.too_many_outputs — a distinct code from the generic
        # invalid-output case, so the UI can name the specific failure.
        raise TooManyOutputsError(block=definition.name, step_id=step.id) from ValueError(
            f"expected 1..16 layers, got {len(result)}"
        )
    if len(result) < 1:
        _raise_invalid(definition, step, "expected at least one Layer, got 0")
    for layer in result:
        if not isinstance(layer, Layer):
            _raise_invalid(definition, step, "expected every layer to be a Layer")


def _raise_invalid(definition: BlockDef, step: PipelineStep, reason: str) -> NoReturn:
    raise InvalidBlockOutput(block=definition.name, step_id=step.id) from ValueError(reason)

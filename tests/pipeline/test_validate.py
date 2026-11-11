"""Tests for pipeline and run-configuration validation (architecture §6.3)."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

from pic_utils.blocks.api import BlockDef, ParamDef, ParamKind
from pic_utils.blocks.registry import BlockRegistry, BlockSnapshot
from pic_utils.core.image import Mask
from pic_utils.pipeline.model import Pipeline, PipelineStep, RunConfig, SweepAxis
from pic_utils.pipeline.validate import (
    Issue,
    is_blocking,
    unreadable_source_issue,
    validate_pipeline,
    validate_run_config,
)

REGISTRY = BlockRegistry().snapshot()


def snapshot(*defs: BlockDef) -> BlockSnapshot:
    definitions = {definition.name: definition for definition in defs}
    return BlockSnapshot(defs=MappingProxyType(definitions), hash="test")


def codes(issues: list[Issue]) -> list[str]:
    return [issue.code for issue in issues]


def valid_pipeline() -> Pipeline:
    return Pipeline(
        steps=(
            PipelineStep(id="s1", block="channel_permute", params={"order": "bgr"}),
            PipelineStep(
                id="s2",
                block="channel_mask",
                params={"channel": "R", "low": 0, "high": 255},
            ),
            PipelineStep(
                id="s3",
                block="conditional_overlay",
                params={"threshold": 128},
                mask_bindings={"mask": "s2"},
            ),
        )
    )


def validate(pipeline: Pipeline, **kwargs: object) -> list[Issue]:
    options: dict[str, object] = {"second_image_available": True, "sources_available": True}
    options.update(kwargs)
    return validate_pipeline(pipeline, REGISTRY, **options)  # type: ignore[arg-type]


def required_mask_def() -> BlockDef:
    base = REGISTRY.get("channel_permute")
    param = ParamDef(
        name="mask",
        kind=ParamKind.MASK,
        annotation=Mask,
        default=None,
        required=True,
        spec=None,
    )
    return replace(base, params=(*base.params, param), mask_inputs=("mask",))


def required_scalar_def() -> BlockDef:
    base = REGISTRY.get("channel_permute")
    params = tuple(
        replace(param, required=True, default=None) if param.name == "order" else param
        for param in base.params
    )
    return replace(base, params=params)


def test_valid_pipeline_has_no_issues() -> None:
    assert validate(valid_pipeline()) == []


def test_empty_pipeline() -> None:
    issues = validate(Pipeline())
    assert codes(issues) == ["pipeline.empty"]
    assert is_blocking(issues)


def test_unknown_block() -> None:
    issues = validate(Pipeline(steps=(PipelineStep(id="a", block="nope"),)))
    assert "step.unknown_block" in codes(issues)
    assert issues[0].step_id == "a"


def test_duplicate_id() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(id="a", block="channel_permute"),
        )
    )
    assert "step.duplicate_id" in codes(validate(pipeline))


def test_layers_not_last() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="permutation_grid"),
            PipelineStep(id="b", block="channel_permute"),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["step.layers_not_last"]
    assert issues[0].step_id == "a"


def test_layers_last_is_valid() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(id="b", block="permutation_grid"),
        )
    )
    assert validate(pipeline) == []


def test_trailing_mask() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(id="b", block="channel_mask"),
        )
    )
    issues = validate(pipeline)
    assert "step.trailing_mask" in codes(issues)
    assert issues[-1].step_id == "b"


def test_bypassed_trailing_mask_is_ignored() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(id="b", block="channel_mask", bypass=True),
        )
    )
    assert validate(pipeline) == []


def test_no_output() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_mask"),))
    assert "step.no_output" in codes(validate(pipeline))


def test_all_bypassed_has_no_output() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute", bypass=True),))
    assert codes(validate(pipeline)) == ["step.no_output"]


def test_param_missing() -> None:
    snap = snapshot(required_scalar_def())
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute"),))
    issues = validate_pipeline(pipeline, snap, second_image_available=True, sources_available=True)
    assert codes(issues) == ["param.missing"]
    assert issues[0].param == "order"


def test_param_unknown() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute", params={"bogus": 1}),))
    issues = validate(pipeline)
    assert codes(issues) == ["param.unknown"]
    assert issues[0].param == "bogus"


def test_param_type() -> None:
    pipeline = Pipeline(
        steps=(PipelineStep(id="a", block="channel_permute", params={"invert_r": "yes"}),)
    )
    issues = validate(pipeline)
    assert codes(issues) == ["param.type"]
    assert issues[0].param == "invert_r"


def test_param_range() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(id="b", block="channel_mask", params={"low": 999}),
            PipelineStep(id="c", block="channel_permute"),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["param.range"]
    assert issues[0].param == "low"


def test_param_choice() -> None:
    pipeline = Pipeline(
        steps=(PipelineStep(id="a", block="channel_permute", params={"order": "zzz"}),)
    )
    issues = validate(pipeline)
    assert codes(issues) == ["param.choice"]
    assert issues[0].param == "order"


def test_param_float_accepts_int() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="overlay", params={"opacity": 1}),))
    assert validate(pipeline) == []


def test_mask_unbound() -> None:
    snap = snapshot(required_mask_def())
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute"),))
    issues = validate_pipeline(pipeline, snap, second_image_available=True, sources_available=True)
    assert codes(issues) == ["mask.unbound"]
    assert issues[0].param == "mask"


def test_mask_unknown_target() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(
                id="b",
                block="conditional_overlay",
                mask_bindings={"mask": "ghost"},
            ),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["mask.unbound"]
    assert "ghost" in issues[0].message


def test_mask_forward_reference() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(
                id="a",
                block="conditional_overlay",
                mask_bindings={"mask": "b"},
            ),
            PipelineStep(id="b", block="channel_mask"),
        )
    )
    issues = validate(pipeline)
    assert "mask.forward_reference" in codes(issues)


def test_mask_self_reference_is_forward() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(
                id="b",
                block="conditional_overlay",
                mask_bindings={"mask": "b"},
            ),
        )
    )
    assert "mask.forward_reference" in codes(validate(pipeline))


def test_mask_source_bypassed() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_mask", bypass=True),
            PipelineStep(
                id="b",
                block="conditional_overlay",
                mask_bindings={"mask": "a"},
            ),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["mask.source_bypassed"]


def test_mask_type_mismatch() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute"),
            PipelineStep(
                id="b",
                block="conditional_overlay",
                mask_bindings={"mask": "a"},
            ),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["mask.type_mismatch"]


def test_mask_binding_to_unknown_param() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_mask"),
            PipelineStep(
                id="b",
                block="channel_permute",
                mask_bindings={"ghost": "a"},
            ),
        )
    )
    issues = validate(pipeline)
    assert codes(issues) == ["param.unknown"]
    assert issues[0].param == "ghost"


def test_second_image_missing() -> None:
    issues = validate(valid_pipeline(), second_image_available=False)
    assert "image.second_missing" in codes(issues)
    assert [issue.step_id for issue in issues if issue.code == "image.second_missing"] == ["s3"]


def test_api_version_warning() -> None:
    snap = snapshot(replace(REGISTRY.get("channel_permute"), api_version=2))
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute"),))
    issues = validate_pipeline(pipeline, snap, second_image_available=True, sources_available=True)
    assert codes(issues) == ["block.api_version"]
    assert issues[0].severity == "warning"
    assert not is_blocking(issues)


def test_sources_none() -> None:
    issues = validate(valid_pipeline(), sources_available=False)
    assert codes(issues) == ["source.none"]
    assert is_blocking(issues)


def test_unreadable_source_issue() -> None:
    issue = unreadable_source_issue("broken.png")
    assert issue.code == "source.unreadable"
    assert issue.severity == "warning"
    assert "broken.png" in issue.message
    assert not is_blocking([issue])


def test_is_blocking_ignores_warnings() -> None:
    assert not is_blocking([])
    assert is_blocking([Issue(severity="error", code="x", message="m")])


def run_config(**overrides: object) -> RunConfig:
    values: dict[str, object] = {"mode": "grid"}
    values.update(overrides)
    return RunConfig(**values)  # type: ignore[arg-type]


def test_valid_run_config() -> None:
    config = run_config(
        axes=(SweepAxis(step_id="s2", param="low", values=(0, 64, 128)),),
    )
    assert validate_run_config(config, valid_pipeline(), REGISTRY, source_count=3) == []


def test_random_requires_sampled() -> None:
    issues = validate_run_config(run_config(mode="random"), valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.no_sampled"]


def test_random_requires_positive_batch() -> None:
    config = run_config(mode="random", batch_size=0, sampled=(("s1", "order"),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.no_sampled"]


def test_hybrid_requires_samples() -> None:
    config = run_config(mode="hybrid", random_samples=0)
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.no_sampled"]


def test_axis_unknown_step() -> None:
    config = run_config(axes=(SweepAxis(step_id="ghost", param="low", values=(1,)),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]
    assert issues[0].step_id == "ghost"


def test_axis_unknown_param() -> None:
    config = run_config(axes=(SweepAxis(step_id="s2", param="ghost", values=(1,)),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]


def test_axis_incompatible_kind() -> None:
    config = run_config(axes=(SweepAxis(step_id="s3", param="mask", values=(1,)),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]


def test_axis_without_values() -> None:
    config = run_config(axes=(SweepAxis(step_id="s2", param="low", values=()),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]


def test_axis_value_out_of_range() -> None:
    config = run_config(axes=(SweepAxis(step_id="s2", param="low", values=(999,)),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["param.range"]


def test_sampled_invalid_param() -> None:
    config = run_config(mode="random", sampled=(("s1", "image"),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]


def test_sampled_unknown_step() -> None:
    config = run_config(mode="random", sampled=(("ghost", "order"),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY)
    assert codes(issues) == ["run.invalid_axis"]


def test_too_many_experiments() -> None:
    config = run_config(axes=(SweepAxis(step_id="s2", param="low", values=tuple(range(101))),))
    issues = validate_run_config(config, valid_pipeline(), REGISTRY, source_count=100)
    assert codes(issues) == ["run.too_many_experiments"]


def test_experiment_limit_boundary() -> None:
    config = run_config(axes=(SweepAxis(step_id="s2", param="low", values=tuple(range(100))),))
    assert validate_run_config(config, valid_pipeline(), REGISTRY, source_count=100) == []


def test_hybrid_experiment_count() -> None:
    config = run_config(
        mode="hybrid",
        random_samples=2,
        axes=(SweepAxis(step_id="s2", param="low", values=(0, 128)),),
    )
    assert validate_run_config(config, valid_pipeline(), REGISTRY, source_count=2) == []


def test_grid_without_axes_counts_sources() -> None:
    assert validate_run_config(run_config(), valid_pipeline(), REGISTRY, source_count=5) == []


def test_random_with_sampled_is_valid() -> None:
    config = run_config(mode="random", batch_size=3, sampled=(("s1", "order"),))
    assert validate_run_config(config, valid_pipeline(), REGISTRY) == []


def test_hybrid_with_samples_is_valid() -> None:
    config = run_config(mode="hybrid", random_samples=1)
    assert validate_run_config(config, valid_pipeline(), REGISTRY) == []

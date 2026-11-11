"""Tests for experiment planning (architecture §7.2)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from pic_utils.blocks.api import (
    Choice,
    Color,
    FloatRange,
    IntRange,
    ParamDef,
    ParamKind,
    Seed,
)
from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.errors import ValidationError
from pic_utils.pipeline.model import Pipeline, PipelineStep, RunConfig, SweepAxis, derive_seed
from pic_utils.pipeline.plan import (
    ESTIMATED_RESULT_BYTES,
    MAX_EXPERIMENTS,
    build_plan,
    disk_preflight,
    estimate_run_bytes,
    sample_parameter,
)

SNAPSHOT = BlockRegistry().snapshot()


def pipeline() -> Pipeline:
    return Pipeline(
        steps=(
            PipelineStep(id="perm", block="channel_permute", params={"order": "rgb"}),
            PipelineStep(id="mask", block="channel_mask", params={"low": 0, "high": 255}),
            PipelineStep(id="overlay", block="overlay"),
        )
    )


def config(**overrides: object) -> RunConfig:
    values: dict[str, object] = {"mode": "grid"}
    values.update(overrides)
    return RunConfig(**values)  # type: ignore[arg-type]


def test_grid_product_per_source() -> None:
    plan = build_plan(
        config(
            axes=(
                SweepAxis(step_id="perm", param="order", values=("rgb", "bgr")),
                SweepAxis(step_id="mask", param="low", values=(0, 64, 128)),
            )
        ),
        pipeline(),
        SNAPSHOT,
        source_count=2,
    )
    assert len(plan) == 12
    assert [spec.index for spec in plan] == list(range(12))
    assert [spec.source_index for spec in plan] == [0] * 6 + [1] * 6
    assert plan[0].params == {"perm": {"order": "rgb"}, "mask": {"low": 0}}
    assert plan[1].params == {"perm": {"order": "rgb"}, "mask": {"low": 64}}
    assert plan[3].params == {"perm": {"order": "bgr"}, "mask": {"low": 0}}


def test_grid_without_axes_is_one_per_source() -> None:
    plan = build_plan(config(), pipeline(), SNAPSHOT, source_count=3)
    assert len(plan) == 3
    assert [spec.source_index for spec in plan] == [0, 1, 2]
    assert all(spec.params == {} for spec in plan)


def test_grid_seeds_are_derived() -> None:
    plan = build_plan(config(master_seed=7), pipeline(), SNAPSHOT, source_count=2)
    assert [spec.seed for spec in plan] == [
        derive_seed(7, "experiment", 0),
        derive_seed(7, "experiment", 1),
    ]


def test_random_round_robins_sources() -> None:
    plan = build_plan(
        config(mode="random", batch_size=5, sampled=(("mask", "low"),)),
        pipeline(),
        SNAPSHOT,
        source_count=2,
    )
    assert len(plan) == 5
    assert [spec.source_index for spec in plan] == [0, 1, 0, 1, 0]


def test_random_samples_values_in_range() -> None:
    plan = build_plan(
        config(
            mode="random",
            batch_size=20,
            sampled=(("mask", "low"), ("perm", "order")),
        ),
        pipeline(),
        SNAPSHOT,
        source_count=1,
    )
    for spec in plan:
        assert 0 <= spec.params["mask"]["low"] <= 255
        assert spec.params["perm"]["order"] in ("rgb", "rbg", "grb", "gbr", "brg", "bgr")


def test_random_is_deterministic() -> None:
    run_config = config(mode="random", batch_size=8, sampled=(("mask", "low"),), master_seed=3)
    first = build_plan(run_config, pipeline(), SNAPSHOT, source_count=2)
    second = build_plan(run_config, pipeline(), SNAPSHOT, source_count=2)
    assert first == second


def test_master_seed_changes_the_plan() -> None:
    base = config(mode="random", batch_size=8, sampled=(("mask", "low"),), master_seed=3)
    other = config(mode="random", batch_size=8, sampled=(("mask", "low"),), master_seed=4)
    first = build_plan(base, pipeline(), SNAPSHOT, source_count=1)
    second = build_plan(other, pipeline(), SNAPSHOT, source_count=1)
    assert [spec.seed for spec in first] != [spec.seed for spec in second]
    assert [spec.params for spec in first] != [spec.params for spec in second]


def test_hybrid_count_and_content() -> None:
    plan = build_plan(
        config(
            mode="hybrid",
            random_samples=2,
            axes=(SweepAxis(step_id="perm", param="order", values=("rgb", "bgr")),),
            sampled=(("mask", "low"),),
        ),
        pipeline(),
        SNAPSHOT,
        source_count=2,
    )
    assert len(plan) == 2 * 2 * 2
    assert [spec.source_index for spec in plan] == [0] * 4 + [1] * 4
    for spec in plan:
        assert spec.params["perm"]["order"] in ("rgb", "bgr")
        assert 0 <= spec.params["mask"]["low"] <= 255
    first_source = plan[:4]
    assert [spec.params["perm"]["order"] for spec in first_source] == [
        "rgb",
        "rgb",
        "bgr",
        "bgr",
    ]
    assert len({spec.params["mask"]["low"] for spec in first_source}) > 1


def test_unsampled_params_keep_the_pipeline_value() -> None:
    plan = build_plan(
        config(mode="random", batch_size=3, sampled=(("mask", "low"),)),
        pipeline(),
        SNAPSHOT,
        source_count=1,
    )
    for spec in plan:
        assert "order" not in spec.params.get("perm", {})


def test_unresolved_sampled_entry_is_skipped() -> None:
    plan = build_plan(
        config(mode="random", batch_size=2, sampled=(("ghost", "low"), ("mask", "ghost"))),
        pipeline(),
        SNAPSHOT,
        source_count=1,
    )
    assert len(plan) == 2
    assert all("ghost" not in spec.params for spec in plan)


def test_zero_sources_yields_no_plan() -> None:
    assert build_plan(config(), pipeline(), SNAPSHOT, source_count=0) == []


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown run mode"):
        build_plan(config(mode="shuffle"), pipeline(), SNAPSHOT, source_count=1)


def test_too_many_experiments_raises() -> None:
    run_config = config(axes=(SweepAxis(step_id="mask", param="low", values=tuple(range(101))),))
    with pytest.raises(ValidationError) as caught:
        build_plan(run_config, pipeline(), SNAPSHOT, source_count=100)
    assert caught.value.issues[0].code == "run.too_many_experiments"


def test_experiment_limit_boundary_is_allowed() -> None:
    run_config = config(axes=(SweepAxis(step_id="mask", param="low", values=tuple(range(100))),))
    plan = build_plan(run_config, pipeline(), SNAPSHOT, source_count=100)
    assert len(plan) == MAX_EXPERIMENTS


def test_random_limit_is_checked() -> None:
    run_config = config(mode="random", batch_size=MAX_EXPERIMENTS + 1)
    with pytest.raises(ValidationError):
        build_plan(run_config, pipeline(), SNAPSHOT, source_count=1)


def test_hybrid_limit_is_checked() -> None:
    run_config = config(
        mode="hybrid",
        random_samples=2,
        axes=(SweepAxis(step_id="mask", param="low", values=tuple(range(51))),),
    )
    with pytest.raises(ValidationError):
        build_plan(run_config, pipeline(), SNAPSHOT, source_count=100)


def test_estimate_run_bytes() -> None:
    assert estimate_run_bytes(0) == 0
    assert estimate_run_bytes(1) == ESTIMATED_RESULT_BYTES
    assert estimate_run_bytes(10, results_per_experiment=2) == 20 * ESTIMATED_RESULT_BYTES


def test_disk_preflight_requires_headroom() -> None:
    estimate = estimate_run_bytes(10)
    assert disk_preflight(10, 2 * estimate)
    assert not disk_preflight(10, 2 * estimate - 1)


def int_param(spec: IntRange) -> ParamDef:
    return ParamDef(
        name="value",
        kind=ParamKind.INT,
        annotation=int,
        default=spec.min,
        required=False,
        spec=spec,
    )


def float_param(spec: FloatRange) -> ParamDef:
    return ParamDef(
        name="value",
        kind=ParamKind.FLOAT,
        annotation=float,
        default=spec.min,
        required=False,
        spec=spec,
    )


def test_sample_int_aligned_to_step() -> None:
    rng = np.random.default_rng(0)
    param = int_param(IntRange(0, 100, step=10))
    values = {sample_parameter(param, rng) for _ in range(200)}
    assert values <= {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
    assert len(values) > 1


def test_sample_int_without_spec_stays_in_byte_range() -> None:
    rng = np.random.default_rng(1)
    param = ParamDef(
        name="value",
        kind=ParamKind.INT,
        annotation=int,
        default=0,
        required=False,
        spec=None,
    )
    values = [sample_parameter(param, rng) for _ in range(100)]
    assert all(0 <= value <= 255 for value in values)


def test_sample_float_is_quantized() -> None:
    rng = np.random.default_rng(2)
    param = float_param(FloatRange(0.0, 1.0, step=0.01))
    for _ in range(100):
        value = sample_parameter(param, rng)
        assert 0.0 <= value <= 1.0
        assert abs(value * 100 - round(value * 100)) < 1e-9


def test_sample_float_log_range() -> None:
    rng = np.random.default_rng(3)
    param = float_param(FloatRange(1.0, 1000.0, step=1.0, log=True))
    values = [sample_parameter(param, rng) for _ in range(200)]
    assert all(1.0 <= value <= 1000.0 for value in values)
    assert max(values) > 100.0


def test_sample_choice_and_bool() -> None:
    rng = np.random.default_rng(4)
    choice = ParamDef(
        name="value",
        kind=ParamKind.CHOICE,
        annotation=str,
        default="a",
        required=False,
        spec=Choice(("a", "b", "c")),
    )
    assert {sample_parameter(choice, rng) for _ in range(50)} == {"a", "b", "c"}
    boolean = ParamDef(
        name="value",
        kind=ParamKind.BOOL,
        annotation=bool,
        default=False,
        required=False,
        spec=None,
    )
    values = {sample_parameter(boolean, rng) for _ in range(50)}
    assert values == {True, False}


def test_sample_color() -> None:
    rng = np.random.default_rng(5)
    param = ParamDef(
        name="value",
        kind=ParamKind.COLOR,
        annotation=str,
        default="#000000",
        required=False,
        spec=Color(),
    )
    values = [sample_parameter(param, rng) for _ in range(20)]
    assert all(re.fullmatch(r"#[0-9a-f]{6}", value) for value in values)
    assert len(set(values)) > 1


def test_seed_is_never_sampled() -> None:
    rng = np.random.default_rng(6)
    param = ParamDef(
        name="seed",
        kind=ParamKind.SEED,
        annotation=int | None,
        default=None,
        required=False,
        spec=Seed(),
    )
    assert sample_parameter(param, rng) is None

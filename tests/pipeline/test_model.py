"""Tests for the pipeline model and seed derivation (architecture §6.1, §7.2)."""

from __future__ import annotations

import json

import pytest

from pic_utils.pipeline.model import (
    ExperimentSpec,
    Pipeline,
    PipelineStep,
    RunConfig,
    SweepAxis,
    derive_seed,
)


def _sample_pipeline() -> Pipeline:
    return Pipeline(
        steps=(
            PipelineStep(
                id="a1",
                block="channel_permute",
                params={"order": "bgr", "invert_r": True},
            ),
            PipelineStep(
                id="b2",
                block="channel_mask",
                params={"channel": "R", "low": 10, "high": 200, "soft": False},
            ),
            PipelineStep(
                id="c3",
                block="overlay",
                params={"opacity": 0.5, "x": -4, "z_order": "over"},
                mask_bindings={"mask": "b2"},
            ),
            PipelineStep(id="d4", block="double_exposure", params={"seed": None}, bypass=True),
        )
    )


def test_round_trip_is_lossless() -> None:
    pipeline = _sample_pipeline()
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert restored == pipeline
    assert [step.id for step in restored.steps] == ["a1", "b2", "c3", "d4"]


def test_serialization_is_json_safe() -> None:
    payload = _sample_pipeline().to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["steps"][0] == {
        "id": "a1",
        "block": "channel_permute",
        "params": {"order": "bgr", "invert_r": True},
        "bypass": False,
        "mask_bindings": {},
    }


def test_empty_pipeline_round_trips() -> None:
    pipeline = Pipeline()
    assert pipeline.to_dict() == {"steps": []}
    assert Pipeline.from_dict({"steps": []}) == pipeline


def test_step_ids_survive_round_trip() -> None:
    pipeline = _sample_pipeline()
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert [step.id for step in restored.steps] == [step.id for step in pipeline.steps]


def test_from_dict_fills_optional_fields() -> None:
    pipeline = Pipeline.from_dict({"steps": [{"id": "x", "block": "channel_permute"}]})
    assert pipeline.steps == (PipelineStep(id="x", block="channel_permute"),)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"steps": "nope"},
        {"steps": [42]},
        {"steps": [{"block": "channel_permute"}]},
        {"steps": [{"id": "", "block": "channel_permute"}]},
        {"steps": [{"id": "x"}]},
        {"steps": [{"id": "x", "block": "b", "params": []}]},
        {"steps": [{"id": "x", "block": "b", "mask_bindings": {"m": 1}}]},
        {"steps": [{"id": "x", "block": "b", "bypass": "yes"}]},
    ],
)
def test_from_dict_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(ValueError):
        Pipeline.from_dict(payload)  # type: ignore[arg-type]


def test_derive_seed_is_stable() -> None:
    # Pinned values: changing the derivation silently invalidates every saved
    # run manifest, so the algorithm is frozen by these assertions.
    assert derive_seed(0, "experiment", 0) == 717365314
    assert derive_seed(0, "step", "abc") == 984941355
    assert derive_seed(42, "experiment", 7) == 1396991723


def test_derive_seed_varies_with_master_and_parts() -> None:
    assert derive_seed(0, "experiment", 0) != derive_seed(1, "experiment", 0)
    assert derive_seed(0, "experiment", 0) != derive_seed(0, "experiment", 1)
    assert 0 <= derive_seed(2**63 - 1, "x") <= 0x7FFFFFFF


def test_plan_value_types_are_frozen() -> None:
    axis = SweepAxis(step_id="s", param="p", values=(1, 2))
    config = RunConfig(mode="grid", axes=(axis,), sampled=(("s", "q"),))
    spec = ExperimentSpec(index=0, source_index=0, params={"s": {"p": 1}}, seed=3, label="e")
    assert config.axes[0] is axis
    assert spec.params == {"s": {"p": 1}}
    with pytest.raises(AttributeError):
        config.mode = "random"  # type: ignore[misc]

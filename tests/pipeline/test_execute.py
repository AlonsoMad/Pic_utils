"""Tests for pipeline execution (architecture §6.2, §5.6, §3.1)."""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated

import fixtures
import numpy as np
import pytest

from pic_utils.blocks.api import OutputKind, Seed, block
from pic_utils.blocks.builtin.channels import channel_permute, permutation_grid
from pic_utils.blocks.builtin.compose import conditional_overlay, overlay
from pic_utils.blocks.builtin.masks import channel_mask
from pic_utils.blocks.registry import BlockRegistry, BlockSnapshot
from pic_utils.core.errors import InvalidBlockOutput
from pic_utils.core.image import Image, Layer, Mask
from pic_utils.pipeline.execute import PipelineOutput, execute_pipeline
from pic_utils.pipeline.model import ExperimentSpec, Pipeline, PipelineStep, derive_seed

REGISTRY = BlockRegistry().snapshot()


@block(
    name="drop_alpha",
    category="test",
    description="Copy the RGB plane and drop alpha.",
    output=OutputKind.IMAGE,
)
def drop_alpha(image: Image) -> Image:
    return Image(image.rgb.copy())


@block(
    name="shrink",
    category="test",
    description="Crop to the top-left 2x2 pixels.",
    output=OutputKind.IMAGE,
)
def shrink(image: Image) -> Image:
    return Image(image.rgb[:2, :2].copy())


@block(
    name="make_opaque",
    category="test",
    description="Return the RGB plane with an explicit opaque alpha.",
    output=OutputKind.IMAGE,
)
def make_opaque(image: Image) -> Image:
    rgb = image.rgb.copy()
    return Image(rgb, np.full(rgb.shape[:2], 255, np.uint8))


@block(
    name="seed_probe",
    category="test",
    description="Encode the resolved seed into every pixel.",
    output=OutputKind.IMAGE,
)
def seed_probe(image: Image, seed: Annotated[int | None, Seed()] = None) -> Image:
    value = 0 if seed is None else seed % 256
    return Image(np.full_like(image.rgb, value))


@block(
    name="boom",
    category="test",
    description="Raise while executing.",
    output=OutputKind.IMAGE,
)
def boom(image: Image) -> Image:
    raise RuntimeError("boom")


@block(
    name="bad_image",
    category="test",
    description="Return a non-Image.",
    output=OutputKind.IMAGE,
)
def bad_image(image: Image) -> Image:
    return "nope"  # type: ignore[return-value]


@block(
    name="bad_mask",
    category="test",
    description="Return a mask with the wrong shape.",
    output=OutputKind.MASK,
)
def bad_mask(image: Image) -> Mask:
    return Mask(np.zeros((3, 3), np.uint8))


@block(
    name="binary_mask",
    category="test",
    description="Return a binary mask with a non-binary value.",
    output=OutputKind.MASK,
)
def binary_mask(image: Image) -> Mask:
    return Mask(np.full(image.rgb.shape[:2], 128, np.uint8), soft=False)


@block(
    name="bad_layers",
    category="test",
    description="Return an empty layer list.",
    output=OutputKind.LAYERS,
)
def bad_layers(image: Image) -> list[Layer]:
    return []


@block(
    name="bad_layer_items",
    category="test",
    description="Return a list that is not made of layers.",
    output=OutputKind.LAYERS,
)
def bad_layer_items(image: Image) -> list[Layer]:
    return [1, 2]  # type: ignore[list-item]


@block(
    name="too_many_layers",
    category="test",
    description="Return more than sixteen layers.",
    output=OutputKind.LAYERS,
)
def too_many_layers(image: Image) -> list[Layer]:
    return [Layer(image) for _ in range(17)]


SYNTHETIC = (
    drop_alpha,
    shrink,
    make_opaque,
    seed_probe,
    boom,
    bad_image,
    bad_mask,
    binary_mask,
    bad_layers,
    bad_layer_items,
    too_many_layers,
)


def make_snapshot() -> BlockSnapshot:
    definitions = dict(REGISTRY.defs)
    for function in SYNTHETIC:
        definition = function.__block_def__
        definitions[definition.name] = definition
    return BlockSnapshot(defs=MappingProxyType(definitions), hash="test")


SNAPSHOT = make_snapshot()


def experiment(seed: int = 0, params: dict[str, dict[str, object]] | None = None) -> ExperimentSpec:
    return ExperimentSpec(
        index=0,
        source_index=0,
        params=params or {},
        seed=seed,
        label="experiment",
    )


def source_rgb() -> Image:
    return Image(fixtures.rgb_gradient())


def source_rgba() -> Image:
    rgba = fixtures.rgba_gradient()
    return Image(rgba[..., :3], rgba[..., 3])


def test_chain_order() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="a", block="channel_permute", params={"order": "bgr"}),
            PipelineStep(id="b", block="channel_permute", params={"order": "rbg"}),
        )
    )
    source = source_rgb()
    expected = channel_permute(channel_permute(source, order="bgr"), order="rbg")
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    assert isinstance(output, PipelineOutput)
    assert len(output.layers) == 1
    assert np.array_equal(output.layers[0].image.rgb, expected.rgb)


def test_second_image_is_passed() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="overlay", params={"opacity": 1.0}),))
    source = source_rgb()
    second = Image(fixtures.checkerboard()[..., None].repeat(3, axis=-1))
    expected = overlay(source, second, opacity=1.0)
    output = execute_pipeline(pipeline, SNAPSHOT, source, second, experiment())
    assert np.array_equal(output.layers[0].image.rgb, expected.rgb)


def test_mask_side_input() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(
                id="mask",
                block="channel_mask",
                params={"channel": "R", "low": 0, "high": 128},
            ),
            PipelineStep(
                id="overlay",
                block="conditional_overlay",
                params={"threshold": 200},
                mask_bindings={"mask": "mask"},
            ),
        )
    )
    source = source_rgb()
    second = Image(fixtures.checkerboard()[..., None].repeat(3, axis=-1))
    mask = channel_mask(source, channel="R", low=0, high=128)
    expected = conditional_overlay(source, second, mask, threshold=200)
    output = execute_pipeline(pipeline, SNAPSHOT, source, second, experiment())
    assert np.array_equal(output.layers[0].image.rgb, expected.rgb)


def test_mask_does_not_advance_the_chain() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="mask", block="channel_mask"),
            PipelineStep(id="perm", block="channel_permute", params={"order": "bgr"}),
        )
    )
    source = source_rgb()
    expected = channel_permute(source, order="bgr")
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    assert np.array_equal(output.layers[0].image.rgb, expected.rgb)


def test_bypass_passes_the_source_through() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute", bypass=True),))
    source = source_rgb()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    assert output.layers[0].image is source


def test_empty_pipeline_returns_the_source() -> None:
    source = source_rgb()
    output = execute_pipeline(Pipeline(), SNAPSHOT, source, None, experiment())
    assert len(output.layers) == 1
    assert output.layers[0].image is source
    assert output.layers[0].mask is None


def test_layers_is_terminal() -> None:
    pipeline = Pipeline(
        steps=(
            PipelineStep(id="grid", block="permutation_grid"),
            PipelineStep(id="after", block="channel_permute"),
        )
    )
    source = source_rgb()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    expected = permutation_grid(source)
    assert len(output.layers) == len(expected) == 6
    assert [layer.name for layer in output.layers] == [layer.name for layer in expected]


def test_alpha_is_preserved_for_a_pixelwise_block() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="drop_alpha"),))
    source = source_rgba()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    assert output.layers[0].image.alpha is not None
    assert np.array_equal(output.layers[0].image.alpha, source.alpha)


def test_alpha_is_dropped_when_geometry_changes() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="shrink"),))
    source = source_rgba()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    image = output.layers[0].image
    assert image.size == (2, 2)
    assert image.alpha is None


def test_explicit_opaque_alpha_is_kept() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="make_opaque"),))
    source = source_rgba()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    alpha = output.layers[0].image.alpha
    assert alpha is not None
    assert np.all(alpha == 255)


def test_seed_is_derived_per_step() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="probe", block="seed_probe"),))
    output = execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment(seed=5))
    expected = derive_seed(5, "step", "probe") % 256
    assert np.all(output.layers[0].image.rgb == expected)


def test_seed_derivation_is_deterministic() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="probe", block="seed_probe"),))
    first = execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment(seed=5))
    second = execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment(seed=5))
    assert np.array_equal(first.layers[0].image.rgb, second.layers[0].image.rgb)


def test_different_steps_get_different_seeds() -> None:
    first = execute_pipeline(
        Pipeline(steps=(PipelineStep(id="a", block="seed_probe"),)),
        SNAPSHOT,
        source_rgb(),
        None,
        experiment(seed=5),
    )
    second = execute_pipeline(
        Pipeline(steps=(PipelineStep(id="b", block="seed_probe"),)),
        SNAPSHOT,
        source_rgb(),
        None,
        experiment(seed=5),
    )
    assert first.layers[0].image.rgb[0, 0, 0] != second.layers[0].image.rgb[0, 0, 0]


def test_explicit_pipeline_seed_wins() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="probe", block="seed_probe", params={"seed": 7}),))
    output = execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment(seed=5))
    assert np.all(output.layers[0].image.rgb == 7)


def test_experiment_seed_overrides_the_pipeline_value() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="probe", block="seed_probe", params={"seed": 7}),))
    output = execute_pipeline(
        pipeline,
        SNAPSHOT,
        source_rgb(),
        None,
        experiment(seed=5, params={"probe": {"seed": 9}}),
    )
    assert np.all(output.layers[0].image.rgb == 9)


def test_none_seed_uses_the_derived_value() -> None:
    pipeline = Pipeline(
        steps=(PipelineStep(id="probe", block="seed_probe", params={"seed": None}),)
    )
    output = execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment(seed=5))
    assert np.all(output.layers[0].image.rgb == derive_seed(5, "step", "probe") % 256)


def test_parameters_fall_back_to_block_defaults() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute"),))
    source = source_rgb()
    output = execute_pipeline(pipeline, SNAPSHOT, source, None, experiment())
    assert np.array_equal(output.layers[0].image.rgb, channel_permute(source).rgb)


def test_experiment_parameters_override_defaults() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="channel_permute"),))
    source = source_rgb()
    output = execute_pipeline(
        pipeline,
        SNAPSHOT,
        source,
        None,
        experiment(params={"a": {"order": "bgr"}}),
    )
    assert np.array_equal(output.layers[0].image.rgb, channel_permute(source, order="bgr").rgb)


def test_block_exception_propagates() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="boom"),))
    with pytest.raises(RuntimeError, match="boom"):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())


def test_invalid_image_output() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="bad_image"),))
    with pytest.raises(InvalidBlockOutput) as caught:
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())
    assert caught.value.block == "bad_image"
    assert caught.value.step_id == "a"


def test_invalid_mask_shape() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="bad_mask"),))
    with pytest.raises(InvalidBlockOutput):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())


def test_invalid_binary_mask_values() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="binary_mask"),))
    with pytest.raises(InvalidBlockOutput):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())


def test_invalid_layer_count() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="bad_layers"),))
    with pytest.raises(InvalidBlockOutput):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())


def test_invalid_layer_items() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="bad_layer_items"),))
    with pytest.raises(InvalidBlockOutput):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())


def test_too_many_layers() -> None:
    pipeline = Pipeline(steps=(PipelineStep(id="a", block="too_many_layers"),))
    with pytest.raises(InvalidBlockOutput):
        execute_pipeline(pipeline, SNAPSHOT, source_rgb(), None, experiment())

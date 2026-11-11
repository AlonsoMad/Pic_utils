"""Behavior tests for the block authoring API (P2-01, architecture §5.1-§5.3)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Annotated

import pytest

from pic_utils.blocks import (
    BLOCK_API_VERSION,
    BlockDef,
    BlockLoadError,
    Choice,
    Color,
    FloatRange,
    IntRange,
    OutputKind,
    ParamKind,
    Seed,
    block,
)
from pic_utils.core.errors import PicUtilsError
from pic_utils.core.image import Image, Layer, Mask


@block(
    name="all_kinds",
    category="test",
    description="Exercises every parameter kind.",
    output=OutputKind.IMAGE,
)
def all_kinds(
    image: Image,
    second: Image,
    mask: Mask,
    optional_mask: Mask | None = None,
    count: Annotated[int, IntRange(0, 10, step=2)] = 4,
    ratio: Annotated[float, FloatRange(0.0, 1.0, step=0.1)] = 0.5,
    mode: Annotated[str, Choice(("a", "b"))] = "a",
    level: Annotated[int, Choice((1, 2, 3))] = 2,
    gain: Annotated[float, Choice((0.5, 1.5))] = 1.5,
    enabled: bool = True,
    color: Annotated[str, Color()] = "#ff0000",
    seed: Annotated[int | None, Seed()] = None,
) -> Image:
    return image


@block(
    name="mask_block",
    category="test",
    description="Returns a mask.",
    output=OutputKind.MASK,
)
def mask_block(image: Image) -> Mask:
    return Mask(image.rgb[..., 0])


@block(
    name="layers_block",
    category="test",
    description="Returns layers.",
    output=OutputKind.LAYERS,
)
def layers_block(image: Image) -> list[Layer]:
    return [Layer(image)]


@block(
    name="tuple_layers_block",
    category="test",
    description="Returns a tuple of layers.",
    output=OutputKind.LAYERS,
)
def tuple_layers_block(image: Image) -> tuple[Layer, ...]:
    return (Layer(image),)


def definition_of(fn: object) -> BlockDef:
    definition = getattr(fn, "__block_def__", None)
    assert isinstance(definition, BlockDef)
    return definition


def test_decorator_builds_metadata() -> None:
    definition = definition_of(all_kinds)
    assert definition.name == "all_kinds"
    assert definition.category == "test"
    assert definition.description == "Exercises every parameter kind."
    assert definition.output is OutputKind.IMAGE
    assert definition.api_version == BLOCK_API_VERSION
    assert definition.module == __name__
    assert definition.source is None
    assert definition.content_hash == ""


def test_decorator_splits_image_and_mask_inputs() -> None:
    definition = definition_of(all_kinds)
    assert definition.image_inputs == ("image", "second")
    assert definition.mask_inputs == ("mask", "optional_mask")


def test_decorator_introspects_every_parameter_kind() -> None:
    params = {param.name: param for param in definition_of(all_kinds).params}
    assert params["image"].kind is ParamKind.IMAGE
    assert params["image"].required is True
    assert params["second"].kind is ParamKind.IMAGE
    assert params["mask"].kind is ParamKind.MASK
    assert params["mask"].required is True
    assert params["optional_mask"].kind is ParamKind.MASK
    assert params["optional_mask"].required is False
    assert params["optional_mask"].default is None
    assert params["count"].kind is ParamKind.INT
    assert params["count"].spec == IntRange(0, 10, step=2)
    assert params["count"].default == 4
    assert params["ratio"].kind is ParamKind.FLOAT
    assert params["ratio"].spec == FloatRange(0.0, 1.0, step=0.1)
    assert params["mode"].kind is ParamKind.CHOICE
    assert params["mode"].spec == Choice(("a", "b"))
    assert params["level"].kind is ParamKind.CHOICE
    assert params["gain"].kind is ParamKind.CHOICE
    assert params["enabled"].kind is ParamKind.BOOL
    assert params["enabled"].default is True
    assert params["color"].kind is ParamKind.COLOR
    assert params["color"].default == "#ff0000"
    assert params["seed"].kind is ParamKind.SEED
    assert params["seed"].default is None


def test_parameter_order_matches_the_signature() -> None:
    names = [param.name for param in definition_of(all_kinds).params]
    assert names == [
        "image",
        "second",
        "mask",
        "optional_mask",
        "count",
        "ratio",
        "mode",
        "level",
        "gain",
        "enabled",
        "color",
        "seed",
    ]


def test_mask_and_layers_return_annotations_are_accepted() -> None:
    assert definition_of(mask_block).output is OutputKind.MASK
    assert definition_of(layers_block).output is OutputKind.LAYERS
    assert definition_of(tuple_layers_block).output is OutputKind.LAYERS


def test_block_def_is_immutable() -> None:
    definition = definition_of(all_kinds)
    with pytest.raises(dataclasses.FrozenInstanceError):
        definition.name = "other"  # type: ignore[misc]


def test_block_load_error_is_a_pic_utils_error() -> None:
    error = BlockLoadError(reason="bad signature", path=Path("blocks/bad.py"), block="bad")
    assert isinstance(error, PicUtilsError)
    assert "blocks/bad.py" in str(error)
    assert "bad" in str(error)
    assert "bad signature" in str(error)


def test_decorator_rejects_a_non_function() -> None:
    with pytest.raises(TypeError, match="must decorate a function"):
        block(
            name="not_a_function",
            category="test",
            description="x",
            output=OutputKind.IMAGE,
        )(object())  # type: ignore[arg-type]


def test_decorator_rejects_empty_metadata() -> None:
    with pytest.raises(ValueError, match="name"):
        block(name="", category="test", description="x", output=OutputKind.IMAGE)
    with pytest.raises(ValueError, match="category"):
        block(name="x", category="", description="x", output=OutputKind.IMAGE)
    with pytest.raises(ValueError, match="description"):
        block(name="x", category="test", description="", output=OutputKind.IMAGE)


def test_decorator_rejects_non_output_kind() -> None:
    with pytest.raises(TypeError, match="OutputKind"):
        block(
            name="x",
            category="test",
            description="x",
            output="image",  # type: ignore[arg-type]
        )


def test_missing_scalar_default_is_rejected() -> None:
    def invalid(image: Image, count: Annotated[int, IntRange(0, 10)]) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="must have a default"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_varargs_and_kwargs_are_rejected() -> None:
    def varargs(image: Image, *args: int) -> Image:
        return image

    def kwargs(image: Image, **kwargs: int) -> Image:
        return image

    for fn in (varargs, kwargs):
        with pytest.raises(BlockLoadError, match=r"\*args/\*\*kwargs"):
            block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(fn)


def test_positional_only_is_rejected() -> None:
    def invalid(image: Image, /, count: Annotated[int, IntRange(0, 10)] = 0) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="positional-only"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_missing_annotation_is_rejected() -> None:
    def invalid(image: Image, count=0) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="needs a type annotation"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_bare_scalar_annotations_are_rejected() -> None:
    def bare_int(image: Image, count: int = 0) -> Image:
        return image

    def bare_str(image: Image, text: str = "") -> Image:
        return image

    for fn in (bare_int, bare_str):
        with pytest.raises(BlockLoadError, match="needs"):
            block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(fn)


def test_image_input_must_not_have_a_default() -> None:
    def invalid(image: Image = None) -> Image:  # type: ignore[assignment]
        return image

    with pytest.raises(BlockLoadError, match="must not have a default"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_optional_mask_must_default_to_none() -> None:
    def invalid(image: Image, mask: Mask | None = 1) -> Image:  # type: ignore[assignment]
        return image

    with pytest.raises(BlockLoadError, match="must default to None"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_optional_int_needs_seed_metadata() -> None:
    def invalid(image: Image, seed: int | None = None) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="Seed"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_metadata_must_match_the_annotation() -> None:
    def invalid(image: Image, ratio: Annotated[int, FloatRange(0.0, 1.0)] = 0) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="IntRange or Choice"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_choice_options_must_match_the_annotation() -> None:
    def invalid(image: Image, mode: Annotated[str, Choice((1, 2))] = 1) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="does not match"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_default_must_be_inside_the_range() -> None:
    def invalid(image: Image, count: Annotated[int, IntRange(0, 10)] = 20) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="outside its range"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_color_default_must_be_a_hex_string() -> None:
    def invalid(image: Image, color: Annotated[str, Color()] = "red") -> Image:
        return image

    with pytest.raises(BlockLoadError, match="not a hex color"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_seed_default_must_be_none_or_int() -> None:
    def invalid(image: Image, seed: Annotated[int | None, Seed()] = "x") -> Image:
        return image

    with pytest.raises(BlockLoadError, match="None or an int"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_return_annotation_must_match_the_output_kind() -> None:
    def wrong_image(image: Image) -> Mask:
        return Mask(image.rgb[..., 0])

    def wrong_mask(image: Image) -> Image:
        return image

    def wrong_layers(image: Image) -> Image:
        return image

    with pytest.raises(BlockLoadError, match="output is IMAGE"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(wrong_image)
    with pytest.raises(BlockLoadError, match="output is MASK"):
        block(name="bad", category="test", description="x", output=OutputKind.MASK)(wrong_mask)
    with pytest.raises(BlockLoadError, match="output is LAYERS"):
        block(name="bad", category="test", description="x", output=OutputKind.LAYERS)(wrong_layers)


def test_missing_return_annotation_is_rejected() -> None:
    def invalid(image: Image):
        return image

    with pytest.raises(BlockLoadError, match="missing return annotation"):
        block(name="bad", category="test", description="x", output=OutputKind.IMAGE)(invalid)


def test_range_metadata_validates_itself() -> None:
    with pytest.raises(ValueError, match="min"):
        IntRange(10, 0)
    with pytest.raises(ValueError, match="step"):
        IntRange(0, 10, step=0)
    with pytest.raises(ValueError, match="min"):
        FloatRange(1.0, 0.0)
    with pytest.raises(ValueError, match="step"):
        FloatRange(0.0, 1.0, step=0.0)


def test_choice_metadata_validates_itself() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Choice(())
    with pytest.raises(ValueError, match="unique"):
        Choice(("a", "a"))
    with pytest.raises(ValueError, match="labels"):
        Choice(("a", "b"), labels=("A",))

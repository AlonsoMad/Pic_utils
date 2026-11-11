"""Conformance tests for every registered built-in block (P2-07)."""

from __future__ import annotations

import re
from pathlib import Path

import fixtures
import numpy as np
import pytest

from pic_utils.blocks import Choice, Color, FloatRange, IntRange, OutputKind, ParamKind, Seed
from pic_utils.blocks.registry import BlockRegistry
from pic_utils.core.color import parse_hex_color
from pic_utils.core.image import Image, Layer, Mask

SNAPSHOT = BlockRegistry().snapshot()
BLOCK_NAMES = sorted(SNAPSHOT)
BUILTIN_DIR = Path(__file__).resolve().parents[2] / "src" / "pic_utils" / "blocks" / "builtin"


def call_block(definition, image: Image, second: Image, mask: Mask):
    kwargs = {}
    for param in definition.params:
        if param.kind is ParamKind.IMAGE:
            kwargs[param.name] = image if param.name == definition.image_inputs[0] else second
        elif param.kind is ParamKind.MASK:
            kwargs[param.name] = mask
        elif param.kind is ParamKind.SEED:
            kwargs[param.name] = 0
        else:
            kwargs[param.name] = param.default
    return definition.fn(**kwargs)


def assert_output(definition, result: object, size: tuple[int, int]) -> None:
    if definition.output is OutputKind.IMAGE:
        assert isinstance(result, Image)
        assert result.size == size
    elif definition.output is OutputKind.MASK:
        assert isinstance(result, Mask)
        assert result.data.shape == size
    else:
        assert isinstance(result, list | tuple)
        assert 1 <= len(result) <= 16
        for layer in result:
            assert isinstance(layer, Layer)
            assert layer.image.width >= 1
            assert layer.image.height >= 1
            if layer.mask is not None:
                assert layer.mask.data.shape == layer.image.rgb.shape[:2]


def assert_same(left: object, right: object) -> None:
    if isinstance(left, Image):
        assert isinstance(right, Image)
        assert np.array_equal(left.rgb, right.rgb)
        assert (left.alpha is None) == (right.alpha is None)
        if left.alpha is not None and right.alpha is not None:
            assert np.array_equal(left.alpha, right.alpha)
    elif isinstance(left, Mask):
        assert isinstance(right, Mask)
        assert left.soft == right.soft
        assert np.array_equal(left.data, right.data)
    else:
        assert isinstance(left, list | tuple)
        assert isinstance(right, list | tuple)
        assert len(left) == len(right)
        for left_layer, right_layer in zip(left, right, strict=True):
            assert left_layer.name == right_layer.name
            assert_same(left_layer.image, right_layer.image)
            if left_layer.mask is not None:
                assert right_layer.mask is not None
                assert_same(left_layer.mask, right_layer.mask)


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_block_metadata(name: str) -> None:
    definition = SNAPSHOT.get(name)
    assert definition.name == name
    assert definition.category
    assert definition.description
    assert isinstance(definition.output, OutputKind)
    assert definition.api_version == 1
    assert definition.source is None
    assert definition.content_hash == ""
    assert definition.module.startswith("pic_utils.blocks.builtin.")
    param_names = {param.name for param in definition.params}
    assert set(definition.image_inputs) <= param_names
    assert set(definition.mask_inputs) <= param_names


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_block_defaults_are_in_range(name: str) -> None:
    definition = SNAPSHOT.get(name)
    for param in definition.params:
        if param.kind is ParamKind.IMAGE:
            assert param.required
            assert param.default is None
        elif param.kind is ParamKind.MASK:
            if not param.required:
                assert param.default is None
        elif param.kind is ParamKind.INT:
            assert isinstance(param.spec, IntRange)
            assert isinstance(param.default, int) and not isinstance(param.default, bool)
            assert param.spec.min <= param.default <= param.spec.max
        elif param.kind is ParamKind.FLOAT:
            assert isinstance(param.spec, FloatRange)
            assert isinstance(param.default, int | float)
            assert not isinstance(param.default, bool)
            assert param.spec.min <= param.default <= param.spec.max
        elif param.kind is ParamKind.CHOICE:
            assert isinstance(param.spec, Choice)
            assert param.default in param.spec.options
        elif param.kind is ParamKind.BOOL:
            assert isinstance(param.default, bool)
        elif param.kind is ParamKind.COLOR:
            assert isinstance(param.spec, Color)
            assert isinstance(param.default, str)
            parse_hex_color(param.default)
        elif param.kind is ParamKind.SEED:
            assert isinstance(param.spec, Seed)
            assert param.default is None or (
                isinstance(param.default, int) and not isinstance(param.default, bool)
            )


@pytest.mark.parametrize("name", BLOCK_NAMES)
def test_block_output_type_and_determinism(name: str) -> None:
    definition = SNAPSHOT.get(name)
    image = Image.from_rgb(fixtures.rgb_gradient(32))
    second = Image.from_rgba(fixtures.rgba_gradient(32))
    mask = Mask(fixtures.checkerboard(32))

    first = call_block(definition, image, second, mask)
    second_call = call_block(definition, image, second, mask)
    assert_output(definition, first, (32, 32))
    assert_same(first, second_call)


@pytest.mark.parametrize("module_name", ["channels", "masks", "compose", "segment"])
def test_builtin_modules_have_no_forbidden_side_effects(module_name: str) -> None:
    source = (BUILTIN_DIR / f"{module_name}.py").read_text(encoding="utf-8")
    for forbidden in (
        "pic_utils.io",
        "pic_utils.ui",
        "pic_utils.runner",
        "pic_utils.canvas",
    ):
        assert f"import {forbidden}" not in source
        assert f"from {forbidden}" not in source
    assert "print(" not in source
    assert "open(" not in source
    assert "cv2.imread" not in source
    assert "cv2.imwrite" not in source
    assert "PIL" not in source
    assert "plt." not in source
    assert re.search(r"np\.random\.(?!default_rng)", source) is None


def test_segment_uses_bincount_for_superpixel_means() -> None:
    source = (BUILTIN_DIR / "segment.py").read_text(encoding="utf-8")
    assert "np.bincount" in source
    assert "np.unique(segments)" not in source

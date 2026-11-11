"""Behavior tests for the channel blocks (P2-03, architecture §5.7)."""

from __future__ import annotations

import fixtures
import numpy as np
import pytest

from pic_utils.blocks.builtin.channels import channel_permute, permutation_grid
from pic_utils.core.image import Image


@pytest.fixture
def image() -> Image:
    return Image.from_rgba(fixtures.rgba_gradient())


def test_channel_permute_identity_preserves_pixels_and_alpha(image: Image) -> None:
    result = channel_permute(image, order="rgb")
    assert np.array_equal(result.rgb, image.rgb)
    assert result.alpha is not None
    assert image.alpha is not None
    assert np.array_equal(result.alpha, image.alpha)


def test_channel_permute_swaps_channels(image: Image) -> None:
    bgr = channel_permute(image, order="bgr")
    assert np.array_equal(bgr.rgb[..., 0], image.rgb[..., 2])
    assert np.array_equal(bgr.rgb[..., 1], image.rgb[..., 1])
    assert np.array_equal(bgr.rgb[..., 2], image.rgb[..., 0])

    rbg = channel_permute(image, order="rbg")
    assert np.array_equal(rbg.rgb[..., 0], image.rgb[..., 0])
    assert np.array_equal(rbg.rgb[..., 1], image.rgb[..., 2])
    assert np.array_equal(rbg.rgb[..., 2], image.rgb[..., 1])


def test_channel_permute_inverts_selected_output_channels(image: Image) -> None:
    result = channel_permute(image, order="rgb", invert_r=True)
    assert np.array_equal(result.rgb[..., 0], 255 - image.rgb[..., 0])
    assert np.array_equal(result.rgb[..., 1], image.rgb[..., 1])
    assert np.array_equal(result.rgb[..., 2], image.rgb[..., 2])


def test_channel_permute_inverts_after_reordering(image: Image) -> None:
    result = channel_permute(image, order="bgr", invert_r=True)
    assert np.array_equal(result.rgb[..., 0], 255 - image.rgb[..., 2])


def test_channel_permute_rejects_an_invalid_order(image: Image) -> None:
    with pytest.raises(ValueError, match="invalid channel order"):
        channel_permute(image, order="rrr")


def test_permutation_grid_channel_order_has_six_layers(image: Image) -> None:
    layers = permutation_grid(image, mode="channel_order")
    assert len(layers) == 6
    assert [layer.name for layer in layers] == ["rgb", "rbg", "grb", "gbr", "brg", "bgr"]
    for layer in layers:
        assert layer.image.alpha is not None
        assert np.array_equal(layer.image.alpha, image.alpha)


def test_permutation_grid_include_original_adds_a_seventh_layer(image: Image) -> None:
    layers = permutation_grid(image, mode="channel_order", include_original=True)
    assert len(layers) == 7
    assert layers[0].name == "original"
    assert np.array_equal(layers[0].image.rgb, image.rgb)


def test_permutation_grid_invert_has_eight_layers(image: Image) -> None:
    layers = permutation_grid(image, mode="invert")
    assert len(layers) == 8
    assert layers[0].name == "invert_original"
    assert np.array_equal(layers[0].image.rgb, image.rgb)
    assert layers[7].name == "invert_rgb"
    assert np.array_equal(layers[7].image.rgb, 255 - image.rgb)


def test_permutation_grid_is_deterministic(image: Image) -> None:
    first = permutation_grid(image, mode="invert")
    second = permutation_grid(image, mode="invert")
    assert [layer.name for layer in first] == [layer.name for layer in second]
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.image.rgb, right.image.rgb)


def test_permutation_grid_rejects_an_invalid_mode(image: Image) -> None:
    with pytest.raises(ValueError, match="invalid permutation mode"):
        permutation_grid(image, mode="other")

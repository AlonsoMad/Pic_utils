"""Behavior tests for the mask blocks (P2-04, architecture §5.7)."""

from __future__ import annotations

import fixtures
import numpy as np
import pytest

from pic_utils.blocks.builtin.masks import brightness_mask, channel_mask, color_range_mask
from pic_utils.core.image import Image

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


@pytest.fixture
def image() -> Image:
    pixels = np.array([[RED, GREEN], [BLUE, WHITE]], dtype=np.uint8)
    return Image.from_rgb(pixels)


def test_channel_mask_selects_each_channel(image: Image) -> None:
    red = channel_mask(image, channel="R", low=200, high=255)
    assert np.array_equal(red.data, [[255, 0], [0, 255]])

    green = channel_mask(image, channel="G", low=200, high=255)
    assert np.array_equal(green.data, [[0, 255], [0, 255]])

    blue = channel_mask(image, channel="B", low=200, high=255)
    assert np.array_equal(blue.data, [[0, 0], [255, 255]])


def test_channel_mask_luminance(image: Image) -> None:
    mask = channel_mask(image, channel="L", low=100, high=255)
    assert np.array_equal(mask.data, [[0, 255], [0, 255]])


def test_channel_mask_soft_carries_the_channel_value() -> None:
    pixels = np.array([[[0, 0, 0], [100, 0, 0]], [[200, 0, 0], [255, 0, 0]]], dtype=np.uint8)
    mask = channel_mask(Image.from_rgb(pixels), channel="R", low=100, high=200, soft=True)
    assert mask.soft is True
    assert np.array_equal(mask.data, [[0, 100], [200, 0]])


def test_channel_mask_invert_and_empty_range(image: Image) -> None:
    inverted = channel_mask(image, channel="R", low=200, high=255, invert=True)
    assert np.array_equal(inverted.data, [[0, 255], [255, 0]])
    empty = channel_mask(image, channel="R", low=200, high=100)
    assert np.array_equal(empty.data, np.zeros((2, 2), dtype=np.uint8))


def test_channel_mask_rejects_an_invalid_channel(image: Image) -> None:
    with pytest.raises(ValueError, match="invalid channel"):
        channel_mask(image, channel="X")


def test_brightness_mask_selects_dark_pixels(image: Image) -> None:
    mask = brightness_mask(image, threshold=100)
    assert np.array_equal(mask.data, [[255, 0], [255, 0]])


def test_brightness_mask_soft_carries_luminance(image: Image) -> None:
    mask = brightness_mask(image, threshold=100, soft=True)
    assert mask.soft is True
    assert np.array_equal(mask.data, [[54, 0], [18, 0]])


def test_brightness_mask_auto_uses_the_median() -> None:
    image = Image.from_rgb(np.dstack([fixtures.mask_gradient(16)] * 3))
    mask = brightness_mask(image, auto=True)
    assert int(mask.data.sum() // 255) == 128


def test_brightness_mask_invert(image: Image) -> None:
    mask = brightness_mask(image, threshold=100, invert=True)
    assert np.array_equal(mask.data, [[0, 255], [0, 255]])


def test_color_range_mask_strict_requires_all_channels(image: Image) -> None:
    mask = color_range_mask(image, r_low=200, g_low=0, g_high=0, b_low=0, b_high=0)
    assert np.array_equal(mask.data, [[255, 0], [0, 0]])


def test_color_range_mask_loose_matches_any_channel(image: Image) -> None:
    mask = color_range_mask(
        image,
        r_low=200,
        g_low=0,
        g_high=0,
        b_low=0,
        b_high=0,
        strict=False,
    )
    assert np.array_equal(mask.data, [[255, 255], [255, 255]])


def test_color_range_mask_soft_carries_mean_rgb(image: Image) -> None:
    mask = color_range_mask(
        image,
        r_low=200,
        g_low=0,
        g_high=0,
        b_low=0,
        b_high=0,
        soft=True,
    )
    assert mask.soft is True
    assert np.array_equal(mask.data, [[85, 0], [0, 0]])


def test_color_range_mask_invert(image: Image) -> None:
    mask = color_range_mask(
        image,
        r_low=200,
        g_low=0,
        g_high=0,
        b_low=0,
        b_high=0,
        invert=True,
    )
    assert np.array_equal(mask.data, [[0, 255], [255, 255]])


def test_masks_are_read_only(image: Image) -> None:
    mask = brightness_mask(image, threshold=100)
    assert not mask.data.flags.writeable
    assert mask.data.flags.c_contiguous

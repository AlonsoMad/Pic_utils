"""Behavior tests for the compositing blocks (P2-05, architecture §5.7)."""

from __future__ import annotations

import numpy as np
import pytest

from pic_utils.blocks.builtin.compose import conditional_overlay, double_exposure, overlay
from pic_utils.core.image import Image, Mask


def solid(color: tuple[int, int, int], size: tuple[int, int] = (4, 4)) -> Image:
    rgb = np.empty((*size, 3), dtype=np.uint8)
    rgb[:] = color
    return Image.from_rgb(rgb)


def test_overlay_places_the_second_image_at_the_offset() -> None:
    background = solid((255, 0, 0))
    second = solid((0, 0, 255), (2, 2))
    result = overlay(
        background,
        second,
        white_to_transparent=False,
        resize_to_match=False,
    )
    assert result.alpha is None
    assert np.array_equal(result.rgb[:2, :2], np.full((2, 2, 3), (0, 0, 255), np.uint8))
    assert np.array_equal(result.rgb[2:, :], np.full((2, 4, 3), (255, 0, 0), np.uint8))


def test_overlay_white_to_transparent_applies_to_the_overlay() -> None:
    background = Image.from_rgb(
        np.array(
            [[[255, 0, 0]] * 4] * 3 + [[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255]]],
            dtype=np.uint8,
        )
    )
    second = solid((255, 255, 255), (2, 2))
    result = overlay(
        background,
        second,
        white_to_transparent=True,
        resize_to_match=False,
    )
    assert np.array_equal(result.rgb[:2, :2], np.full((2, 2, 3), (255, 0, 0), np.uint8))
    assert np.array_equal(result.rgb[3, 3], [255, 255, 255])


def test_overlay_under_hides_behind_an_opaque_background() -> None:
    result = overlay(
        solid((255, 0, 0)),
        solid((0, 0, 255), (2, 2)),
        z_order="under",
        white_to_transparent=False,
        resize_to_match=False,
    )
    assert np.array_equal(result.rgb, np.full((4, 4, 3), (255, 0, 0), np.uint8))


def test_overlay_under_shows_through_transparent_background() -> None:
    rgb = np.full((4, 4, 3), (255, 0, 0), np.uint8)
    alpha = np.full((4, 4), 255, np.uint8)
    alpha[:2, :2] = 0
    background = Image(rgb, alpha)
    result = overlay(
        background,
        solid((0, 0, 255), (2, 2)),
        z_order="under",
        white_to_transparent=False,
        resize_to_match=False,
    )
    assert np.array_equal(result.rgb[:2, :2], np.full((2, 2, 3), (0, 0, 255), np.uint8))
    assert np.array_equal(result.rgb[2:, :], np.full((2, 4, 3), (255, 0, 0), np.uint8))


def test_overlay_opacity_blends() -> None:
    result = overlay(
        solid((255, 0, 0)),
        solid((0, 0, 255), (2, 2)),
        white_to_transparent=False,
        resize_to_match=False,
        opacity=0.5,
    )
    assert np.allclose(result.rgb[0, 0], [127, 0, 128], atol=1)


def test_overlay_resize_to_match_scales_the_overlay() -> None:
    result = overlay(
        solid((255, 0, 0)),
        solid((0, 0, 255), (2, 2)),
        white_to_transparent=False,
        resize_to_match=True,
    )
    assert np.array_equal(result.rgb, np.full((4, 4, 3), (0, 0, 255), np.uint8))


def test_overlay_negative_offset_crops() -> None:
    result = overlay(
        solid((255, 0, 0)),
        solid((0, 0, 255), (2, 2)),
        x=-1,
        y=-1,
        white_to_transparent=False,
        resize_to_match=False,
    )
    assert np.array_equal(result.rgb[0, 0], [0, 0, 255])
    assert np.array_equal(result.rgb[0, 1], [255, 0, 0])


def test_overlay_preserves_background_alpha() -> None:
    rgb = np.full((4, 4, 3), (255, 0, 0), np.uint8)
    alpha = np.full((4, 4), 128, np.uint8)
    result = overlay(
        Image(rgb, alpha),
        solid((0, 0, 255), (2, 2)),
        white_to_transparent=False,
        resize_to_match=False,
    )
    assert result.alpha is not None
    assert result.alpha[0, 0] == 255
    assert result.alpha[3, 3] == 128


def test_overlay_rejects_an_invalid_z_order() -> None:
    with pytest.raises(ValueError, match="invalid z_order"):
        overlay(solid((255, 0, 0)), solid((0, 0, 255), (2, 2)), z_order="sideways")


def test_double_exposure_blends_and_clips() -> None:
    black = solid((0, 0, 0))
    grey = solid((200, 200, 200))
    halfway = double_exposure(black, grey, exposure_factor=0.5)
    assert np.array_equal(halfway.rgb, np.full((4, 4, 3), 100, np.uint8))

    assert np.array_equal(double_exposure(black, grey, exposure_factor=0.0).rgb, black.rgb)
    assert np.array_equal(double_exposure(black, grey, exposure_factor=1.0).rgb, grey.rgb)
    clipped = double_exposure(solid((255, 255, 255)), black, exposure_factor=2.0)
    assert np.array_equal(clipped.rgb, np.zeros((4, 4, 3), np.uint8))


def test_double_exposure_pads_a_smaller_second_image() -> None:
    first = solid((255, 0, 0))
    second = solid((0, 0, 255), (2, 2))
    result = double_exposure(first, second, exposure_factor=0.5, resize_to_match=False)
    assert result.size == (4, 4)
    assert np.allclose(result.rgb[0, 0], [128, 0, 0], atol=1)
    assert np.allclose(result.rgb[1, 1], [127, 0, 128], atol=1)
    assert np.allclose(result.rgb[3, 3], [128, 0, 0], atol=1)


def test_double_exposure_alpha_precedence() -> None:
    rgb = np.full((4, 4, 3), (255, 0, 0), np.uint8)
    alpha = np.full((4, 4), 128, np.uint8)
    first_with_alpha = Image(rgb, alpha)
    second = solid((0, 0, 255))
    result = double_exposure(first_with_alpha, second)
    assert result.alpha is not None
    assert np.all(result.alpha == 128)

    second_with_alpha = Image(np.full((4, 4, 3), (0, 0, 255), np.uint8), alpha)
    result = double_exposure(solid((255, 0, 0)), second_with_alpha)
    assert result.alpha is not None
    assert np.all(result.alpha == 128)


def test_conditional_overlay_mask_gates_the_overlay() -> None:
    base = solid((255, 0, 0))
    second = solid((0, 0, 255))
    full = Mask(np.full((4, 4), 255, np.uint8))
    empty = Mask(np.zeros((4, 4), np.uint8))
    assert np.array_equal(conditional_overlay(base, second, full).rgb, second.rgb)
    assert np.array_equal(conditional_overlay(base, second, empty).rgb, base.rgb)


def test_conditional_overlay_mask_can_be_partial() -> None:
    base = solid((255, 0, 0))
    second = solid((0, 0, 255))
    data = np.zeros((4, 4), np.uint8)
    data[:, :2] = 255
    result = conditional_overlay(base, second, Mask(data))
    assert np.array_equal(result.rgb[:, :2], np.full((4, 2, 3), (0, 0, 255), np.uint8))
    assert np.array_equal(result.rgb[:, 2:], np.full((4, 2, 3), (255, 0, 0), np.uint8))


def test_conditional_overlay_threshold_restricts_dark_pixels() -> None:
    pixels = np.array([[[50, 50, 50], [200, 200, 200]]], dtype=np.uint8)
    base = Image.from_rgb(pixels)
    second = solid((0, 0, 255), (1, 2))
    mask = Mask(np.full((1, 2), 255, np.uint8))
    result = conditional_overlay(base, second, mask, threshold=100)
    assert np.array_equal(result.rgb[0, 0], [0, 0, 255])
    assert np.array_equal(result.rgb[0, 1], [200, 200, 200])


def test_conditional_overlay_color_ranges_restrict() -> None:
    pixels = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    base = Image.from_rgb(pixels)
    second = solid((0, 0, 255), (1, 2))
    mask = Mask(np.full((1, 2), 255, np.uint8))
    result = conditional_overlay(
        base,
        second,
        mask,
        r_low=200,
        g_low=0,
        g_high=0,
        b_low=0,
        b_high=0,
    )
    assert np.array_equal(result.rgb[0, 0], [0, 0, 255])
    assert np.array_equal(result.rgb[0, 1], [0, 255, 0])


def test_conditional_overlay_soft_mask_scales_alpha() -> None:
    base = solid((255, 0, 0))
    second = solid((0, 0, 255))
    mask = Mask(np.full((4, 4), 128, np.uint8), soft=True)
    result = conditional_overlay(base, second, mask)
    assert np.allclose(result.rgb[0, 0], [127, 0, 128], atol=1)


def test_conditional_overlay_pads_a_smaller_second_image() -> None:
    base = solid((255, 0, 0))
    second = solid((0, 0, 255), (2, 2))
    mask = Mask(np.full((4, 4), 255, np.uint8))
    result = conditional_overlay(base, second, mask)
    assert result.size == (4, 4)
    assert np.array_equal(result.rgb[1:3, 1:3], np.full((2, 2, 3), (0, 0, 255), np.uint8))
    assert np.array_equal(result.rgb[0, :], np.full((4, 3), (255, 0, 0), np.uint8))
    assert np.array_equal(result.rgb[3, :], np.full((4, 3), (255, 0, 0), np.uint8))


def test_conditional_overlay_rejects_a_mismatched_mask() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        conditional_overlay(
            solid((255, 0, 0)),
            solid((0, 0, 255)),
            Mask(np.zeros((2, 2), np.uint8)),
        )

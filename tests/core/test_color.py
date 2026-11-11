"""Behavior tests for color parsing, luminance, and the BGR boundary (P1-02)."""

from __future__ import annotations

import numpy as np
import pytest

from pic_utils.core.color import from_bgr, luminance, parse_hex_color, to_bgr, to_hex_color


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#ff0000", (255, 0, 0)),
        ("00ff00", (0, 255, 0)),
        ("  #0000FF  ", (0, 0, 255)),
        ("#abc", (170, 187, 204)),
        ("AbCdEf", (171, 205, 239)),
    ],
)
def test_parse_hex_color(text: str, expected: tuple[int, int, int]) -> None:
    assert parse_hex_color(text) == expected


@pytest.mark.parametrize("text", ["", "#", "#ff00", "#gg0000", "red", "#1234567", "#12 34 56"])
def test_parse_hex_color_rejects_invalid(text: str) -> None:
    with pytest.raises(ValueError, match="invalid hex color"):
        parse_hex_color(text)


def test_parse_hex_color_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        parse_hex_color(123)  # type: ignore[arg-type]


def test_to_hex_color_round_trips() -> None:
    assert to_hex_color((255, 0, 0)) == "#ff0000"
    assert to_hex_color(np.array([0, 255, 0], dtype=np.uint8)) == "#00ff00"
    assert to_hex_color(parse_hex_color("#AbCdEf")) == "#abcdef"


def test_to_hex_color_rejects_bad_components() -> None:
    with pytest.raises(ValueError, match="three color components"):
        to_hex_color((1, 2))
    with pytest.raises(ValueError, match=r"0\.\.255"):
        to_hex_color((256, 0, 0))
    with pytest.raises(TypeError, match="integers"):
        to_hex_color((1.0, 2.0, 3.0))


def test_luminance_uses_rec_709_weights() -> None:
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    values = luminance(pixels)
    assert values.shape == (1, 5)
    assert values.dtype == np.float32
    np.testing.assert_allclose(
        values[0],
        [0.2126 * 255, 0.7152 * 255, 0.0722 * 255, 255.0, 0.0],
        atol=0.01,
    )


def test_luminance_rejects_bad_shape_and_dtype() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        luminance(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="unsupported image dtype"):
        luminance(np.zeros((4, 4, 3), dtype=np.int16))


def test_bgr_bridge_swaps_channels_and_round_trips() -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    bgr = to_bgr(rgb)
    assert np.array_equal(bgr, np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8))
    assert np.array_equal(from_bgr(bgr), rgb)
    assert bgr is not rgb


def test_bgr_bridge_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        to_bgr(np.zeros((4, 4), dtype=np.uint8))

"""Behavior tests for channel percentiles (P1-02, architecture §5.7)."""

from __future__ import annotations

import numpy as np
import pytest

from pic_utils.core.stats import find_channel_percentiles


def test_percentiles_match_numpy_inverted_cdf() -> None:
    rng = np.random.default_rng(20261111)
    image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    for percentile in (0, 1, 10, 50, 90, 95, 99, 100):
        result = find_channel_percentiles(image, percentile)
        for channel in range(3):
            expected = int(np.percentile(image[..., channel], percentile, method="inverted_cdf"))
            assert result[channel] == expected


def test_grayscale_image_returns_one_channel() -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert find_channel_percentiles(image, 50) == {0: 127}
    assert find_channel_percentiles(image, 0) == {0: 0}
    assert find_channel_percentiles(image, 100) == {0: 255}


def test_percentile_zero_returns_the_channel_minimum() -> None:
    image = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert find_channel_percentiles(image, 0) == {0: 200, 1: 200, 2: 200}


def test_non_uint8_integer_input_is_accepted() -> None:
    image = np.array([[[0, 10], [20, 30]]], dtype=np.int16)
    result = find_channel_percentiles(image, 50)
    assert set(result) == {0, 1}


def test_percentiles_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="2-D or 3-D"):
        find_channel_percentiles(np.zeros((2, 2, 2, 2), dtype=np.uint8), 50)
    with pytest.raises(ValueError, match="empty"):
        find_channel_percentiles(np.zeros((0, 0, 3), dtype=np.uint8), 50)
    with pytest.raises(TypeError, match="integer dtype"):
        find_channel_percentiles(np.zeros((4, 4, 3), dtype=np.float32), 50)
    with pytest.raises(ValueError, match=r"0\.\.255"):
        find_channel_percentiles(np.full((4, 4, 3), 300, dtype=np.int16), 50)
    with pytest.raises(ValueError, match="percentile"):
        find_channel_percentiles(np.zeros((4, 4, 3), dtype=np.uint8), 101)
    with pytest.raises(ValueError, match="percentile"):
        find_channel_percentiles(np.zeros((4, 4, 3), dtype=np.uint8), -1)

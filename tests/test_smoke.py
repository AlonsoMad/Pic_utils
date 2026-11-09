"""Smoke tests: package metadata and synthetic fixture determinism."""

from __future__ import annotations

import importlib.metadata

import fixtures
import numpy as np

import pic_utils


def test_package_version_matches_distribution_metadata() -> None:
    assert pic_utils.__version__ == importlib.metadata.version("pic-utils")
    assert pic_utils.__all__ == ["__version__"]


def test_rgb_gradient_is_deterministic(rgb_gradient) -> None:
    assert rgb_gradient.shape == (8, 8, 3)
    assert rgb_gradient.dtype == np.uint8
    assert np.array_equal(rgb_gradient, fixtures.rgb_gradient())


def test_checkerboard_is_deterministic(checkerboard) -> None:
    assert checkerboard.shape == (8, 8)
    assert checkerboard.dtype == np.uint8
    assert np.array_equal(checkerboard, fixtures.checkerboard())


def test_mask_gradient_is_deterministic(mask_gradient) -> None:
    assert mask_gradient.shape == (8, 8)
    assert mask_gradient.dtype == np.uint8
    assert np.array_equal(mask_gradient, fixtures.mask_gradient())


def test_rgba_gradient_is_deterministic(rgba_gradient) -> None:
    assert rgba_gradient.shape == (8, 8, 4)
    assert rgba_gradient.dtype == np.uint8
    assert np.array_equal(rgba_gradient, fixtures.rgba_gradient())


def test_helpers_honor_explicit_size() -> None:
    assert fixtures.rgb_gradient(4).shape == (4, 4, 3)
    assert fixtures.checkerboard(4).shape == (4, 4)
    assert fixtures.mask_gradient(4).shape == (4, 4)
    assert fixtures.rgba_gradient(4).shape == (4, 4, 4)

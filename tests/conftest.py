"""Shared pytest fixtures for the Pic Utils test suite.

``import fixtures`` works because pytest's rootdir-relative "prepend"
import mode puts ``tests/`` on ``sys.path`` (no ``tests/__init__.py``).
"""

from __future__ import annotations

import fixtures
import numpy as np
import pytest


@pytest.fixture
def rgb_gradient() -> np.ndarray:
    """8x8 RGB gradient, uint8."""
    return fixtures.rgb_gradient()


@pytest.fixture
def checkerboard() -> np.ndarray:
    """8x8 checkerboard, uint8."""
    return fixtures.checkerboard()


@pytest.fixture
def mask_gradient() -> np.ndarray:
    """8x8 single-channel ramp, uint8."""
    return fixtures.mask_gradient()


@pytest.fixture
def rgba_gradient() -> np.ndarray:
    """8x8 RGBA gradient, uint8."""
    return fixtures.rgba_gradient()

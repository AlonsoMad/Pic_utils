"""Deterministic synthetic image helpers for the Pic Utils test suite.

Every helper builds a small numpy array in memory - no binary fixture
files, no randomness, no external data. Default geometry is 8x8 so tests
stay fast; pass an explicit ``size`` for other square geometries.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIZE = 8


def rgb_gradient(size: int = DEFAULT_SIZE) -> np.ndarray:
    """RGB gradient, shape (size, size, 3), uint8; R ramps along x, G along y."""
    x = np.linspace(0.0, 255.0, size, dtype=np.float64)
    y = np.linspace(0.0, 255.0, size, dtype=np.float64)
    r = np.broadcast_to(x, (size, size))
    g = np.broadcast_to(y[:, None], (size, size))
    b = (r + g) / 2.0
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def checkerboard(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Checkerboard, shape (size, size), uint8; square (0, 0) is 0."""
    i = np.arange(size)[:, None]
    j = np.arange(size)[None, :]
    return (((i + j) % 2) * 255).astype(np.uint8)


def mask_gradient(size: int = DEFAULT_SIZE) -> np.ndarray:
    """Single-channel ramp 0..255 along x, shape (size, size), uint8."""
    x = np.linspace(0.0, 255.0, size, dtype=np.float64)
    return np.broadcast_to(x, (size, size)).astype(np.uint8)


def rgba_gradient(size: int = DEFAULT_SIZE) -> np.ndarray:
    """RGBA gradient, shape (size, size, 4), uint8; A = mask_gradient."""
    return np.dstack([rgb_gradient(size), mask_gradient(size)])

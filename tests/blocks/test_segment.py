"""Behavior tests for the segmentation block (P2-06, architecture §5.7)."""

from __future__ import annotations

import numpy as np
import pytest

from pic_utils.blocks.builtin.segment import segment
from pic_utils.core.image import Image


def quadrant_image(size: int = 32) -> Image:
    half = size // 2
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[:half, :half] = (255, 0, 0)
    pixels[:half, half:] = (0, 255, 0)
    pixels[half:, :half] = (0, 0, 255)
    pixels[half:, half:] = (255, 255, 255)
    return Image.from_rgb(pixels)


def test_segment_returns_one_layer_per_cluster() -> None:
    layers = segment(quadrant_image(), clusters=4, use_superpixels=False, seed=0)
    assert 1 <= len(layers) <= 4
    for layer in layers:
        assert layer.image.alpha is not None
        assert layer.image.alpha.shape == layer.image.rgb.shape[:2]
        assert np.any(layer.image.alpha == 255)
        assert layer.name.startswith("cluster_")


def test_segment_layers_are_cropped_to_their_bounding_box() -> None:
    image = quadrant_image()
    layers = segment(image, clusters=4, use_superpixels=False, seed=0)
    for layer in layers:
        assert layer.image.width <= image.width
        assert layer.image.height <= image.height


def test_segment_is_deterministic_with_a_fixed_seed() -> None:
    image = quadrant_image()
    first = segment(image, clusters=4, use_superpixels=False, seed=7)
    second = segment(image, clusters=4, use_superpixels=False, seed=7)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert left.name == right.name
        assert np.array_equal(left.image.rgb, right.image.rgb)
        assert np.array_equal(left.image.alpha, right.image.alpha)


def test_segment_superpixels_path_is_deterministic() -> None:
    image = quadrant_image()
    first = segment(image, clusters=4, use_superpixels=True, superpixel_count=100, seed=3)
    second = segment(image, clusters=4, use_superpixels=True, superpixel_count=100, seed=3)
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.image.rgb, right.image.rgb)
        assert np.array_equal(left.image.alpha, right.image.alpha)


def test_segment_modular_recolor_is_seeded_and_bounded() -> None:
    image = quadrant_image()
    first = segment(
        image,
        clusters=4,
        use_superpixels=False,
        modular_recolor=True,
        modulus=16,
        seed=11,
    )
    second = segment(
        image,
        clusters=4,
        use_superpixels=False,
        modular_recolor=True,
        modulus=16,
        seed=11,
    )
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left.image.rgb, right.image.rgb)
        assert int(left.image.rgb.max()) < 16


def test_segment_skips_clusters_below_the_minimum() -> None:
    with pytest.raises(ValueError, match="no cluster has at least"):
        segment(
            quadrant_image(),
            clusters=4,
            use_superpixels=False,
            min_cluster_pixels=1_000_000,
            seed=0,
        )


def test_segment_rejects_a_zero_modulus() -> None:
    with pytest.raises(ValueError, match="modulus must be positive"):
        segment(quadrant_image(), use_superpixels=False, modulus=0, seed=0)


def test_segment_caps_layers_at_sixteen() -> None:
    layers = segment(quadrant_image(), clusters=16, use_superpixels=False, seed=0)
    assert len(layers) <= 16

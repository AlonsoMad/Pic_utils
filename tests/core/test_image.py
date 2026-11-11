"""Behavior tests for the canonical image types (P1-01, architecture §3)."""

from __future__ import annotations

import fixtures
import numpy as np
import pytest

from pic_utils.core.image import BlendMode, Image, Layer, Mask


def test_image_accepts_rgb_and_exposes_geometry() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    assert image.size == (8, 8)
    assert image.width == 8
    assert image.height == 8
    assert image.alpha is None
    assert image.nbytes == 8 * 8 * 3
    assert image.rgb.dtype == np.uint8
    assert image.rgb.flags.c_contiguous
    assert not image.rgb.flags.writeable


def test_image_rejects_non_uint8_dtype() -> None:
    with pytest.raises(TypeError, match="dtype uint8"):
        Image(fixtures.rgb_gradient().astype(np.float32))


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((8, 8), dtype=np.uint8),
        np.zeros((8, 8, 4), dtype=np.uint8),
        np.zeros((0, 8, 3), dtype=np.uint8),
        np.zeros((8, 0, 3), dtype=np.uint8),
    ],
)
def test_image_rejects_bad_shapes(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Image(array)


def test_image_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match=r"numpy\.ndarray"):
        Image([[0, 0, 0]])  # type: ignore[arg-type]


def test_image_normalizes_non_contiguous_input() -> None:
    source = fixtures.rgb_gradient()[:, ::2]
    assert not source.flags.c_contiguous
    image = Image(source)
    assert image.rgb.flags.c_contiguous
    assert np.array_equal(image.rgb, source)
    assert source.flags.writeable


def test_image_marks_the_input_read_only() -> None:
    source = fixtures.rgb_gradient()
    image = Image(source)
    assert not source.flags.writeable
    assert not image.rgb.flags.writeable


def test_image_validates_alpha_shape_and_dtype() -> None:
    rgb = fixtures.rgb_gradient()
    with pytest.raises(ValueError, match="alpha shape"):
        Image(rgb, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(TypeError, match="alpha must have dtype uint8"):
        Image(rgb, np.zeros((8, 8), dtype=np.float32))


def test_from_rgba_splits_and_freezes_both_planes() -> None:
    rgba = fixtures.rgba_gradient()
    image = Image.from_rgba(rgba)
    assert np.array_equal(image.rgb, rgba[..., :3])
    assert image.alpha is not None
    assert np.array_equal(image.alpha, rgba[..., 3])
    assert image.alpha.flags.c_contiguous
    assert not image.alpha.flags.writeable


def test_from_rgba_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 4\)"):
        Image.from_rgba(fixtures.rgb_gradient())


def test_to_rgba_fills_opaque_alpha() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    rgba = image.to_rgba()
    assert rgba.shape == (8, 8, 4)
    assert np.array_equal(rgba[..., :3], image.rgb)
    assert np.all(rgba[..., 3] == 255)
    assert rgba.flags.writeable


def test_to_rgba_preserves_alpha() -> None:
    rgba = fixtures.rgba_gradient()
    image = Image.from_rgba(rgba)
    assert np.array_equal(image.to_rgba(), rgba)


def test_with_alpha_replaces_and_clears_alpha() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    alpha = fixtures.mask_gradient()
    with_alpha = image.with_alpha(alpha)
    assert with_alpha.alpha is not None
    assert np.array_equal(with_alpha.alpha, alpha)
    assert with_alpha.with_alpha(None).alpha is None


def test_copy_is_independent() -> None:
    image = Image.from_rgba(fixtures.rgba_gradient())
    copy = image.copy()
    assert copy.rgb is not image.rgb
    assert copy.alpha is not image.alpha
    assert np.array_equal(copy.rgb, image.rgb)
    assert copy.alpha is not None
    assert image.alpha is not None
    assert np.array_equal(copy.alpha, image.alpha)


def test_resized_scales_both_planes() -> None:
    image = Image.from_rgba(fixtures.rgba_gradient())
    resized = image.resized(4, 2)
    assert resized.size == (4, 2)
    assert resized.alpha is not None
    assert resized.alpha.shape == (2, 4)
    assert image.resized(8, 8) is image


def test_resized_rejects_empty_target() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    with pytest.raises(ValueError, match="at least 1x1"):
        image.resized(0, 2)


def test_to_proxy_preserves_aspect_ratio_and_resizes_alpha() -> None:
    image = Image.from_rgba(np.zeros((4, 8, 4), dtype=np.uint8))
    proxy = image.to_proxy(4)
    assert proxy.size == (4, 2)
    assert proxy.alpha is not None
    assert proxy.alpha.shape == (2, 4)
    assert image.to_proxy(8) is image


def test_to_proxy_rejects_non_positive_bound() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    with pytest.raises(ValueError, match="positive"):
        image.to_proxy(0)


def test_mask_validates_plane_and_freezes_it() -> None:
    mask = Mask(fixtures.mask_gradient())
    assert mask.data.dtype == np.uint8
    assert mask.data.flags.c_contiguous
    assert not mask.data.flags.writeable
    assert mask.soft is False
    assert Mask(fixtures.mask_gradient(), soft=True).soft is True


def test_mask_accepts_non_binary_values_at_construction() -> None:
    # Binary validation is the executor's job (§5.6), not a construction cost.
    Mask(fixtures.mask_gradient())


def test_mask_rejects_bad_dtype_and_shape() -> None:
    with pytest.raises(TypeError, match="dtype uint8"):
        Mask(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="2-dimensional"):
        Mask(np.zeros((8,), dtype=np.uint8))


def test_mask_rejects_non_bool_soft_flag() -> None:
    with pytest.raises(TypeError, match="soft must be a bool"):
        Mask(fixtures.mask_gradient(), soft=1)  # type: ignore[arg-type]


def test_layer_pairs_image_and_matching_mask() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    mask = Mask(fixtures.mask_gradient(), soft=True)
    layer = Layer(image, mask, "base")
    assert layer.image is image
    assert layer.mask is mask
    assert layer.name == "base"


def test_layer_rejects_mismatched_mask() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    with pytest.raises(ValueError, match="mask shape"):
        Layer(image, Mask(np.zeros((4, 4), dtype=np.uint8)))


def test_layer_rejects_wrong_types() -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    with pytest.raises(TypeError, match="image must be an Image"):
        Layer("not an image")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mask must be a Mask"):
        Layer(image, "not a mask")  # type: ignore[arg-type]


def test_blend_mode_has_the_six_required_values() -> None:
    assert {mode.value for mode in BlendMode} == {
        "normal",
        "multiply",
        "screen",
        "overlay",
        "add",
        "difference",
    }

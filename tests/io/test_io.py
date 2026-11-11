"""Behavior tests for the loader and saver (P1-03, architecture §4)."""

from __future__ import annotations

import fixtures
import numpy as np
import pytest
from PIL import Image as PILImage

from pic_utils.core.errors import (
    ExportError,
    ImageLoadError,
    ImageTooLargeError,
    UnsupportedImageError,
)
from pic_utils.core.image import Image, Mask
from pic_utils.io import formats, load_image, load_mask, probe_image, save_image, save_mask


def test_png_round_trip_rgb(tmp_path) -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    path = tmp_path / "rgb.png"
    save_image(image, path)
    loaded = load_image(path)
    assert np.array_equal(loaded.rgb, image.rgb)
    assert loaded.alpha is None
    assert loaded.rgb.flags.c_contiguous
    assert not loaded.rgb.flags.writeable


def test_png_round_trip_rgba(tmp_path) -> None:
    image = Image.from_rgba(fixtures.rgba_gradient())
    path = tmp_path / "rgba.png"
    save_image(image, path)
    loaded = load_image(path)
    assert np.array_equal(loaded.rgb, image.rgb)
    assert loaded.alpha is not None
    assert image.alpha is not None
    assert np.array_equal(loaded.alpha, image.alpha)


def test_tiff_round_trip_rgb(tmp_path) -> None:
    image = Image.from_rgb(fixtures.rgb_gradient())
    path = tmp_path / "rgb.tif"
    save_image(image, path)
    assert np.array_equal(load_image(path).rgb, image.rgb)


def test_tiff_round_trip_rgba(tmp_path) -> None:
    image = Image.from_rgba(fixtures.rgba_gradient())
    path = tmp_path / "rgba.tiff"
    save_image(image, path)
    loaded = load_image(path)
    assert np.array_equal(loaded.rgb, image.rgb)
    assert loaded.alpha is not None
    assert image.alpha is not None
    assert np.array_equal(loaded.alpha, image.alpha)


def test_jpeg_round_trip(tmp_path) -> None:
    pixels = np.full((8, 8, 3), (10, 120, 240), dtype=np.uint8)
    path = tmp_path / "solid.jpg"
    save_image(Image.from_rgb(pixels), path, jpeg_quality=100)
    loaded = load_image(path)
    assert loaded.size == (8, 8)
    assert np.allclose(loaded.rgb, pixels, atol=8)


def test_probe_image_reports_metadata(tmp_path) -> None:
    path = tmp_path / "meta.png"
    save_image(Image.from_rgba(fixtures.rgba_gradient()), path)
    info = probe_image(path)
    assert info.width == 8
    assert info.height == 8
    assert info.mode == "RGBA"
    assert info.format == "PNG"
    assert info.has_alpha is True


def test_grayscale_loads_as_rgb(tmp_path) -> None:
    gray = fixtures.mask_gradient()
    path = tmp_path / "gray.png"
    PILImage.fromarray(gray, mode="L").save(path)
    loaded = load_image(path)
    assert loaded.alpha is None
    for channel in range(3):
        assert np.array_equal(loaded.rgb[..., channel], gray)


def test_exif_orientation_is_applied_and_not_reemitted(tmp_path) -> None:
    stored = np.zeros((2, 4, 3), dtype=np.uint8)
    stored[0, 0] = [255, 0, 0]
    stored[0, 3] = [0, 255, 0]
    stored[1, 0] = [0, 0, 255]
    stored[1, 3] = [255, 255, 255]
    source = tmp_path / "oriented.png"
    exif = PILImage.Exif()
    exif[274] = 6
    PILImage.fromarray(stored, mode="RGB").save(source, format="PNG", exif=exif)

    loaded = load_image(source)
    assert loaded.size == (2, 4)
    assert np.array_equal(loaded.rgb[0, 0], [0, 0, 255])
    assert np.array_equal(loaded.rgb[0, 1], [255, 0, 0])
    assert np.array_equal(loaded.rgb[3, 0], [255, 255, 255])
    assert np.array_equal(loaded.rgb[3, 1], [0, 255, 0])

    target = tmp_path / "reencoded.jpg"
    save_image(loaded, target)
    with PILImage.open(target) as reopened:
        assert reopened.getexif().get(274) is None


def test_16bit_input_is_rejected(tmp_path) -> None:
    path = tmp_path / "16bit.png"
    array = np.arange(16, dtype=np.uint16).reshape(4, 4) * 4000
    PILImage.fromarray(array).save(path)
    with pytest.raises(UnsupportedImageError, match=r"image\.16bit"):
        load_image(path)


def test_too_large_input_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(formats, "MAX_PIXELS", 1)
    path = tmp_path / "large.png"
    PILImage.fromarray(fixtures.rgb_gradient()).save(path)
    with pytest.raises(ImageTooLargeError) as info:
        load_image(path)
    assert info.value.width == 8
    assert info.value.height == 8


def test_missing_file_raises_image_load_error(tmp_path) -> None:
    with pytest.raises(ImageLoadError, match="cannot load"):
        load_image(tmp_path / "missing.png")


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".tif"])
def test_srgb_profile_is_embedded(tmp_path, suffix: str) -> None:
    path = tmp_path / f"profile{suffix}"
    save_image(Image.from_rgb(fixtures.rgb_gradient()), path)
    with PILImage.open(path) as reopened:
        assert reopened.info.get("icc_profile")


def test_mask_round_trip(tmp_path) -> None:
    mask = Mask(fixtures.mask_gradient(), soft=True)
    path = tmp_path / "mask.png"
    save_mask(mask, path)
    loaded = load_mask(path)
    assert np.array_equal(loaded.data, mask.data)
    assert loaded.soft is True


def test_atomic_save_failure_leaves_original_intact(tmp_path, monkeypatch) -> None:
    target = tmp_path / "image.png"
    target.write_bytes(b"original")

    def failing_replace(src: object, dst: object, **kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("pic_utils.io.saver.os.replace", failing_replace)
    with pytest.raises(ExportError, match="cannot save"):
        save_image(Image.from_rgb(fixtures.rgb_gradient()), target)
    assert target.read_bytes() == b"original"
    assert [entry.name for entry in tmp_path.iterdir()] == ["image.png"]


def test_save_image_rejects_unsupported_extension(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported image extension"):
        save_image(Image.from_rgb(fixtures.rgb_gradient()), tmp_path / "x.bmp")


def test_save_image_rejects_bad_jpeg_quality(tmp_path) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        save_image(
            Image.from_rgb(fixtures.rgb_gradient()),
            tmp_path / "x.jpg",
            jpeg_quality=0,
        )


def test_save_image_requires_an_image(tmp_path) -> None:
    with pytest.raises(TypeError, match="image must be an Image"):
        save_image("not an image", tmp_path / "x.png")  # type: ignore[arg-type]


def test_load_mask_rejects_non_png(tmp_path) -> None:
    path = tmp_path / "mask.tif"
    PILImage.fromarray(fixtures.mask_gradient(), mode="L").save(path, format="TIFF")
    with pytest.raises(UnsupportedImageError, match="mask must be a PNG"):
        load_mask(path)

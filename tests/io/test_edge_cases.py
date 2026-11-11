"""I/O edge cases (P1-04, architecture §4.2 and §20.3)."""

from __future__ import annotations

import os
import stat

import fixtures
import numpy as np
import pytest
from PIL import Image as PILImage

from pic_utils.core.errors import ExportError, ImageLoadError, UnsupportedImageError
from pic_utils.core.image import Image, Mask
from pic_utils.io import load_image, load_mask, probe_image, save_image, save_mask


def test_corrupt_file_raises_image_load_error(tmp_path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not an image")
    with pytest.raises(ImageLoadError, match="cannot load"):
        load_image(path)
    with pytest.raises(ImageLoadError, match="cannot load"):
        probe_image(path)


def test_corrupt_mask_raises_image_load_error(tmp_path) -> None:
    path = tmp_path / "mask.png"
    path.write_bytes(b"not a png")
    with pytest.raises(ImageLoadError, match="cannot load"):
        load_mask(path)


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permissions are required and root ignores them",
)
def test_unwritable_target_raises_export_error(tmp_path) -> None:
    directory = tmp_path / "readonly"
    directory.mkdir()
    directory.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        with pytest.raises(ExportError, match="cannot save"):
            save_image(Image.from_rgb(fixtures.rgb_gradient()), directory / "x.png")
        assert list(directory.iterdir()) == []
    finally:
        directory.chmod(stat.S_IRWXU)


def test_missing_destination_directory_raises_export_error(tmp_path) -> None:
    with pytest.raises(ExportError, match="cannot save"):
        save_image(
            Image.from_rgb(fixtures.rgb_gradient()),
            tmp_path / "missing" / "x.png",
        )


def test_cmyk_loads_as_rgb(tmp_path) -> None:
    path = tmp_path / "cmyk.tif"
    pil = PILImage.new("CMYK", (2, 1))
    pil.putpixel((0, 0), (0, 0, 0, 0))
    pil.putpixel((1, 0), (0, 0, 0, 255))
    pil.save(path, format="TIFF")
    loaded = load_image(path)
    assert loaded.alpha is None
    assert np.array_equal(loaded.rgb[0, 0], [255, 255, 255])
    assert np.array_equal(loaded.rgb[0, 1], [0, 0, 0])


def test_palette_loads_as_rgb(tmp_path) -> None:
    path = tmp_path / "palette.png"
    pil = PILImage.new("P", (2, 1))
    pil.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    pil.putpixel((0, 0), 0)
    pil.putpixel((1, 0), 1)
    pil.save(path, format="PNG")
    loaded = load_image(path)
    assert np.array_equal(loaded.rgb[0, 0], [255, 0, 0])
    assert np.array_equal(loaded.rgb[0, 1], [0, 255, 0])


def test_jpeg_flattens_alpha_onto_white(tmp_path) -> None:
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:4, :4] = [255, 0, 0, 0]
    rgba[:4, 4:] = [0, 0, 255, 255]
    rgba[4:, :4] = [0, 255, 0, 128]
    rgba[4:, 4:] = [0, 0, 0, 0]
    path = tmp_path / "alpha.jpg"
    save_image(Image.from_rgba(rgba), path, jpeg_quality=100)
    loaded = load_image(path)
    assert np.allclose(loaded.rgb[2, 2], [255, 255, 255], atol=10)
    assert np.allclose(loaded.rgb[2, 6], [0, 0, 255], atol=10)
    assert np.allclose(loaded.rgb[6, 2], [127, 255, 127], atol=10)
    assert np.allclose(loaded.rgb[6, 6], [255, 255, 255], atol=10)


def test_unsupported_format_is_rejected(tmp_path) -> None:
    path = tmp_path / "image.bmp"
    PILImage.fromarray(fixtures.rgb_gradient()).save(path, format="BMP")
    with pytest.raises(UnsupportedImageError, match="unsupported image format"):
        load_image(path)


def test_save_mask_requires_png(tmp_path) -> None:
    with pytest.raises(ValueError, match="masks must be saved as PNG"):
        save_mask(Mask(fixtures.mask_gradient(), soft=True), tmp_path / "mask.tif")

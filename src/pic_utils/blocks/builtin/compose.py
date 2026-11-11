"""Compositing blocks (architecture §5.7).

The legacy compositing defects are fixed here: white-to-transparent applies
to the overlay (not the background), size mismatches pad/crop instead of
raising, and RGB is explicit throughout.
"""

from __future__ import annotations

from typing import Annotated

import cv2
import numpy as np

from pic_utils.blocks import Choice, FloatRange, IntRange, OutputKind, block
from pic_utils.core.color import luminance
from pic_utils.core.image import Image, Mask

__all__ = ["conditional_overlay", "double_exposure", "overlay"]


def _alpha(image: Image) -> np.ndarray:
    if image.alpha is None:
        return np.full(image.rgb.shape[:2], 255, dtype=np.uint8)
    return image.alpha


def _pad_or_crop(array: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = array.shape[:2]
    if source_w > width:
        left = (source_w - width) // 2
        array = array[:, left : left + width]
    if source_h > height:
        top = (source_h - height) // 2
        array = array[top : top + height, :]
    source_h, source_w = array.shape[:2]
    if source_w < width or source_h < height:
        pad_x = width - source_w
        pad_y = height - source_h
        left = pad_x // 2
        right = pad_x - left
        top = pad_y // 2
        bottom = pad_y - top
        pad_width = [(top, bottom), (left, right)]
        if array.ndim == 3:
            pad_width.append((0, 0))
        array = np.pad(array, pad_width, mode="constant", constant_values=0)
    return array


def _fit_to(
    rgb: np.ndarray,
    alpha: np.ndarray,
    width: int,
    height: int,
    *,
    resize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if resize:
        if rgb.shape[1] != width or rgb.shape[0] != height:
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
        return rgb, alpha
    return _pad_or_crop(rgb, width, height), _pad_or_crop(alpha, width, height)


def _overlay_canvas(
    rgb: np.ndarray,
    alpha: np.ndarray,
    width: int,
    height: int,
    x: int,
    y: int,
    opacity: float,
) -> tuple[np.ndarray, np.ndarray]:
    canvas_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((height, width), dtype=np.uint8)
    source_x = max(0, -x)
    source_y = max(0, -y)
    target_x = max(0, x)
    target_y = max(0, y)
    copy_w = min(rgb.shape[1] - source_x, width - target_x)
    copy_h = min(rgb.shape[0] - source_y, height - target_y)
    if copy_w > 0 and copy_h > 0:
        canvas_rgb[target_y : target_y + copy_h, target_x : target_x + copy_w] = rgb[
            source_y : source_y + copy_h, source_x : source_x + copy_w
        ]
        scaled = np.rint(
            alpha[source_y : source_y + copy_h, source_x : source_x + copy_w].astype(np.float32)
            * opacity
        )
        canvas_alpha[target_y : target_y + copy_h, target_x : target_x + copy_w] = scaled.clip(
            0, 255
        ).astype(np.uint8)
    return canvas_rgb, canvas_alpha


def _composite(
    background_rgb: np.ndarray,
    background_alpha: np.ndarray,
    foreground_rgb: np.ndarray,
    foreground_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = foreground_alpha.astype(np.float32)[..., None] / 255.0
    destination = background_alpha.astype(np.float32)[..., None] / 255.0
    out_alpha = source + destination * (1.0 - source)
    numerator = foreground_rgb.astype(np.float32) * source + background_rgb.astype(
        np.float32
    ) * destination * (1.0 - source)
    out_rgb = np.zeros_like(background_rgb, dtype=np.float32)
    np.divide(numerator, out_alpha, out=out_rgb, where=out_alpha > 0)
    rgb = np.rint(out_rgb).clip(0, 255).astype(np.uint8)
    alpha = np.rint(out_alpha[..., 0] * 255).clip(0, 255).astype(np.uint8)
    return rgb, alpha


def _finish(rgb: np.ndarray, alpha: np.ndarray) -> Image:
    if np.all(alpha == 255):
        return Image(rgb)
    return Image(rgb, alpha)


@block(
    name="overlay",
    category="compose",
    description="Place a second image over or under the chain image.",
    output=OutputKind.IMAGE,
)
def overlay(
    image: Image,
    second: Image,
    x: Annotated[int, IntRange(-4096, 4096, step=1)] = 0,
    y: Annotated[int, IntRange(-4096, 4096, step=1)] = 0,
    z_order: Annotated[str, Choice(("over", "under"))] = "over",
    white_to_transparent: bool = True,
    resize_to_match: bool = True,
    opacity: Annotated[float, FloatRange(0.0, 1.0, step=0.01)] = 1.0,
) -> Image:
    """Composite ``second`` onto ``image`` at ``(x, y)``.

    ``white_to_transparent`` applies to the overlay's near-white pixels.
    ``resize_to_match`` scales the overlay to the chain image; without it the
    overlay keeps its size and is cropped by the canvas. ``z_order`` chooses
    whether the overlay is above (``over``) or below (``under``) the chain
    image.
    """
    overlay_rgb = second.rgb
    overlay_alpha = _alpha(second)
    if white_to_transparent:
        white = np.all(overlay_rgb > 240, axis=-1)
        overlay_alpha = np.where(white, 0, overlay_alpha).astype(np.uint8)
    if resize_to_match:
        overlay_rgb, overlay_alpha = _fit_to(
            overlay_rgb,
            overlay_alpha,
            image.width,
            image.height,
            resize=True,
        )
    canvas_rgb, canvas_alpha = _overlay_canvas(
        overlay_rgb,
        overlay_alpha,
        image.width,
        image.height,
        x,
        y,
        opacity,
    )
    background_alpha = _alpha(image)
    if z_order == "over":
        out_rgb, out_alpha = _composite(image.rgb, background_alpha, canvas_rgb, canvas_alpha)
    elif z_order == "under":
        out_rgb, out_alpha = _composite(canvas_rgb, canvas_alpha, image.rgb, background_alpha)
    else:
        raise ValueError(f"invalid z_order {z_order!r}")
    return _finish(out_rgb, out_alpha)


@block(
    name="double_exposure",
    category="compose",
    description="Blend two images with a configurable exposure factor.",
    output=OutputKind.IMAGE,
)
def double_exposure(
    image: Image,
    second: Image,
    exposure_factor: Annotated[float, FloatRange(0.0, 2.0, step=0.01)] = 0.5,
    resize_to_match: bool = True,
) -> Image:
    """Blend ``image`` and ``second`` as ``image*(1-f) + second*f``.

    Values are clipped to 0..255. With ``resize_to_match=False`` the second
    image is padded/cropped to the first image's size instead of raising.
    """
    second_rgb, second_alpha = _fit_to(
        second.rgb,
        _alpha(second),
        image.width,
        image.height,
        resize=resize_to_match,
    )
    blended = (
        image.rgb.astype(np.float32) * (1.0 - exposure_factor)
        + second_rgb.astype(np.float32) * exposure_factor
    )
    rgb = np.rint(np.clip(blended, 0, 255)).astype(np.uint8)
    if image.alpha is not None:
        alpha = image.alpha
    elif second.alpha is not None:
        alpha = second_alpha
    else:
        alpha = None
    return Image(rgb, alpha)


@block(
    name="conditional_overlay",
    category="compose",
    description="Overlay a second image where a mask and color conditions allow.",
    output=OutputKind.IMAGE,
)
def conditional_overlay(
    image: Image,
    second: Image,
    mask: Mask,
    threshold: Annotated[int, IntRange(0, 255, step=1)] = 255,
    r_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    r_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    g_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    g_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    b_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    b_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    strict: bool = True,
) -> Image:
    """Overlay ``second`` where the mask, brightness, and color ranges agree.

    The mask is the primary gate; its soft value also scales the overlay's
    alpha. ``threshold`` keeps pixels at or below that luminance, and the
    per-channel ranges use ``strict`` (all channels) or loose (any channel)
    semantics. The second image is padded/cropped to the chain image's size.
    """
    if mask.data.shape != image.rgb.shape[:2]:
        raise ValueError(
            f"mask shape {mask.data.shape} does not match image shape {image.rgb.shape[:2]}"
        )
    second_rgb, second_alpha = _fit_to(
        second.rgb,
        _alpha(second),
        image.width,
        image.height,
        resize=False,
    )
    values = np.rint(luminance(image.rgb)).clip(0, 255).astype(np.uint8)
    brightness_ok = values <= threshold
    red = image.rgb[..., 0]
    green = image.rgb[..., 1]
    blue = image.rgb[..., 2]
    conditions = (
        (red >= r_low) & (red <= r_high),
        (green >= g_low) & (green <= g_high),
        (blue >= b_low) & (blue <= b_high),
    )
    if strict:
        color_ok = conditions[0] & conditions[1] & conditions[2]
    else:
        color_ok = conditions[0] | conditions[1] | conditions[2]
    condition = (mask.data > 0) & brightness_ok & color_ok
    combined_alpha = np.where(
        condition,
        (second_alpha.astype(np.uint32) * mask.data.astype(np.uint32)) // 255,
        0,
    ).astype(np.uint8)
    out_rgb, out_alpha = _composite(image.rgb, _alpha(image), second_rgb, combined_alpha)
    return _finish(out_rgb, out_alpha)

"""Mask blocks (architecture §5.7)."""

from __future__ import annotations

from typing import Annotated

import numpy as np

from pic_utils.blocks import Choice, IntRange, OutputKind, block
from pic_utils.core.color import luminance
from pic_utils.core.image import Image, Mask
from pic_utils.core.stats import find_channel_percentiles

__all__ = ["brightness_mask", "channel_mask", "color_range_mask"]

_CHANNEL_INDEX = {"R": 0, "G": 1, "B": 2}


def _as_uint8(values: np.ndarray) -> np.ndarray:
    return np.rint(values).clip(0, 255).astype(np.uint8)


def _mask_from_condition(
    inside: np.ndarray,
    values: np.ndarray,
    *,
    soft: bool,
    invert: bool,
) -> Mask:
    if soft:
        data = np.where(inside, values, 0).astype(np.uint8)
    else:
        data = np.where(inside, 255, 0).astype(np.uint8)
    if invert:
        data = 255 - data
    return Mask(data, soft=soft)


@block(
    name="channel_mask",
    category="masks",
    description="Keep pixels whose channel value falls inside a range.",
    output=OutputKind.MASK,
)
def channel_mask(
    image: Image,
    channel: Annotated[str, Choice(("R", "G", "B", "L"))] = "B",
    low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    soft: bool = False,
    invert: bool = False,
) -> Mask:
    """Select pixels whose R/G/B channel or luminance is in ``low..high``.

    A soft mask carries the channel intensity inside the range and 0 outside;
    a binary mask is 255 inside and 0 outside.
    """
    if channel == "L":
        values = _as_uint8(luminance(image.rgb))
    elif channel in _CHANNEL_INDEX:
        values = image.rgb[..., _CHANNEL_INDEX[channel]]
    else:
        raise ValueError(f"invalid channel {channel!r}")
    inside = (values >= low) & (values <= high)
    return _mask_from_condition(inside, values, soft=soft, invert=invert)


@block(
    name="brightness_mask",
    category="masks",
    description="Keep pixels at or below a brightness threshold.",
    output=OutputKind.MASK,
)
def brightness_mask(
    image: Image,
    threshold: Annotated[int, IntRange(0, 255, step=1)] = 128,
    auto: bool = False,
    soft: bool = False,
    invert: bool = False,
) -> Mask:
    """Select pixels whose Rec. 709 luminance is at most ``threshold``.

    ``auto`` replaces ``threshold`` with the median luminance. A soft mask
    carries the luminance inside the selection and 0 outside.
    """
    values = _as_uint8(luminance(image.rgb))
    effective = find_channel_percentiles(values, 50)[0] if auto else threshold
    inside = values <= effective
    return _mask_from_condition(inside, values, soft=soft, invert=invert)


@block(
    name="color_range_mask",
    category="masks",
    description="Keep pixels whose R/G/B values fall inside per-channel ranges.",
    output=OutputKind.MASK,
)
def color_range_mask(
    image: Image,
    r_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    r_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    g_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    g_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    b_low: Annotated[int, IntRange(0, 255, step=1)] = 0,
    b_high: Annotated[int, IntRange(0, 255, step=1)] = 255,
    strict: bool = True,
    soft: bool = False,
    invert: bool = False,
) -> Mask:
    """Select pixels by per-channel ranges.

    ``strict`` requires all three channels to match; otherwise any matching
    channel is enough. A soft mask carries the mean RGB value inside the
    selection and 0 outside.
    """
    red = image.rgb[..., 0]
    green = image.rgb[..., 1]
    blue = image.rgb[..., 2]
    conditions = (
        (red >= r_low) & (red <= r_high),
        (green >= g_low) & (green <= g_high),
        (blue >= b_low) & (blue <= b_high),
    )
    if strict:
        inside = conditions[0] & conditions[1] & conditions[2]
    else:
        inside = conditions[0] | conditions[1] | conditions[2]
    values = ((red.astype(np.uint16) + green + blue) // 3).astype(np.uint8)
    return _mask_from_condition(inside, values, soft=soft, invert=invert)

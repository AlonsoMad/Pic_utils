"""Channel permutation blocks (architecture §5.7)."""

from __future__ import annotations

from typing import Annotated

import numpy as np

from pic_utils.blocks import Choice, OutputKind, block
from pic_utils.core.image import Image, Layer

__all__ = ["channel_permute", "permutation_grid"]

_ORDERS = ("rgb", "rbg", "grb", "gbr", "brg", "bgr")
_INDEX = {"r": 0, "g": 1, "b": 2}


def _permuted(image: Image, order: str) -> np.ndarray:
    if len(order) != 3 or set(order) != set(_INDEX):
        raise ValueError(f"invalid channel order {order!r}")
    return image.rgb[..., [_INDEX[character] for character in order]]


@block(
    name="channel_permute",
    category="channels",
    description="Reorder and optionally invert the RGB channels.",
    output=OutputKind.IMAGE,
)
def channel_permute(
    image: Image,
    order: Annotated[str, Choice(_ORDERS)] = "rgb",
    invert_r: bool = False,
    invert_g: bool = False,
    invert_b: bool = False,
) -> Image:
    """Return the image with channels reordered by ``order``.

    ``order`` names the input channels in output order: ``bgr`` swaps red and
    blue. ``invert_*`` invert the corresponding output channel. Alpha is
    preserved.
    """
    rgb = _permuted(image, order)
    if invert_r or invert_g or invert_b:
        rgb = rgb.copy()
        for channel, invert in enumerate((invert_r, invert_g, invert_b)):
            if invert:
                rgb[..., channel] = 255 - rgb[..., channel]
    return Image(rgb, image.alpha)


@block(
    name="permutation_grid",
    category="channels",
    description="Emit the channel-order or channel-inversion permutations.",
    output=OutputKind.LAYERS,
)
def permutation_grid(
    image: Image,
    mode: Annotated[str, Choice(("channel_order", "invert"))] = "channel_order",
    include_original: bool = False,
) -> list[Layer]:
    """Return 6-8 layers covering the selected permutation family.

    ``channel_order`` emits the six channel orders (plus the original first
    when ``include_original`` is set). ``invert`` emits the eight
    combinations of per-channel inversion; the original is the no-inversion
    combination, so ``include_original`` has no effect there.
    """
    layers: list[Layer] = []
    if mode == "channel_order":
        if include_original:
            layers.append(Layer(image, name="original"))
        for order in _ORDERS:
            layers.append(Layer(Image(_permuted(image, order), image.alpha), name=order))
    elif mode == "invert":
        for bits in range(8):
            rgb = image.rgb.copy()
            inverted = [channel for channel in range(3) if bits & (1 << channel)]
            for channel in inverted:
                rgb[..., channel] = 255 - rgb[..., channel]
            suffix = "".join("rgb"[channel] for channel in inverted) or "original"
            layers.append(Layer(Image(rgb, image.alpha), name=f"invert_{suffix}"))
    else:
        raise ValueError(f"invalid permutation mode {mode!r}")
    return layers

"""Channel statistics used by mask auto-thresholds (architecture §5.7).

``find_channel_percentiles`` is a utility, not a block. It uses a 256-bin
histogram for uint8 data, so a 48 MP image costs one linear pass per channel
instead of a sort.
"""

from __future__ import annotations

import numpy as np

__all__ = ["find_channel_percentiles"]


def find_channel_percentiles(image: np.ndarray, percentile: float) -> dict[int, int]:
    """Return the intensity at ``percentile`` for every channel.

    The input is a 2-D grayscale or 3-D channel-last array with integer
    values in 0..255. The result maps channel index to the smallest
    intensity whose cumulative histogram reaches the requested percentile
    (the nearest-rank definition; percentile 0 is the channel minimum).
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"expected a 2-D or 3-D array, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("cannot compute percentiles of an empty image")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"image must have an integer dtype, got {array.dtype}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be in 0..100, got {percentile}")

    if array.dtype == np.uint8:
        data = array
    else:
        if int(array.min()) < 0 or int(array.max()) > 255:
            raise ValueError("image values must be in 0..255")
        data = array.astype(np.uint8)

    result: dict[int, int] = {}
    for channel_index in range(data.shape[2]):
        channel = data[:, :, channel_index]
        if percentile == 0:
            result[channel_index] = int(channel.min())
            continue
        histogram = np.bincount(channel.ravel(), minlength=256)
        cumulative = np.cumsum(histogram)
        target = percentile * cumulative[-1] / 100.0
        result[channel_index] = int(np.searchsorted(cumulative, target, side="left"))
    return result

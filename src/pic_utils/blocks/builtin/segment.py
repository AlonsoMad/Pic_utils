"""Segmentation block (architecture §5.7)."""

from __future__ import annotations

from typing import Annotated

import cv2
import numpy as np
from skimage.segmentation import slic
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from pic_utils.blocks import IntRange, OutputKind, Seed, block
from pic_utils.core.image import Image, Layer

__all__ = ["segment"]


def _superpixel_means(lab: np.ndarray, segments: np.ndarray) -> np.ndarray:
    labels = segments.ravel()
    count = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=count).astype(np.float32)
    means = np.empty((count, 3), dtype=np.float32)
    for channel in range(3):
        sums = np.bincount(labels, weights=lab[..., channel].ravel(), minlength=count)
        means[:, channel] = sums / np.maximum(counts, 1)
    return means


def _cluster_labels(
    lab: np.ndarray,
    *,
    use_superpixels: bool,
    superpixel_count: int,
    clusters: int,
    seed: int | None,
) -> np.ndarray:
    if use_superpixels:
        segments = slic(
            lab,
            n_segments=superpixel_count,
            compactness=10,
            start_label=0,
            channel_axis=-1,
        )
        samples = _superpixel_means(lab, segments)
        effective = min(clusters, samples.shape[0])
        if effective < 1:
            raise ValueError("not enough superpixels to cluster")
        model = MiniBatchKMeans(
            n_clusters=effective,
            random_state=seed,
            batch_size=4096,
            n_init=3,
        )
        return model.fit_predict(samples)[segments]

    samples = lab.reshape((-1, 3))
    effective = min(clusters, samples.shape[0])
    if effective < 1:
        raise ValueError("not enough pixels to cluster")
    model = MiniBatchKMeans(
        n_clusters=effective,
        random_state=seed,
        batch_size=4096,
        n_init=3,
    )
    return model.fit_predict(samples).reshape(lab.shape[:2])


def _bounding_box(mask: np.ndarray) -> tuple[slice, slice]:
    rows = np.flatnonzero(np.any(mask, axis=1))
    columns = np.flatnonzero(np.any(mask, axis=0))
    return slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1)


@block(
    name="segment",
    category="segment",
    description="Cluster the image and emit one cropped layer per cluster.",
    output=OutputKind.LAYERS,
)
def segment(
    image: Image,
    clusters: Annotated[int, IntRange(2, 16, step=1)] = 6,
    use_superpixels: bool = True,
    superpixel_count: Annotated[int, IntRange(100, 5000, step=1)] = 500,
    min_cluster_pixels: Annotated[int, IntRange(0, 1_000_000, step=1)] = 0,
    modular_recolor: bool = False,
    modulus: Annotated[int, IntRange(2, 256, step=1)] = 256,
    seed: Annotated[int | None, Seed()] = None,
) -> list[Layer]:
    """Segment into color clusters and return one layer per cluster.

    Superpixels (SLIC on Lab) are clustered by their means when enabled;
    otherwise every pixel is clustered. Each layer is cropped to its
    cluster's bounding box, with alpha 0 outside the cluster. With
    ``modular_recolor`` each cluster gets a seeded modular color transform
    ``(channel * offset) % modulus``. Clusters below ``min_cluster_pixels``
    are skipped; if none remain the block fails instead of returning an
    empty layer list.
    """
    if modulus < 1:
        raise ValueError(f"modulus must be positive, got {modulus}")

    lab = cv2.cvtColor(image.rgb, cv2.COLOR_RGB2LAB)
    labels = _cluster_labels(
        lab,
        use_superpixels=use_superpixels,
        superpixel_count=superpixel_count,
        clusters=clusters,
        seed=seed,
    )
    source_alpha = (
        image.alpha
        if image.alpha is not None
        else np.full(image.rgb.shape[:2], 255, dtype=np.uint8)
    )
    rng = np.random.default_rng(seed)
    layers: list[Layer] = []
    for cluster_index in range(int(labels.max()) + 1):
        mask = labels == cluster_index
        pixel_count = int(mask.sum())
        if pixel_count == 0 or pixel_count < min_cluster_pixels:
            continue
        row_slice, column_slice = _bounding_box(mask)
        cropped_mask = mask[row_slice, column_slice]
        cropped_rgb = image.rgb[row_slice, column_slice]
        cropped_alpha = np.where(
            cropped_mask,
            source_alpha[row_slice, column_slice],
            0,
        ).astype(np.uint8)
        if modular_recolor:
            offsets = rng.integers(0, modulus, size=3).astype(np.uint32)
            recolored = (cropped_rgb.astype(np.uint32) * offsets) % modulus
            cropped_rgb = np.where(cropped_mask[..., None], recolored, 0).astype(np.uint8)
        else:
            cropped_rgb = np.where(cropped_mask[..., None], cropped_rgb, 0).astype(np.uint8)
        layers.append(
            Layer(
                Image(cropped_rgb, cropped_alpha),
                name=f"cluster_{cluster_index}",
            )
        )

    if not layers:
        raise ValueError(f"no cluster has at least {min_cluster_pixels} pixels")
    return layers[:16]

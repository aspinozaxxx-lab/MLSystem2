"""Растеризация маски окна."""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize

from .contracts import HARD_NEGATIVE_LABEL


def rasterize_window_mask(geometries: list[object], out_shape: tuple[int, int], transform) -> np.ndarray:
    if not geometries:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        [(geometry, 1) for geometry in geometries],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return (mask > 0).astype(np.uint8)


def rasterize_instance_mask(
    geometries: list[object],
    out_shape: tuple[int, int],
    transform,
    nodata_pixels: np.ndarray,
) -> np.ndarray:
    if not geometries:
        return np.zeros(out_shape, dtype=np.int64)
    mask = rasterize(
        [(geometry, index) for index, geometry in enumerate(geometries, start=1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    ).astype(np.int64, copy=False)
    mask[nodata_pixels] = 0
    return mask


def build_supervision_mask(
    *,
    positive_layers: list[tuple[int, list[object]]],
    hard_negative_geometries: list[object],
    out_shape: tuple[int, int],
    transform,
    nodata_pixels: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(out_shape, dtype=np.int64)
    if hard_negative_geometries:
        hard_negative_mask = rasterize_window_mask(
            hard_negative_geometries,
            out_shape=out_shape,
            transform=transform,
        )
        mask[hard_negative_mask == 1] = HARD_NEGATIVE_LABEL
    for label_id, geometries in positive_layers:
        if not geometries:
            continue
        positive_mask = rasterize_window_mask(
            geometries,
            out_shape=out_shape,
            transform=transform,
        )
        mask[positive_mask == 1] = int(label_id)
    mask[nodata_pixels] = 0
    return mask

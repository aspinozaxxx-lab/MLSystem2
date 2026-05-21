"""Растеризация маски окна."""

from __future__ import annotations

import numpy as np
from rasterio.features import rasterize


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

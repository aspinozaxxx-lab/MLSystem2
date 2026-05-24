"""Внутренний coarse-index фактического valid-data footprint VRT."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from rasterio.io import DatasetReader
from rasterio.windows import Window

from ._windows import TileWindow


_VALID_FOOTPRINT_STRIDE = 64
_VALID_VALUE_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class ValidFootprintDiagnostics:
    candidate_window_count_before_valid_filter: int
    valid_window_count: int
    black_filtered_window_count: int
    valid_footprint_stride: int
    valid_footprint_total_cells: int
    valid_footprint_valid_cells: int


def filter_valid_windows(
    dataset: DatasetReader,
    windows: list[TileWindow],
    *,
    nodata: object,
) -> tuple[list[TileWindow], ValidFootprintDiagnostics]:
    valid_footprint = _read_valid_footprint(dataset, nodata=nodata)
    coarse_valid_windows = [
        window for window in windows if _window_intersects_valid_footprint(window, valid_footprint)
    ]
    valid_windows = _filter_sparse_black_windows(dataset, coarse_valid_windows, nodata=nodata)
    total_cells = int(valid_footprint.size)
    valid_cells = int(np.count_nonzero(valid_footprint))
    return valid_windows, ValidFootprintDiagnostics(
        candidate_window_count_before_valid_filter=len(windows),
        valid_window_count=len(valid_windows),
        black_filtered_window_count=len(windows) - len(valid_windows),
        valid_footprint_stride=_VALID_FOOTPRINT_STRIDE,
        valid_footprint_total_cells=total_cells,
        valid_footprint_valid_cells=valid_cells,
    )


def _read_valid_footprint(dataset: DatasetReader, *, nodata: object) -> np.ndarray:
    coarse_width = max(1, math.ceil(dataset.width / _VALID_FOOTPRINT_STRIDE))
    coarse_height = max(1, math.ceil(dataset.height / _VALID_FOOTPRINT_STRIDE))
    out_shape = (dataset.count, coarse_height, coarse_width)

    masks = dataset.read_masks(out_shape=out_shape)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    valid_by_mask = np.any(masks > 0, axis=0)

    data = dataset.read(
        out_shape=out_shape,
        masked=False,
        fill_value=nodata,
        boundless=False,
    )
    if data.ndim == 2:
        data = data[None, :, :]
    data_f32 = data.astype(np.float32, copy=False)
    valid_by_value = np.any(np.abs(data_f32) > _VALID_VALUE_EPS, axis=0)
    return np.logical_and(valid_by_mask, valid_by_value)


def _window_intersects_valid_footprint(window: TileWindow, valid_footprint: np.ndarray) -> bool:
    height, width = valid_footprint.shape
    x0 = max(0, window.x // _VALID_FOOTPRINT_STRIDE)
    y0 = max(0, window.y // _VALID_FOOTPRINT_STRIDE)
    x1 = min(width, math.ceil((window.x + window.width) / _VALID_FOOTPRINT_STRIDE))
    y1 = min(height, math.ceil((window.y + window.height) / _VALID_FOOTPRINT_STRIDE))
    if x0 >= x1 or y0 >= y1:
        return False
    return bool(np.any(valid_footprint[y0:y1, x0:x1]))


def _filter_sparse_black_windows(
    dataset: DatasetReader,
    windows: list[TileWindow],
    *,
    nodata: object,
) -> list[TileWindow]:
    if not windows:
        return []

    x_positions, y_positions = _sparse_sample_positions(dataset, windows)
    if not x_positions or not y_positions:
        return []

    sparse_valid = _read_sparse_valid_grid(dataset, x_positions, y_positions, nodata=nodata)
    x_index = {value: index for index, value in enumerate(x_positions)}
    y_index = {value: index for index, value in enumerate(y_positions)}
    return [
        window
        for window in windows
        if _window_has_sparse_valid_sample(window, sparse_valid, x_index, y_index)
    ]


def _sparse_sample_positions(
    dataset: DatasetReader,
    windows: list[TileWindow],
) -> tuple[list[int], list[int]]:
    x_positions: set[int] = set()
    y_positions: set[int] = set()
    for window in windows:
        for x_offset in _sparse_offsets(window.width):
            x = window.x + x_offset
            if 0 <= x < dataset.width:
                x_positions.add(x)
        for y_offset in _sparse_offsets(window.height):
            y = window.y + y_offset
            if 0 <= y < dataset.height:
                y_positions.add(y)
    return sorted(x_positions), sorted(y_positions)


def _sparse_offsets(size: int) -> list[int]:
    if size <= 0:
        return []
    offsets = set(range(0, size, _VALID_FOOTPRINT_STRIDE))
    offsets.add(size // 2)
    offsets.add(size - 1)
    return sorted(offsets)


def _read_sparse_valid_grid(
    dataset: DatasetReader,
    x_positions: list[int],
    y_positions: list[int],
    *,
    nodata: object,
) -> np.ndarray:
    valid = np.zeros((len(y_positions), len(x_positions)), dtype=bool)
    x_indices = np.asarray(x_positions, dtype=np.intp)
    for row_index, y in enumerate(y_positions):
        window = Window(0, y, dataset.width, 1)
        masks = dataset.read_masks(window=window)
        if masks.ndim == 2:
            masks = masks[None, :, :]
        mask_values = masks[:, 0, x_indices]
        valid_by_mask = np.any(mask_values > 0, axis=0)

        data = dataset.read(
            window=window,
            masked=False,
            fill_value=nodata,
            boundless=False,
        )
        if data.ndim == 2:
            data = data[None, :, :]
        data_values = data[:, 0, x_indices].astype(np.float32, copy=False)
        valid_by_value = np.any(np.abs(data_values) > _VALID_VALUE_EPS, axis=0)
        valid[row_index, :] = np.logical_and(valid_by_mask, valid_by_value)
    return valid


def _window_has_sparse_valid_sample(
    window: TileWindow,
    sparse_valid: np.ndarray,
    x_index: dict[int, int],
    y_index: dict[int, int],
) -> bool:
    x_offsets = _sparse_offsets(window.width)
    y_offsets = _sparse_offsets(window.height)
    for y_offset in y_offsets:
        y = window.y + y_offset
        y_grid_index = y_index.get(y)
        if y_grid_index is None:
            continue
        for x_offset in x_offsets:
            x = window.x + x_offset
            x_grid_index = x_index.get(x)
            if x_grid_index is not None and sparse_valid[y_grid_index, x_grid_index]:
                return True
    return False

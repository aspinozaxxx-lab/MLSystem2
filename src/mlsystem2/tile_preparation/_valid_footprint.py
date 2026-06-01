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
_MAX_FULL_FOOTPRINT_CELLS = 10_000_000


@dataclass(frozen=True, slots=True)
class ValidFootprintDiagnostics:
    candidate_window_count_before_valid_filter: int
    valid_window_count: int
    black_filtered_window_count: int
    valid_footprint_stride: int
    valid_footprint_total_cells: int
    valid_footprint_valid_cells: int


@dataclass(frozen=True, slots=True)
class _SparseValidDiagnostics:
    total_cells: int
    valid_cells: int


def filter_valid_windows(
    dataset: DatasetReader,
    windows: list[TileWindow],
    *,
    nodata: object,
) -> tuple[list[TileWindow], ValidFootprintDiagnostics]:
    footprint_cells = _full_footprint_cell_count(dataset)
    if footprint_cells > _MAX_FULL_FOOTPRINT_CELLS:
        valid_windows, sparse_diagnostics = _filter_sparse_black_windows(
            dataset,
            windows,
            nodata=nodata,
        )
        return valid_windows, ValidFootprintDiagnostics(
            candidate_window_count_before_valid_filter=len(windows),
            valid_window_count=len(valid_windows),
            black_filtered_window_count=len(windows) - len(valid_windows),
            valid_footprint_stride=_VALID_FOOTPRINT_STRIDE,
            valid_footprint_total_cells=sparse_diagnostics.total_cells,
            valid_footprint_valid_cells=sparse_diagnostics.valid_cells,
        )

    valid_footprint = _read_valid_footprint(dataset, nodata=nodata)
    coarse_valid_windows = [
        window for window in windows if _window_intersects_valid_footprint(window, valid_footprint)
    ]
    valid_windows, _sparse_diagnostics = _filter_sparse_black_windows(
        dataset,
        coarse_valid_windows,
        nodata=nodata,
    )
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


def _full_footprint_cell_count(dataset: DatasetReader) -> int:
    coarse_width = max(1, math.ceil(dataset.width / _VALID_FOOTPRINT_STRIDE))
    coarse_height = max(1, math.ceil(dataset.height / _VALID_FOOTPRINT_STRIDE))
    return coarse_width * coarse_height


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
) -> tuple[list[TileWindow], _SparseValidDiagnostics]:
    if not windows:
        return [], _SparseValidDiagnostics(total_cells=0, valid_cells=0)

    sparse_positions = _sparse_sample_positions_by_row(dataset, windows)
    if not sparse_positions:
        return [], _SparseValidDiagnostics(total_cells=0, valid_cells=0)

    sparse_valid, diagnostics = _read_sparse_valid_rows(
        dataset,
        sparse_positions,
        nodata=nodata,
    )
    valid_windows = [
        window
        for window in windows
        if _window_has_sparse_valid_sample(window, sparse_valid)
    ]
    return valid_windows, diagnostics


def _sparse_sample_positions_by_row(
    dataset: DatasetReader,
    windows: list[TileWindow],
) -> dict[int, set[int]]:
    positions: dict[int, set[int]] = {}
    for window in windows:
        for y_offset in _sparse_offsets(window.height):
            y = window.y + y_offset
            if not 0 <= y < dataset.height:
                continue
            row_positions = positions.setdefault(y, set())
            for x_offset in _sparse_offsets(window.width):
                x = window.x + x_offset
                if 0 <= x < dataset.width:
                    row_positions.add(x)
    return positions


def _sparse_offsets(size: int) -> list[int]:
    if size <= 0:
        return []
    offsets = set(range(0, size, _VALID_FOOTPRINT_STRIDE))
    offsets.add(size // 2)
    offsets.add(size - 1)
    return sorted(offsets)


def _read_sparse_valid_rows(
    dataset: DatasetReader,
    sparse_positions: dict[int, set[int]],
    *,
    nodata: object,
) -> tuple[dict[int, set[int]], _SparseValidDiagnostics]:
    valid: dict[int, set[int]] = {}
    total_cells = 0
    valid_cells = 0
    for y, row_positions in sorted(sparse_positions.items()):
        x_positions = sorted(row_positions)
        total_cells += len(x_positions)
        for run in _position_runs(x_positions):
            run_valid = _read_sparse_valid_run(dataset, y, run, nodata=nodata)
            if not np.any(run_valid):
                continue
            row_valid = valid.setdefault(y, set())
            for x, is_valid in zip(run, run_valid, strict=True):
                if bool(is_valid):
                    row_valid.add(x)
                    valid_cells += 1
    return valid, _SparseValidDiagnostics(total_cells=total_cells, valid_cells=valid_cells)


def _position_runs(x_positions: list[int]) -> list[list[int]]:
    if not x_positions:
        return []
    runs: list[list[int]] = [[x_positions[0]]]
    for x in x_positions[1:]:
        if x - runs[-1][-1] <= _VALID_FOOTPRINT_STRIDE:
            runs[-1].append(x)
        else:
            runs.append([x])
    return runs


def _read_sparse_valid_run(
    dataset: DatasetReader,
    y: int,
    x_positions: list[int],
    *,
    nodata: object,
) -> np.ndarray:
    x_start = x_positions[0]
    x_stop = x_positions[-1] + 1
    window = Window(x_start, y, x_stop - x_start, 1)
    local_x_indices = np.asarray([x - x_start for x in x_positions], dtype=np.intp)

    masks = dataset.read_masks(window=window)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    mask_values = masks[:, 0, local_x_indices]
    valid_by_mask = np.any(mask_values > 0, axis=0)

    data = dataset.read(
        window=window,
        masked=False,
        fill_value=nodata,
        boundless=False,
    )
    if data.ndim == 2:
        data = data[None, :, :]
    data_values = data[:, 0, local_x_indices].astype(np.float32, copy=False)
    valid_by_value = np.any(np.abs(data_values) > _VALID_VALUE_EPS, axis=0)
    return np.logical_and(valid_by_mask, valid_by_value)


def _window_has_sparse_valid_sample(
    window: TileWindow,
    sparse_valid: dict[int, set[int]],
) -> bool:
    x_offsets = _sparse_offsets(window.width)
    y_offsets = _sparse_offsets(window.height)
    for y_offset in y_offsets:
        y = window.y + y_offset
        row_valid = sparse_valid.get(y)
        if not row_valid:
            continue
        for x_offset in x_offsets:
            x = window.x + x_offset
            if x in row_valid:
                return True
    return False

"""Проверка требований к raster-файлам."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import rasterio
from affine import Affine
from rasterio.coords import BoundingBox

TOLERANCE = 1e-9


@dataclass(frozen=True)
class RasterInfo:
    scene_id: str
    path: Path
    width: int
    height: int
    band_count: int
    dtypes: tuple[str, ...]
    crs_wkt: str
    transform: Affine
    bounds: BoundingBox
    nodata: float | int
    resolution: tuple[float, float]


@dataclass(frozen=True)
class RasterValidationResult:
    rasters: list[RasterInfo]
    errors: list[str]


def validate_rasters(scene_to_image: dict[str, Path]) -> RasterValidationResult:
    errors: list[str] = []
    rasters: list[RasterInfo] = []
    baseline_crs = None
    baseline_band_count: int | None = None
    baseline_dtypes: tuple[str, ...] | None = None
    baseline_resolution: tuple[float, float] | None = None
    baseline_nodata: float | int | None = None

    for scene_id, path in scene_to_image.items():
        try:
            with rasterio.open(path) as dataset:
                transform = dataset.transform
                crs = dataset.crs
                nodata_values = tuple(dataset.nodatavals)
                if crs is None:
                    errors.append(f"У снимка нет CRS: {path}")
                    continue
                if not _is_north_up(transform):
                    errors.append(f"Снимок имеет поворот или неподдерживаемый transform: {path}")
                    continue
                if any(value is None for value in nodata_values):
                    errors.append(f"У снимка не задан nodata: {path}")
                    continue
                scene_nodata = nodata_values[0]
                if not all(_nodata_equal(scene_nodata, value) for value in nodata_values):
                    errors.append(f"У снимка nodata различается по каналам: {path}")
                    continue

                resolution = (abs(float(transform.a)), abs(float(transform.e)))
                info = RasterInfo(
                    scene_id=scene_id,
                    path=path,
                    width=dataset.width,
                    height=dataset.height,
                    band_count=dataset.count,
                    dtypes=tuple(dataset.dtypes),
                    crs_wkt=crs.to_wkt(),
                    transform=transform,
                    bounds=dataset.bounds,
                    nodata=scene_nodata,
                    resolution=resolution,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Снимок не открывается через rasterio: {path}: {exc}")
            continue

        if baseline_crs is None:
            baseline_crs = crs
            baseline_band_count = info.band_count
            baseline_dtypes = info.dtypes
            baseline_resolution = info.resolution
            baseline_nodata = info.nodata
        else:
            if crs != baseline_crs:
                errors.append(f"CRS снимка отличается от общего CRS: {path}")
            if info.band_count != baseline_band_count:
                errors.append(f"Количество каналов снимка отличается: {path}")
            if info.dtypes != baseline_dtypes:
                errors.append(f"dtype каналов снимка отличается: {path}")
            if baseline_resolution is not None and not _resolution_equal(info.resolution, baseline_resolution):
                errors.append(f"Pixel resolution снимка отличается: {path}")
            if baseline_nodata is not None and not _nodata_equal(info.nodata, baseline_nodata):
                errors.append(f"nodata снимка отличается: {path}")
        rasters.append(info)

    errors.extend(_grid_alignment_errors(rasters))
    return RasterValidationResult(rasters=rasters, errors=errors)


def _is_north_up(transform: Affine) -> bool:
    return (
        math.isclose(transform.b, 0.0, abs_tol=TOLERANCE)
        and math.isclose(transform.d, 0.0, abs_tol=TOLERANCE)
        and transform.a > 0
        and transform.e < 0
    )


def _resolution_equal(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return math.isclose(left[0], right[0], abs_tol=TOLERANCE) and math.isclose(
        left[1],
        right[1],
        abs_tol=TOLERANCE,
    )


def _nodata_equal(left: object, right: object) -> bool:
    if isinstance(left, float) and isinstance(right, float) and math.isnan(left) and math.isnan(right):
        return True
    return left == right


def _grid_alignment_errors(rasters: list[RasterInfo]) -> list[str]:
    if not rasters:
        return []
    common_left = min(raster.bounds.left for raster in rasters)
    common_top = max(raster.bounds.top for raster in rasters)
    resolution = rasters[0].resolution
    errors: list[str] = []
    for raster in rasters:
        x_offset = (raster.transform.c - common_left) / resolution[0]
        y_offset = (common_top - raster.transform.f) / resolution[1]
        if not _is_integer_offset(x_offset) or not _is_integer_offset(y_offset):
            errors.append(f"Снимок не выровнен по общей grid-мозаике: {raster.path}")
    return errors


def _is_integer_offset(value: float) -> bool:
    return math.isclose(value, round(value), abs_tol=TOLERANCE)

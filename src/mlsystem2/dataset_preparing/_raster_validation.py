"""Проверка блокирующих требований к raster-файлам."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.enums import MaskFlags

from ._constants import TARGET_CRS

TOLERANCE = 1e-12


@dataclass(frozen=True)
class RasterInfo:
    scene_id: str
    path: Path
    width: int
    height: int
    band_count: int
    dtypes: tuple[str, ...]
    crs: CRS
    transform: Affine
    bounds: BoundingBox


@dataclass(frozen=True)
class RasterValidationResult:
    rasters: list[RasterInfo]
    errors: list[str]


def validate_rasters(scene_to_image: dict[str, Path]) -> RasterValidationResult:
    errors: list[str] = []
    rasters: list[RasterInfo] = []
    baseline_band_count: int | None = None
    baseline_dtypes: tuple[str, ...] | None = None

    for scene_id, path in scene_to_image.items():
        try:
            with rasterio.open(path) as dataset:
                crs = dataset.crs
                transform = dataset.transform
                nodata_values = tuple(dataset.nodatavals)
                if crs is None:
                    errors.append(f"У снимка нет CRS: {path}")
                    continue
                if crs != CRS.from_string(TARGET_CRS):
                    errors.append(f"CRS снимка должен быть {TARGET_CRS}: {path}")
                    continue
                if not _is_valid_geotransform(transform):
                    errors.append(f"У снимка некорректный geotransform: {path}")
                    continue
                if all(value is None for value in nodata_values) and not _has_usable_mask(dataset):
                    errors.append(f"У снимка нет usable mask или nodata: {path}")
                    continue

                info = RasterInfo(
                    scene_id=scene_id,
                    path=path,
                    width=dataset.width,
                    height=dataset.height,
                    band_count=dataset.count,
                    dtypes=tuple(dataset.dtypes),
                    crs=crs,
                    transform=transform,
                    bounds=dataset.bounds,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Снимок не открывается через rasterio: {path}: {exc}")
            continue

        if baseline_band_count is None:
            baseline_band_count = info.band_count
            baseline_dtypes = info.dtypes
        else:
            if info.band_count != baseline_band_count:
                errors.append(f"Количество каналов снимка отличается: {path}")
            if info.dtypes != baseline_dtypes:
                errors.append(f"dtype каналов снимка отличается: {path}")
        rasters.append(info)

    return RasterValidationResult(rasters=rasters, errors=errors)


def _is_valid_geotransform(transform: Affine) -> bool:
    values = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    determinant = transform.a * transform.e - transform.b * transform.d
    return not math.isclose(determinant, 0.0, abs_tol=TOLERANCE)


def _has_usable_mask(dataset: rasterio.io.DatasetReader) -> bool:
    flag_sets = [set(flags) for flags in dataset.mask_flag_enums]
    if flag_sets and all(MaskFlags.all_valid not in flags for flags in flag_sets):
        return True

    sample_height = min(dataset.height, 256)
    sample_width = min(dataset.width, 256)
    if sample_height == 0 or sample_width == 0:
        return False
    mask = dataset.dataset_mask(out_shape=(sample_height, sample_width))
    return bool(mask.min() < mask.max())

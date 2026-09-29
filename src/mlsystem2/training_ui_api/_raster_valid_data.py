"""Точная обрезка векторных геометрий по валидным пикселям TIFF."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from affine import Affine
from rasterio.enums import ColorInterp, Resampling

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import transform as window_transform
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid

from .contracts import TrainingUIAPIError


_VALID_FOOTPRINT_MAX_SIDE = 4096
_VALID_FOOTPRINT_SIMPLIFY_CELLS = 0.75


def valid_data_footprint(image_path: Path) -> BaseGeometry:
    try:
        status = image_path.stat()
    except OSError as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать TIFF {image_path.name}: {exc}") from exc
    return _cached_valid_data_footprint(
        str(image_path.resolve()),
        status.st_mtime_ns,
        status.st_size,
    )


@lru_cache(maxsize=64)
def _cached_valid_data_footprint(
    image_path: str,
    _modified_ns: int,
    _size_bytes: int,
) -> BaseGeometry:
    try:
        with rasterio.open(image_path) as source:
            if source.width <= 0 or source.height <= 0:
                raise TrainingUIAPIError(f"TIFF не содержит пикселей: {Path(image_path).name}")
            scale = min(
                1.0,
                _VALID_FOOTPRINT_MAX_SIDE / max(source.width, source.height),
            )
            sample_width = max(1, int(round(source.width * scale)))
            sample_height = max(1, int(round(source.height * scale)))
            valid_mask = (
                source.dataset_mask(
                    out_shape=(sample_height, sample_width),
                    resampling=Resampling.nearest,
                )
                > 0
            )
            # GDAL может предпочесть nodata непрозрачных RGB-каналов и проигнорировать alpha.
            # Для видимого контура прозрачность должна ограничивать любую маску TIFF.
            if ColorInterp.alpha in source.colorinterp:
                alpha_band = source.colorinterp.index(ColorInterp.alpha) + 1
                valid_mask &= source.read(alpha_band, out_shape=(sample_height, sample_width),
                                          resampling=Resampling.nearest) > 0
            if not bool(valid_mask.any()):
                raise TrainingUIAPIError(
                    f"TIFF не содержит валидных пикселей: {Path(image_path).name}"
                )
            mask_transform = source.transform * Affine.scale(
                source.width / sample_width,
                source.height / sample_height,
            )
            if bool(valid_mask.all()):
                footprint: BaseGeometry = Polygon(
                    (
                        source.transform * (0, 0),
                        source.transform * (source.width, 0),
                        source.transform * (source.width, source.height),
                        source.transform * (0, source.height),
                    )
                )
            else:
                parts = [
                    shape(geometry)
                    for geometry, value in shapes(
                        valid_mask.astype("uint8", copy=False),
                        mask=valid_mask,
                        transform=mask_transform,
                    )
                    if int(value) == 1
                ]
                footprint = _polygonal_geometry(unary_union(parts))
                tolerance = (
                    max(
                        abs(mask_transform.a),
                        abs(mask_transform.b),
                        abs(mask_transform.d),
                        abs(mask_transform.e),
                    )
                    * _VALID_FOOTPRINT_SIMPLIFY_CELLS
                )
                if tolerance > 0:
                    footprint = _polygonal_geometry(
                        footprint.simplify(tolerance, preserve_topology=True)
                    )
    except TrainingUIAPIError:
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF {Path(image_path).name}: {exc}") from exc
    if footprint.is_empty or footprint.area <= 0:
        raise TrainingUIAPIError(
            f"Не удалось построить footprint валидных данных: {Path(image_path).name}"
        )
    return footprint


def clip_geometries_to_valid_data(
    dataset: rasterio.io.DatasetReader,
    geometries: Sequence[BaseGeometry],
) -> tuple[BaseGeometry, ...]:
    """Обрезать геометрии по нативной ``dataset_mask`` без полной загрузки растра."""

    results: list[list[BaseGeometry]] = [[] for _ in geometries]
    indexed_geometries: list[BaseGeometry] = []
    source_positions: list[int] = []
    for position, geometry in enumerate(geometries):
        polygonal = _polygonal_geometry(geometry)
        if polygonal.is_empty or polygonal.area <= 0:
            continue
        indexed_geometries.append(polygonal)
        source_positions.append(position)
    if not indexed_geometries:
        return tuple(GeometryCollection() for _ in geometries)

    tree = STRtree(indexed_geometries)
    for _, window in dataset.block_windows(1):
        block_footprint = _window_footprint(dataset, window)
        tree_positions = tuple(
            int(value) for value in tree.query(block_footprint, predicate="intersects")
        )
        if not tree_positions:
            continue

        candidates: list[tuple[int, BaseGeometry]] = []
        for tree_position in tree_positions:
            clipped_to_block = _polygonal_geometry(
                indexed_geometries[tree_position].intersection(block_footprint)
            )
            if clipped_to_block.is_empty or clipped_to_block.area <= 0:
                continue
            candidates.append((tree_position, clipped_to_block))
        if not candidates:
            continue

        mask = dataset.dataset_mask(window=window)
        expected_shape = (int(window.height), int(window.width))
        if mask.shape != expected_shape:
            raise ValueError(
                "Размер dataset_mask не совпадает с размером блока TIFF: "
                f"{mask.shape} != {expected_shape}."
            )
        valid_pixels = mask != 0
        if not bool(np.any(valid_pixels)):
            continue
        valid_footprint = (
            block_footprint
            if bool(np.all(valid_pixels))
            else _valid_pixels_footprint(
                valid_pixels,
                transform=window_transform(window, dataset.transform),
            )
        )
        if valid_footprint.is_empty:
            continue

        for tree_position, clipped_to_block in candidates:
            clipped_to_valid_data = _polygonal_geometry(
                clipped_to_block.intersection(valid_footprint)
            )
            if clipped_to_valid_data.is_empty or clipped_to_valid_data.area <= 0:
                continue
            results[source_positions[tree_position]].append(clipped_to_valid_data)

    return tuple(
        _polygonal_geometry(unary_union(parts)) if parts else GeometryCollection()
        for parts in results
    )


def _window_footprint(
    dataset: rasterio.io.DatasetReader,
    window: rasterio.windows.Window,
) -> Polygon:
    transform = window_transform(window, dataset.transform)
    return Polygon(
        (
            transform * (0, 0),
            transform * (window.width, 0),
            transform * (window.width, window.height),
            transform * (0, window.height),
        )
    )


def _valid_pixels_footprint(
    valid_pixels: np.ndarray,
    *,
    transform: rasterio.Affine,
) -> BaseGeometry:
    parts = [
        _polygonal_geometry(shape(geometry))
        for geometry, value in shapes(
            valid_pixels.astype(np.uint8, copy=False),
            mask=valid_pixels,
            transform=transform,
        )
        if int(value) == 1
    ]
    polygonal_parts = [part for part in parts if not part.is_empty and part.area > 0]
    return (
        _polygonal_geometry(unary_union(polygonal_parts))
        if polygonal_parts
        else GeometryCollection()
    )


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    repaired = make_valid(geometry) if not geometry.is_valid else geometry
    if isinstance(repaired, (Polygon, MultiPolygon)):
        return repaired
    if isinstance(repaired, GeometryCollection):
        parts: list[Polygon] = []
        for item in repaired.geoms:
            polygonal = _polygonal_geometry(item)
            if isinstance(polygonal, Polygon):
                parts.append(polygonal)
            elif isinstance(polygonal, MultiPolygon):
                parts.extend(polygonal.geoms)
        if not parts:
            return GeometryCollection()
        merged = unary_union(parts)
        return make_valid(merged) if not merged.is_valid else merged
    return GeometryCollection()


__all__ = ["clip_geometries_to_valid_data", "valid_data_footprint"]

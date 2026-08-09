"""Точная обрезка векторных геометрий по валидным пикселям TIFF."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import transform as window_transform
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid


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


__all__ = ["clip_geometries_to_valid_data"]

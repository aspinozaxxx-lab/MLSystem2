"""Пиксельное сравнение готовых псевдоразметок на общей нативной сетке TIFF."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.enums import ColorInterp
from rasterio.features import rasterize, shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import Polygon, shape
from shapely.strtree import STRtree
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._pseudo_viewer import _ready_result, _result_images, _scene_id
from ._raster_http import raster_revision
from ._raster_valid_data import _polygonal_geometry
from .contracts import (
    PseudoMarkupComparisonCounts, PseudoMarkupComparisonLayers, PseudoMarkupComparisonRequest,
    PseudoMarkupPixelComparison, TrainingUIAPIError,
)


def _inputs(session, config, request):
    entries, warnings = [], []
    for result_id in request.result_ids:
        result = _ready_result(session, result_id)
        paths, problems = _result_images(session, config, result)
        warnings.extend(problems)
        path = Path(result.geojson_file.path).resolve()
        stat = path.stat()
        entries.append((result_id, (str(path), stat.st_mtime_ns, stat.st_size),
                        {_scene_id(image): image for image in paths}))
    return entries, list(dict.fromkeys(warnings))


def _revision(path, scene_id, request):
    expected = request.scene_revisions.get(scene_id)
    if expected is not None and expected != raster_revision(path):
        raise HTTPException(412, "Исходный снимок изменился. Перечитайте сравнение.")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def pseudo_comparison_counts(
    session: Session, config: TrainingUIAPIConfig, request: PseudoMarkupComparisonRequest,
) -> PseudoMarkupComparisonCounts:
    if len(request.result_ids) != 2:
        raise TrainingUIAPIError("Пиксельные счётчики доступны для двух активных псевдоразметок.")
    entries, warnings = _inputs(session, config, request)
    shared = entries[0][2].keys() & entries[1][2].keys()
    counts = {}
    for scene_id in sorted(shared):
        path = entries[0][2][scene_id]
        try:
            key = _revision(path, scene_id, request)
            counts[scene_id] = PseudoMarkupPixelComparison(**_count_scene(key, tuple(item[1] for item in entries)))
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            raise TrainingUIAPIError(f"Не удалось сравнить снимок {path.name}.") from exc
    if not shared:
        warnings.append("У двух активных разметок нет общих доступных TIFF. Пиксельные счётчики не рассчитаны.")
    return PseudoMarkupComparisonCounts(
        result_ids=request.result_ids, scenes=counts,
        total=PseudoMarkupPixelComparison(**{name: sum(getattr(value, name) for value in counts.values())
                                             for name in ("intersection", "only_first", "only_second")}),
        warnings=warnings,
    )


def pseudo_comparison_layers(
    session: Session, config: TrainingUIAPIConfig, scene_id: str, request: PseudoMarkupComparisonRequest,
) -> PseudoMarkupComparisonLayers:
    entries, _ = _inputs(session, config, request)
    available = [item for item in entries if scene_id in item[2]]
    if not available:
        raise TrainingUIAPIError("Снимок не входит в активные псевдоразметки или больше недоступен.")
    path = available[0][2][scene_id]
    try:
        key = _revision(path, scene_id, request)
        features, values = _layer_scene(key, tuple((str(item[0]), item[1]) for item in available))
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать слои снимка {path.name}.") from exc
    return PseudoMarkupComparisonLayers(
        scene_id=scene_id, available_result_ids=[item[0] for item in available],
        geojson={"type": "FeatureCollection", "features": features},
        counts=PseudoMarkupPixelComparison(**values) if len(entries) == len(available) == 2 else None,
    )


@lru_cache(maxsize=12)
def _geometries(file_key):
    data = json.loads(Path(file_key[0]).read_text(encoding="utf-8-sig"))
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("Файл псевдоразметки повреждён.")
    # Каноническая псевдоразметка, как и обычный просмотр, хранится в EPSG:4326.
    geometries = []
    for feature in data["features"]:
        if feature.get("geometry"):
            geometry = _polygonal_geometry(shape(feature["geometry"]))
            if not geometry.is_empty:
                geometries.append(geometry)
    return STRtree(geometries)


def _scene_trees(source, file_keys):
    corners = Polygon([source.transform * pixel for pixel in
                       ((0, 0), (source.width, 0), (source.width, source.height), (0, source.height))])
    geographic = shape(transform_geom(source.crs, "EPSG:4326", corners.__geo_interface__))
    trees = []
    for key in file_keys:
        tree = _geometries(key)
        indices = tree.query(geographic, predicate="intersects")
        native = [_polygonal_geometry(shape(transform_geom("EPSG:4326", source.crs,
                                                          tree.geometries[index].__geo_interface__)))
                  for index in indices]
        trees.append(STRtree([geometry for geometry in native if not geometry.is_empty]))
    return trees


def _windows(source, trees):
    # Даже полосовой TIFF не заставляет выделять маску размером со весь снимок.
    for row in range(0, source.height, 512):
        for column in range(0, source.width, 512):
            window = Window(column, row, min(512, source.width - column), min(512, source.height - row))
            transform = window_transform(window, source.transform)
            footprint = Polygon([transform * pixel for pixel in
                                 ((0, 0), (window.width, 0), (window.width, window.height), (0, window.height))])
            candidates = [[tree.geometries[index] for index in tree.query(footprint, predicate="intersects")]
                          for tree in trees]
            if not any(candidates):
                continue
            valid = source.dataset_mask(window=window) > 0
            if ColorInterp.alpha in source.colorinterp:
                valid &= source.read(source.colorinterp.index(ColorInterp.alpha) + 1, window=window) > 0
            masks = [rasterize([(geometry, 1) for geometry in items], out_shape=valid.shape,
                               transform=transform, dtype="uint8").astype(bool) & valid
                     if items else np.zeros(valid.shape, dtype=bool) for items in candidates]
            yield transform, masks


def _counts(masks):
    first, second = masks
    return dict(intersection=int(np.count_nonzero(first & second)),
                only_first=int(np.count_nonzero(first & ~second)),
                only_second=int(np.count_nonzero(second & ~first)))


@lru_cache(maxsize=128)
def _count_scene(image_key, file_keys):
    total = dict(intersection=0, only_first=0, only_second=0)
    with rasterio.open(image_key[0]) as source:
        trees = _scene_trees(source, file_keys)
        for _, masks in _windows(source, trees):
            for name, value in _counts(masks).items():
                total[name] += value
    return total


@lru_cache(maxsize=8)
def _layer_scene(image_key, entries):
    features = []
    total = dict(intersection=0, only_first=0, only_second=0)
    with rasterio.open(image_key[0]) as source:
        trees = _scene_trees(source, tuple(key for _, key in entries))
        for transform, masks in _windows(source, trees):
            coverage = np.sum(masks, axis=0, dtype=np.uint8)
            regions = [(mask, "layer", result_id) for (result_id, _), mask in zip(entries, masks)]
            if len(masks) > 1:
                regions.append((coverage >= 2, "intersection", None))
                regions.extend((mask & (coverage == 1), "difference", result_id)
                               for (result_id, _), mask in zip(entries, masks))
            if len(masks) == 2:
                for name, value in _counts(masks).items():
                    total[name] += value
            for mask, kind, result_id in regions:
                for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
                    if value:
                        features.append({"type": "Feature", "properties": {"comparison_kind": kind,
                                                                           "comparison_result_id": result_id},
                                         "geometry": transform_geom(source.crs, "EPSG:3857", geometry)})
    return features, total

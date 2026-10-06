"""Пиксельное сравнение готовых псевдоразметок на общей нативной сетке TIFF."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock

import numpy as np
import rasterio
from fastapi import HTTPException
from rasterio.enums import ColorInterp
from rasterio.features import geometry_window, rasterize, shapes
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds, transform_geom
from affine import Affine
from rasterio.windows import Window, intersect, transform as window_transform
from shapely.geometry import Polygon, box, shape
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


class _ComparisonCancelled(Exception):
    """Просмотр закрыт или выбран другой снимок; частичный расчёт не кэшируется."""


def _check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise _ComparisonCancelled()


_COUNT_CACHE = OrderedDict()
_COUNT_LOCK = Lock()


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
    *, cancel=None,
) -> PseudoMarkupComparisonCounts:
    if len(request.result_ids) != 2:
        raise TrainingUIAPIError("Пиксельные счётчики доступны для двух активных псевдоразметок.")
    entries, warnings = _inputs(session, config, request)
    shared = entries[0][2].keys() & entries[1][2].keys()
    if request.scene_id is not None:
        if request.scene_id not in shared:
            raise TrainingUIAPIError("Снимок не входит в обе активные псевдоразметки.")
        shared = {request.scene_id}
    counts = {}
    for scene_id in sorted(shared):
        _check_cancelled(cancel)
        path = entries[0][2][scene_id]
        try:
            key = _revision(path, scene_id, request)
            counts[scene_id] = PseudoMarkupPixelComparison(**_count_scene(key, tuple(item[1] for item in entries), cancel))
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
    *, cancel=None,
) -> PseudoMarkupComparisonLayers:
    entries, _ = _inputs(session, config, request)
    available = [item for item in entries if scene_id in item[2]]
    if not available:
        raise TrainingUIAPIError("Снимок не входит в активные псевдоразметки или больше недоступен.")
    path = available[0][2][scene_id]
    try:
        key = _revision(path, scene_id, request)
        entries_key = tuple((str(item[0]), item[1]) for item in available)
        _check_cancelled(cancel)
        if request.viewport is not None:
            features = _viewport_layers(key, entries_key, request.viewport, cancel)
            values = None
        else:
            features, values = _layer_scene(key, entries_key)
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать слои снимка {path.name}.") from exc
    return PseudoMarkupComparisonLayers(
        scene_id=scene_id, available_result_ids=[item[0] for item in available],
        geojson={"type": "FeatureCollection", "features": features},
        counts=PseudoMarkupPixelComparison(**values) if values is not None and len(entries) == len(available) == 2 else None,
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
    # В географической системе прямые края проекционного TIFF могут стать дугами.
    # Плотно преобразованный bbox сохраняет кандидатов у края; точный отбор идёт
    # уже в нативной системе в каждом окне, поэтому лишние кандидаты безопасны.
    geographic = box(*transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21))
    trees = []
    for key in file_keys:
        tree = _geometries(key)
        indices = tree.query(geographic, predicate="intersects")
        native = [_polygonal_geometry(shape(transform_geom("EPSG:4326", source.crs,
                                                          tree.geometries[index].__geo_interface__)))
                  for index in indices]
        native = [geometry for geometry in native if not geometry.is_empty]
        # GDAL получает готовые координаты: сложный полигон может пересекать сотни
        # окон, но его __geo_interface__ достаточно построить один раз на снимок.
        trees.append((STRtree(native), tuple(geometry.__geo_interface__ for geometry in native)))
    return trees


def _masks(source, trees, window, step=1):
    height, width = math.ceil(window.height / step), math.ceil(window.width / step)
    transform = window_transform(window, source.transform) * Affine.scale(window.width / width, window.height / height)
    footprint = Polygon([transform * pixel for pixel in ((0, 0), (width, 0), (width, height), (0, height))])
    candidates = [[coordinates[index] for index in tree.query(footprint, predicate="intersects")]
                  for tree, coordinates in trees]
    if not any(candidates):
        return None
    valid = source.dataset_mask(window=window, out_shape=(height, width)) > 0
    if ColorInterp.alpha in source.colorinterp:
        valid &= source.read(source.colorinterp.index(ColorInterp.alpha) + 1, window=window,
                             out_shape=(height, width)) > 0
    masks = [_raster_mask(items, valid.shape, transform) & valid
             if items else np.zeros(valid.shape, dtype=bool) for items in candidates]
    return transform, masks


def _raster_mask(geometries, dimensions, transform):
    # Вывод в уже открытый GDAL dataset обходит промежуточный MemoryDataset,
    # который создаёт облачную сессию для каждого numpy-массива. Всё остаётся в RAM.
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=dimensions[1], height=dimensions[0], count=1,
                         dtype="uint8", transform=transform) as target:
            rasterize([(geometry, 1) for geometry in geometries], dst_path=target, transform=transform)
            return target.read(1).astype(bool)


def _windows(source, trees, cancel=None):
    # Даже полосовой TIFF не заставляет выделять маску размером со весь снимок.
    for row in range(0, source.height, 512):
        for column in range(0, source.width, 512):
            _check_cancelled(cancel)
            window = Window(column, row, min(512, source.width - column), min(512, source.height - row))
            value = _masks(source, trees, window)
            if value is not None:
                yield value


def _counts(masks):
    first, second = masks
    return dict(intersection=int(np.count_nonzero(first & second)),
                only_first=int(np.count_nonzero(first & ~second)),
                only_second=int(np.count_nonzero(second & ~first)))


def _count_scene(image_key, file_keys, cancel=None):
    cache_key = image_key, file_keys
    _check_cancelled(cancel)
    with _COUNT_LOCK:
        cached = _COUNT_CACHE.get(cache_key)
        if cached is not None:
            _COUNT_CACHE.move_to_end(cache_key)
            return cached
    total = dict(intersection=0, only_first=0, only_second=0)
    # Одна среда GDAL на запрос, без повторной инициализации при преобразовании CRS.
    with rasterio.Env(), rasterio.open(image_key[0]) as source:
        trees = _scene_trees(source, file_keys)
        for _, masks in _windows(source, trees, cancel):
            for name, value in _counts(masks).items():
                total[name] += value
    _check_cancelled(cancel)
    with _COUNT_LOCK:
        _COUNT_CACHE[cache_key] = total
        _COUNT_CACHE.move_to_end(cache_key)
        while len(_COUNT_CACHE) > 128:
            _COUNT_CACHE.popitem(last=False)
    return total


@lru_cache(maxsize=8)
def _layer_scene(image_key, entries):
    features = []
    total = dict(intersection=0, only_first=0, only_second=0)
    with rasterio.Env(), rasterio.open(image_key[0]) as source:
        trees = _scene_trees(source, tuple(key for _, key in entries))
        for transform, masks in _windows(source, trees):
            if len(masks) == 2:
                for name, value in _counts(masks).items():
                    total[name] += value
            features.extend(_region_features(source.crs, transform, masks, entries))
    return features, total


def _region_features(crs, transform, masks, entries, cancel=None):
    coverage = np.sum(masks, axis=0, dtype=np.uint8)
    regions = [(mask, "layer", result_id) for (result_id, _), mask in zip(entries, masks)]
    if len(masks) > 1:
        regions.append((coverage >= 2, "intersection", None))
        regions.extend((mask & (coverage == 1), "difference", result_id)
                       for (result_id, _), mask in zip(entries, masks))
    for mask, kind, result_id in regions:
        _check_cancelled(cancel)
        for geometry, value in shapes(mask.astype("uint8"), mask=mask, transform=transform):
            _check_cancelled(cancel)
            if value:
                yield {"type": "Feature", "properties": {"comparison_kind": kind, "comparison_result_id": result_id},
                       "geometry": transform_geom(crs, "EPSG:3857", geometry)}


def _viewport_layers(image_key, entries, viewport, cancel):
    """Слои по масштабу карты; точные счётчики считаются отдельным запросом."""
    features = []
    with rasterio.Env(), rasterio.open(image_key[0]) as source:
        # Большой обзор мира может выходить за область обратного преобразования
        # проекции TIFF. Сначала ограничиваем его охватом снимка в проекции карты.
        scene_bounds = transform_bounds(source.crs, "EPSG:3857", *source.bounds, densify_pts=21)
        visible = (max(viewport.bounds[0], scene_bounds[0]), max(viewport.bounds[1], scene_bounds[1]),
                   min(viewport.bounds[2], scene_bounds[2]), min(viewport.bounds[3], scene_bounds[3]))
        if visible[0] >= visible[2] or visible[1] >= visible[3]:
            return features
        bounds = transform_bounds("EPSG:3857", source.crs, *visible, densify_pts=21)
        window = geometry_window(source, [box(*bounds).__geo_interface__], boundless=True)
        if not intersect(window, Window(0, 0, source.width, source.height)):
            return features
        window = window.intersection(Window(0, 0, source.width, source.height))
        # Степень двойки сохраняет сетку обзора при сдвиге; вблизи шаг равен одному
        # и показываются центры исходных пикселей. Окно и память ограничены размером карты.
        width = max(1, viewport.width * (visible[2] - visible[0]) / (viewport.bounds[2] - viewport.bounds[0]))
        height = max(1, viewport.height * (visible[3] - visible[1]) / (viewport.bounds[3] - viewport.bounds[1]))
        step = 2 ** max(0, math.ceil(math.log2(max(window.width / width, window.height / height)) - 1e-6))
        left = max(0, math.floor(window.col_off / step) * step)
        top = max(0, math.floor(window.row_off / step) * step)
        right = min(source.width, math.ceil((window.col_off + window.width) / step) * step)
        bottom = min(source.height, math.ceil((window.row_off + window.height) / step) * step)
        window = Window(left, top, right - left, bottom - top)
        _check_cancelled(cancel)
        trees = _scene_trees(source, tuple(key for _, key in entries))
        value = _masks(source, trees, window, step)
        if value is None:
            return features
        transform, masks = value
        features.extend(_region_features(source.crs, transform, masks, entries, cancel))
    return features

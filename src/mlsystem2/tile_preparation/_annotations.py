"""Загрузка GeoJSON-разметки и пространственный индекс."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform as shapely_transform
from shapely.strtree import STRtree

from .contracts import TilePreparationError


class AnnotationIndex:
    def __init__(self, geometries: list[Polygon | MultiPolygon]) -> None:
        self._geometries = geometries
        self._tree = STRtree(geometries) if geometries else None

    def query_bounds(self, bounds: tuple[float, float, float, float]) -> list[Polygon | MultiPolygon]:
        if self._tree is None:
            return []

        window_geometry = box(*bounds)
        candidates = self._tree.query(window_geometry)
        result: list[Polygon | MultiPolygon] = []
        for candidate in candidates:
            geometry = self._resolve_candidate(candidate)
            if geometry is not None and geometry.intersects(window_geometry):
                result.append(geometry)
        return result

    def _resolve_candidate(self, candidate: object) -> Polygon | MultiPolygon | None:
        if isinstance(candidate, np.integer):
            return self._geometries[int(candidate)]
        if isinstance(candidate, int):
            return self._geometries[candidate]
        if isinstance(candidate, (Polygon, MultiPolygon)):
            return candidate
        return None


def load_annotation_index(
    annotation_file: str | Path,
    vrt_crs: str | None,
) -> AnnotationIndex:
    path = Path(annotation_file)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TilePreparationError(f"Не удалось прочитать GeoJSON-разметку: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TilePreparationError(f"Не удалось разобрать GeoJSON-разметку: {path}") from exc

    if payload.get("type") != "FeatureCollection":
        raise TilePreparationError("GeoJSON-разметка должна быть FeatureCollection")

    target_crs = _crs_from_user_input(vrt_crs)
    source_crs = _geojson_crs(payload) or target_crs
    transformer = _build_transformer(source_crs, target_crs)

    geometries: list[Polygon | MultiPolygon] = []
    for feature in payload.get("features", []):
        geometry_payload = feature.get("geometry") if isinstance(feature, dict) else None
        if geometry_payload is None:
            continue
        geometry = _safe_geometry(geometry_payload)
        if geometry is None:
            continue
        if transformer is not None:
            geometry = shapely_transform(transformer.transform, geometry)
        geometries.append(geometry)
    return AnnotationIndex(geometries)


def _safe_geometry(geometry_payload: dict[str, object]) -> Polygon | MultiPolygon | None:
    try:
        geometry = shape(geometry_payload)
    except Exception:
        return None

    if not isinstance(geometry, (Polygon, MultiPolygon)):
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        repaired = geometry.buffer(0)
        if not isinstance(repaired, (Polygon, MultiPolygon)) or repaired.is_empty:
            return None
        geometry = repaired
    return geometry


def _geojson_crs(payload: dict[str, object]) -> CRS | None:
    crs_payload = payload.get("crs")
    if not isinstance(crs_payload, dict):
        return None

    properties = crs_payload.get("properties")
    if not isinstance(properties, dict):
        return None

    name = properties.get("name") or properties.get("href")
    if not isinstance(name, str) or not name:
        return None
    return _crs_from_user_input(name)


def _crs_from_user_input(value: object) -> CRS | None:
    if value is None:
        return None
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise TilePreparationError(f"Некорректная CRS разметки: {value}") from exc


def _build_transformer(source_crs: CRS | None, target_crs: CRS | None) -> Transformer | None:
    if source_crs is None or target_crs is None or source_crs == target_crs:
        return None
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)

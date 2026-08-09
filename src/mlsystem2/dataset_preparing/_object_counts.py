"""Подсчет объектов по сценам."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Any

import rasterio
from rasterio.crs import CRS
from shapely.geometry import MultiPolygon, Polygon, shape

from ._scene_matching import scene_basename, scene_match_key, scene_stem

SCENE_PROPERTY_FIELDS = (
    "scene",
    "scene_id",
    "scene_name",
    "image",
    "image_name",
    "image_id",
    "filename",
    "file",
    "tif",
    "tiff",
    "raster",
    "source",
    "source_file",
    "src",
)
PER_IMAGE_ROLE_PROPERTY = "_mlsystem2_role"
PER_IMAGE_POSITIVE_ROLE = "positive"
PER_IMAGE_HARD_NEGATIVE_ROLE = "hard_negative"


@dataclass
class LoadedFeature:
    properties: dict[str, Any]
    geometry: Any | None = None


@dataclass
class SceneObjectCount:
    scene_name: str
    image_path: Path | None
    object_count: int


@dataclass(frozen=True)
class ImageGeometryScore:
    image_path: Path
    object_count: int
    distance_to_annotation: float


def count_per_image_annotation_roles(
    annotation_path: Path,
    image_path: Path | None = None,
) -> tuple[int, int]:
    payload = _load_json(annotation_path)
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise ValueError(f"GeoJSON должен быть FeatureCollection: {annotation_path}")
    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON должен содержать массив features: {annotation_path}")

    if image_path is not None:
        annotation_crs = _required_geojson_crs(payload, annotation_path)
        try:
            with rasterio.open(image_path) as dataset:
                image_crs = dataset.crs
        except rasterio.errors.RasterioError as exc:
            raise ValueError(f"Не удалось открыть TIFF {image_path}: {exc}") from exc
        if image_crs is None:
            raise ValueError(f"У TIFF отсутствует CRS: {image_path}")
        if annotation_crs != image_crs:
            raise ValueError(
                f"CRS GeoJSON ({annotation_crs}) не совпадает с CRS TIFF "
                f"({image_crs}): {annotation_path}"
            )

    positive = 0
    hard_negative = 0
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict):
            raise ValueError(f"Feature #{index} должен быть объектом: {annotation_path}")
        raw_properties = feature.get("properties")
        if raw_properties is None:
            properties: dict[str, Any] = {}
        elif isinstance(raw_properties, dict):
            properties = raw_properties
        else:
            raise ValueError(
                f"properties Feature #{index} должен быть объектом: {annotation_path}"
            )
        role = properties.get(PER_IMAGE_ROLE_PROPERTY, PER_IMAGE_POSITIVE_ROLE)
        if role == PER_IMAGE_POSITIVE_ROLE:
            positive += 1
        elif role == PER_IMAGE_HARD_NEGATIVE_ROLE:
            hard_negative += 1
        else:
            raise ValueError(
                f"Feature #{index} содержит неизвестную роль {role!r}: {annotation_path}"
            )
        geometry_payload = feature.get("geometry")
        try:
            geometry = shape(geometry_payload)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Некорректная геометрия Feature #{index}: {annotation_path}"
            ) from exc
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise ValueError(
                f"Feature #{index} должен содержать Polygon или MultiPolygon: "
                f"{annotation_path}"
            )
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise ValueError(
                f"Геометрия Feature #{index} пуста или невалидна: {annotation_path}"
            )
    return positive, hard_negative


def _required_geojson_crs(payload: dict[str, Any], annotation_path: Path) -> CRS:
    raw_crs = payload.get("crs")
    value: Any = raw_crs
    if isinstance(raw_crs, dict):
        properties = raw_crs.get("properties")
        if isinstance(properties, dict):
            value = properties.get("name") or properties.get("href")
        if not value:
            value = raw_crs.get("name")
    if not value:
        raise ValueError(f"В GeoJSON должен быть явно указан CRS снимка: {annotation_path}")
    try:
        return CRS.from_user_input(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Некорректный CRS GeoJSON {value!r}: {annotation_path}") from exc


def count_objects_per_scene(
    scene_names: list[str],
    scene_to_image: dict[str, Path],
    annotation_path: Path,
) -> list[SceneObjectCount]:
    payload = _load_json(annotation_path)
    counts = _count_from_simple_mapping(scene_names, payload)
    if counts is None:
        features = _load_features(payload)
        counts = _count_by_properties(scene_names, features)
        if features and sum(counts.values()) == 0:
            geometry_counts = _count_by_geometry(scene_names, scene_to_image, features, annotation_path)
            if sum(geometry_counts.values()) > 0:
                counts = geometry_counts

    return [
        SceneObjectCount(
            scene_name=scene,
            image_path=scene_to_image.get(scene),
            object_count=max(0, int(counts.get(scene, 0))),
        )
        for scene in scene_names
    ]


def score_images_by_annotation_geometry(
    image_paths: list[Path],
    annotation_path: Path,
) -> dict[Path, ImageGeometryScore]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from shapely.geometry import box
    except Exception:  # noqa: BLE001
        return {}

    payload = _load_json(annotation_path)
    features = _load_features(payload)
    geometries = [feature.geometry for feature in features if feature.geometry is not None]
    if not geometries:
        return {}

    annotation_crs = _load_geojson_crs(annotation_path)
    if annotation_crs is None:
        max_abs = max(max(map(abs, geometry.bounds)) for geometry in geometries)
        annotation_crs = "EPSG:3857" if max_abs > 1000 else "EPSG:4326"

    min_x = min(geometry.bounds[0] for geometry in geometries)
    min_y = min(geometry.bounds[1] for geometry in geometries)
    max_x = max(geometry.bounds[2] for geometry in geometries)
    max_y = max(geometry.bounds[3] for geometry in geometries)
    annotation_bounds = box(min_x, min_y, max_x, max_y)

    scores: dict[Path, ImageGeometryScore] = {}
    for image_path in image_paths:
        try:
            with rasterio.open(image_path) as dataset:
                scene_crs = str(dataset.crs) if dataset.crs else annotation_crs
                bounds = tuple(dataset.bounds)
                if scene_crs != annotation_crs:
                    bounds = transform_bounds(scene_crs, annotation_crs, *bounds, densify_pts=21)
                scene_bounds = box(*bounds)
        except Exception:  # noqa: BLE001
            continue

        object_count = sum(1 for geometry in geometries if geometry.intersects(scene_bounds))
        distance = scene_bounds.distance(annotation_bounds) if annotation_bounds.is_valid else inf
        scores[image_path] = ImageGeometryScore(
            image_path=image_path,
            object_count=object_count,
            distance_to_annotation=float(distance),
        )
    return scores


def _load_json(annotation_path: Path) -> Any:
    return json.loads(Path(annotation_path).read_text(encoding="utf-8-sig"))


def _count_from_simple_mapping(scene_names: list[str], payload: Any) -> Counter[str] | None:
    if not isinstance(payload, dict) or payload.get("type") in {"FeatureCollection", "Feature"}:
        return None

    lookup = _scene_name_lookup(scene_names)
    counts: Counter[str] = Counter()
    matched_any = False
    for raw_scene, raw_value in payload.items():
        matched_scene = _feature_scene_match(str(raw_scene), lookup)
        if matched_scene is None:
            continue
        matched_any = True
        counts[matched_scene] += _object_count_from_value(raw_value)
    return counts if matched_any else None


def _object_count_from_value(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, dict):
        for key in ("objects", "features", "annotations"):
            raw_items = value.get(key)
            if isinstance(raw_items, list):
                return len(raw_items)
        for key in ("object_count", "count"):
            raw_count = value.get(key)
            if isinstance(raw_count, int):
                return max(0, raw_count)
    return 0


def _load_features(payload: Any) -> list[LoadedFeature]:
    try:
        from shapely.geometry import shape
    except Exception:  # noqa: BLE001
        shape = None

    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        raw_features = payload.get("features") or []
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        raw_features = [payload]
    elif isinstance(payload, list):
        raw_features = payload
    else:
        return []

    loaded: list[LoadedFeature] = []
    for raw_feature in raw_features:
        if not isinstance(raw_feature, dict):
            continue
        raw_properties = raw_feature.get("properties") or {}
        properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
        geometry_payload = raw_feature.get("geometry")
        geometry = None
        if geometry_payload:
            if shape is None:
                geometry = geometry_payload
            else:
                try:
                    candidate = shape(geometry_payload)
                except Exception:  # noqa: BLE001
                    candidate = None
                if candidate is not None and not candidate.is_empty and candidate.is_valid:
                    geometry = candidate
        loaded.append(LoadedFeature(properties=properties, geometry=geometry))
    return loaded


def _count_by_properties(scene_names: list[str], features: list[LoadedFeature]) -> Counter[str]:
    lookup = _scene_name_lookup(scene_names)
    counts: Counter[str] = Counter()
    for feature in features:
        feature_scene = _extract_scene_name_from_feature_properties(feature.properties)
        if not feature_scene:
            continue
        matched_scene = _feature_scene_match(feature_scene, lookup)
        if matched_scene:
            counts[matched_scene] += 1
    return counts


def _count_by_geometry(
    scene_names: list[str],
    scene_to_image: dict[str, Path],
    features: list[LoadedFeature],
    annotation_path: Path,
) -> Counter[str]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from shapely.geometry import box
    except Exception:  # noqa: BLE001
        return Counter()

    annotation_crs = _load_geojson_crs(annotation_path)
    valid_features = [
        (index, feature.geometry)
        for index, feature in enumerate(features)
        if feature.geometry is not None
    ]
    if not valid_features:
        return Counter()
    if annotation_crs is None:
        max_abs = max(max(map(abs, geometry.bounds)) for _, geometry in valid_features)
        annotation_crs = "EPSG:3857" if max_abs > 1000 else "EPSG:4326"

    counts: Counter[str] = Counter()
    seen_features: set[int] = set()
    for scene in scene_names:
        image_path = scene_to_image.get(scene)
        if image_path is None:
            continue
        try:
            with rasterio.open(image_path) as dataset:
                scene_crs = str(dataset.crs) if dataset.crs else annotation_crs
                bounds = tuple(dataset.bounds)
                if scene_crs != annotation_crs:
                    bounds = transform_bounds(scene_crs, annotation_crs, *bounds, densify_pts=21)
                scene_bounds = box(*bounds)
        except Exception:  # noqa: BLE001
            continue
        for index, geometry in valid_features:
            if index in seen_features:
                continue
            if geometry.intersects(scene_bounds):
                counts[scene] += 1
                seen_features.add(index)
    return counts


def _load_geojson_crs(annotation_path: Path) -> str | None:
    payload = _load_json(annotation_path)
    if not isinstance(payload, dict):
        return None
    crs = payload.get("crs")
    if isinstance(crs, dict):
        properties = crs.get("properties") or {}
        name = properties.get("name") or crs.get("name")
        if name:
            value = str(name)
            match = re.search(r"EPSG[:/](\d+)", value, flags=re.IGNORECASE)
            return f"EPSG:{match.group(1)}" if match else value
    if isinstance(crs, str):
        return crs
    return None


def _extract_scene_name_from_feature_properties(properties: dict[str, Any]) -> str | None:
    for field_name in SCENE_PROPERTY_FIELDS:
        if field_name not in properties or properties[field_name] is None:
            continue
        value = str(properties[field_name]).strip()
        if not value:
            continue
        embedded = re.search(r"([^\\/\s;,\"]+\.(?:tif|tiff))", value, flags=re.IGNORECASE)
        if embedded:
            return scene_basename(embedded.group(1))
        basename = scene_basename(value)
        if basename:
            return basename
    return None


def _scene_name_lookup(scene_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for scene in scene_names:
        for key in {
            scene_basename(scene),
            scene_basename(scene).casefold(),
            scene_stem(scene),
            scene_stem(scene).casefold(),
            scene_match_key(scene),
        }:
            lookup.setdefault(key, scene)
    return lookup


def _feature_scene_match(feature_scene_name: str, lookup: dict[str, str]) -> str | None:
    return (
        lookup.get(scene_basename(feature_scene_name))
        or lookup.get(scene_basename(feature_scene_name).casefold())
        or lookup.get(scene_stem(feature_scene_name))
        or lookup.get(scene_stem(feature_scene_name).casefold())
        or lookup.get(scene_match_key(feature_scene_name))
    )

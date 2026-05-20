"""Подсчет объектов по сценам."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class LoadedFeature:
    properties: dict[str, Any]
    geometry: Any | None = None


@dataclass
class SceneObjectCount:
    scene_name: str
    image_path: Path | None
    object_count: int


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

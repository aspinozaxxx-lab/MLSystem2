"""Зоны разметки и виртуальные сцены без копирования исходных TIFF."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import rasterio
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from .contracts import AnnotationRegion, DatasetPreparationError, PreparedScene


def annotation_regions(payload: dict, class_slugs: list[str] | None = None):
    """None — весь снимок; пустой список — зоны есть, общей территории нет."""
    zones = []
    used_ids = set()
    for index, feature in enumerate(payload.get("features", [])):
        properties = feature.get("properties") or {}
        if properties.get("_mlsystem2_role") != "annotation_zone":
            continue
        try:
            geometry = shape(feature["geometry"])
        except Exception as exc:
            raise DatasetPreparationError("Некорректная геометрия размеченной зоны") from exc
        if not isinstance(geometry, (Polygon, MultiPolygon)) or not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
            raise DatasetPreparationError("Размеченная зона должна быть непустым валидным Polygon/MultiPolygon")
        slug = properties.get("_mlsystem2_class")
        if slug is not None and slug not in (class_slugs or []):
            raise DatasetPreparationError(f"Неизвестный класс размеченной зоны: {slug}")
        zone_id = str(feature.get("id") or properties.get("_mlsystem2_origin_key") or
                      hashlib.sha256(json.dumps([index, mapping(geometry)], sort_keys=True).encode()).hexdigest())
        if zone_id in used_ids:
            raise DatasetPreparationError(f"Повторяющийся ID размеченной зоны: {zone_id}")
        used_ids.add(zone_id)
        zones.append((zone_id, slug, geometry))
    if not zones:
        return None
    common = None
    for slug in class_slugs or []:
        parts = [geometry for _, target, geometry in zones if target in (None, slug)]
        if parts:
            coverage = unary_union(parts)
            common = coverage if common is None else common.intersection(coverage)
    result = []
    for zone_id, _, geometry in zones:
        if common is not None:
            geometry = geometry.intersection(common)
        geometry = _polygonal(geometry)
        if not geometry.is_empty and geometry.area > 0:
            result.append(AnnotationRegion(zone_id=zone_id, geometry=mapping(geometry)))
    return result


def _polygonal(geometry):
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    return unary_union([part for part in getattr(geometry, "geoms", [])
                        if isinstance(part, (Polygon, MultiPolygon))])


def expand_annotation_scenes(scenes, class_slugs, warnings):
    result = []
    counts = {}
    for scene in scenes:
        payload = json.loads(Path(scene.annotation_file).read_text(encoding="utf-8-sig"))
        regions = annotation_regions(payload, class_slugs)
        if regions is None:
            result.append(scene)
            counts[scene.scene_id] = (0, 1)
            continue
        zone_count = sum((item.get("properties") or {}).get("_mlsystem2_role") == "annotation_zone"
                         for item in payload.get("features", []))
        prepared = []
        with rasterio.open(scene.image_path) as source:
            footprint = Polygon([source.transform * point for point in
                                 [(0, 0), (source.width, 0), (source.width, source.height), (0, source.height)]])
            inverse = ~source.transform
            for region in regions:
                geometry = _polygonal(shape(region.geometry).intersection(footprint))
                if geometry.is_empty or geometry.area <= 0:
                    continue
                pixel_geometry = transform(lambda x, y, z=None: inverse * (x, y), geometry)
                left, top, right, bottom = pixel_geometry.bounds
                x, y = max(0, math.floor(left)), max(0, math.floor(top))
                width = min(source.width, math.ceil(right)) - x
                height = min(source.height, math.ceil(bottom)) - y
                if width <= 0 or height <= 0:
                    continue
                prepared.append(PreparedScene(**{
                    **scene.model_dump(), "scene_id": f"{scene.scene_id}::zone:{region.zone_id}",
                    "parent_scene_id": scene.scene_id, "zone_id": region.zone_id,
                    "region_geometry": mapping(geometry), "region_window": (x, y, width, height),
                }))
        counts[scene.scene_id] = (zone_count, len(prepared))
        if len(prepared) < zone_count:
            warnings.append(f"{scene.scene_id}: зон {zone_count}, пригодных виртуальных сцен {len(prepared)}; пустые пересечения пропущены")
        result.extend(prepared)
    if not result:
        raise DatasetPreparationError("Размеченные зоны не содержат общей территории, пригодной для обучения")
    return result, counts

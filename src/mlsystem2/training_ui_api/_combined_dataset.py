"""Детерминированная сборка комбинированных per-image multiclass-датасетов."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid

from mlsystem2.dataset_preparing.api import per_image_annotation_name
from mlsystem2.dataset_preparing.contracts import (
    DatasetClassDefinition,
    DatasetManifest,
    DatasetSourceRevision,
)

from ._markup_export import find_intersecting_images
from .contracts import TrainingUIAPIError


@dataclass(frozen=True, slots=True)
class _SourceFeature:
    geometry_wgs84: BaseGeometry
    properties: dict[str, Any]
    feature_id: Any
    origin_key: str
    origin_hash: str
    source_path: str
    role: str
    class_slug: str | None


@dataclass(frozen=True, slots=True)
class CombinedDatasetBuild:
    files: dict[str, dict[str, Any]]
    manifest: DatasetManifest
    class_counts: dict[str, int]
    hard_negative_count: int
    warnings: tuple[str, ...]


def build_combined_dataset(
    *,
    manifest: DatasetManifest,
    repo_root: Path,
    images_root: Path,
    code_revision: str | None = None,
    build_id: str | None = None,
    built_at: str | None = None,
) -> CombinedDatasetBuild:
    if not manifest.combined or manifest.task != "multiclass":
        raise TrainingUIAPIError("Пересборка доступна только комбинированному multiclass-датасету.")
    root = repo_root.resolve()
    class_by_slug = {item.slug: item for item in manifest.classes}
    positives: list[_SourceFeature] = []
    hard_negatives: list[_SourceFeature] = []
    warnings_list: list[str] = []
    resolved_sources: list[DatasetSourceRevision] = []
    for source in manifest.sources:
        source_dir = _safe_source_dir(root, source.path)
        class_definition = class_by_slug[source.class_slug]
        file_hashes = _folder_file_hashes(source_dir)
        positive_files = [
            path
            for path in sorted(source_dir.glob("*.geojson"), key=lambda item: item.name.casefold())
            if "hard_negative" not in path.stem.casefold()
        ]
        hard_negative_files = [
            path
            for path in sorted(source_dir.glob("*.geojson"), key=lambda item: item.name.casefold())
            if "hard_negative" in path.stem.casefold()
        ]
        if len(positive_files) != 1:
            raise TrainingUIAPIError(
                f"В исходной папке {source.path} ожидается ровно один positive GeoJSON."
            )
        positives.extend(
            _load_source_features(
                positive_files[0],
                root,
                role="positive",
                class_slug=class_definition.slug,
                warnings_list=warnings_list,
            )
        )
        for path in hard_negative_files:
            hard_negatives.extend(
                _load_source_features(
                    path,
                    root,
                    role="hard_negative",
                    class_slug=None,
                    warnings_list=warnings_list,
                )
            )
        resolved_sources.append(
            DatasetSourceRevision(
                path=PurePosixPath(source.path).as_posix(),
                class_slug=source.class_slug,
                git_revision=_git_head(root),
                tree_revision=_tree_revision(file_hashes),
                file_hashes=file_hashes,
            )
        )

    positives = _apply_positive_priorities(positives, manifest.classes)
    positive_union = unary_union([item.geometry_wgs84 for item in positives])
    clipped_hard_negatives: list[_SourceFeature] = []
    for item in hard_negatives:
        geometry = _polygonal(item.geometry_wgs84.difference(positive_union))
        if geometry.is_empty or geometry.area <= 0:
            continue
        clipped_hard_negatives.append(replace(item, geometry_wgs84=geometry))
    all_features = [*positives, *clipped_hard_negatives]
    if not all_features:
        raise TrainingUIAPIError("После проверки исходников не осталось валидных объектов.")
    aoi = _polygonal(unary_union([item.geometry_wgs84 for item in all_features]))
    selected = find_intersecting_images(aoi, images_root)
    warnings_list.extend(selected.warnings)

    classes_payload = [item.model_dump(mode="json") for item in manifest.classes]
    files: dict[str, dict[str, Any]] = {}
    baseline_hashes: dict[str, dict[str, str]] = {}
    scene_ids: list[str] = []
    class_counts = {item.slug: 0 for item in manifest.classes}
    hard_negative_count = 0
    for image in selected.images:
        with rasterio.open(image.path) as dataset:
            if dataset.crs is None:
                warnings_list.append(f"Пропущен TIFF без CRS: {image.source_id}")
                continue
            from ._dataset_editor import _valid_data_footprint

            footprint = _valid_data_footprint(image.path)
            transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
            target_features: list[dict[str, Any]] = []
            target_hashes: dict[str, str] = {}
            for item in all_features:
                transformed = transform_geometry(transformer.transform, item.geometry_wgs84)
                clipped = _polygonal(transformed.intersection(footprint))
                if clipped.is_empty or clipped.area <= 0:
                    continue
                target_id = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"mlsystem2:{item.origin_key}:{image.source_id}",
                    )
                )
                properties = {
                    **item.properties,
                    "_mlsystem2_role": item.role,
                    "_mlsystem2_origin_key": item.origin_key,
                    "_mlsystem2_origin_hash": item.origin_hash,
                    "_mlsystem2_source_path": item.source_path,
                }
                if item.class_slug is not None:
                    properties["_mlsystem2_class"] = item.class_slug
                    class_counts[item.class_slug] += 1
                else:
                    properties.pop("_mlsystem2_class", None)
                    hard_negative_count += 1
                target = {
                    "type": "Feature",
                    "id": target_id,
                    "properties": properties,
                    "geometry": mapping(clipped),
                }
                target_features.append(target)
                target_hashes[item.origin_key] = _sha256_json(target)
            if not target_features:
                continue
            annotation_name = per_image_annotation_name(image.path)
            if annotation_name.casefold() in {name.casefold() for name in files}:
                raise TrainingUIAPIError(
                    f"Несколько TIFF дают одинаковое имя per-image GeoJSON: {annotation_name}"
                )
            files[annotation_name] = {
                "type": "FeatureCollection",
                "crs": {
                    "type": "name",
                    "properties": {"name": PyprojCRS.from_user_input(dataset.crs).to_string()},
                },
                "_mlsystem2_schema_version": manifest.schema_version,
                "_mlsystem2_task": manifest.task,
                "_mlsystem2_classes": classes_payload,
                "features": sorted(target_features, key=lambda feature: str(feature["id"])),
            }
            baseline_hashes[annotation_name] = target_hashes
            scene_ids.append(image.source_id)

    resolved_manifest = manifest.model_copy(
        update={
            "sources": resolved_sources,
            "build_id": build_id or str(uuid.uuid4()),
            "built_at": built_at or datetime.now(timezone.utc).isoformat(),
            "code_revision": code_revision,
            "scene_ids": sorted(scene_ids),
            "source_warnings": sorted(set(warnings_list)),
            "baseline_hashes": baseline_hashes,
        }
    )
    return CombinedDatasetBuild(
        files=files,
        manifest=resolved_manifest,
        class_counts=class_counts,
        hard_negative_count=hard_negative_count,
        warnings=tuple(sorted(set(warnings_list))),
    )


def feature_hash(feature: dict[str, Any]) -> str:
    return _sha256_json(feature)


def folder_file_hashes(folder: Path) -> dict[str, str]:
    return _folder_file_hashes(folder)


def tree_revision(file_hashes: dict[str, str]) -> str:
    return _tree_revision(file_hashes)


def _load_source_features(
    path: Path,
    repo_root: Path,
    *,
    role: str,
    class_slug: str | None,
    warnings_list: list[str],
) -> list[_SourceFeature]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать исходный GeoJSON {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise TrainingUIAPIError(f"Исходный GeoJSON должен быть FeatureCollection: {path}")
    crs = _payload_crs(payload, path)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    relative = path.resolve().relative_to(repo_root).as_posix()
    output: list[_SourceFeature] = []
    for index, feature in enumerate(payload.get("features") or [], start=1):
        if not isinstance(feature, dict):
            warnings_list.append(f"Пропущен некорректный объект {relative}#{index}")
            continue
        geometry_payload = feature.get("geometry")
        try:
            geometry = _polygonal(shape(geometry_payload))
        except Exception:  # noqa: BLE001
            geometry = Polygon()
        if geometry.is_empty or geometry.area <= 0:
            warnings_list.append(f"Пропущена пустая геометрия {relative}#{index}")
            continue
        geometry_wgs84 = _polygonal(transform_geometry(transformer.transform, geometry))
        if geometry_wgs84.is_empty or geometry_wgs84.area <= 0:
            warnings_list.append(f"Пропущена геометрия после преобразования {relative}#{index}")
            continue
        raw_properties = feature.get("properties")
        properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
        clean_properties = {
            key: value for key, value in properties.items() if not key.startswith("_mlsystem2_")
        }
        source_identity = feature.get("id")
        if source_identity is None:
            source_identity = next(
                (properties[key] for key in ("id", "fid") if properties.get(key) is not None),
                None,
            )
        origin_payload: dict[str, Any] = {
            "source_path": relative,
            "role": role,
            "class_slug": class_slug,
        }
        if source_identity is not None:
            origin_payload["source_id"] = source_identity
        else:
            origin_payload["feature"] = {
                "geometry": mapping(geometry),
                "properties": clean_properties,
            }
        origin_key = hashlib.sha256(_canonical_json(origin_payload).encode("utf-8")).hexdigest()
        origin_hash = hashlib.sha256(
            _canonical_json(
                {"geometry": mapping(geometry), "properties": clean_properties}
            ).encode("utf-8")
        ).hexdigest()
        output.append(
            _SourceFeature(
                geometry_wgs84=geometry_wgs84,
                properties=clean_properties,
                feature_id=source_identity,
                origin_key=origin_key,
                origin_hash=origin_hash,
                source_path=relative,
                role=role,
                class_slug=class_slug,
            )
        )
    return output


def _apply_positive_priorities(
    features: list[_SourceFeature],
    classes: list[DatasetClassDefinition],
) -> list[_SourceFeature]:
    by_slug: dict[str, list[_SourceFeature]] = {item.slug: [] for item in classes}
    for feature in features:
        if feature.class_slug in by_slug:
            by_slug[str(feature.class_slug)].append(feature)
    occupied: BaseGeometry = Polygon()
    output: list[_SourceFeature] = []
    ordered = sorted(classes, key=lambda item: (-item.priority, item.id))
    for definition in ordered:
        class_features: list[_SourceFeature] = []
        for feature in by_slug[definition.slug]:
            geometry = _polygonal(feature.geometry_wgs84.difference(occupied))
            if geometry.is_empty or geometry.area <= 0:
                continue
            class_features.append(replace(feature, geometry_wgs84=geometry))
        output.extend(class_features)
        if class_features:
            occupied = _polygonal(
                unary_union([occupied, *(item.geometry_wgs84 for item in class_features)])
            )
    return output


def _safe_source_dir(repo_root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError(f"Некорректный путь исходной папки: {relative_value}")
    path = repo_root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise TrainingUIAPIError(f"Исходная папка выходит за пределы MLMarkup: {relative_value}") from exc
    if not path.is_dir():
        raise TrainingUIAPIError(f"Исходная папка недоступна: {relative_value}")
    return path


def _payload_crs(payload: dict[str, Any], path: Path) -> PyprojCRS:
    raw = payload.get("crs")
    value: Any = raw
    if isinstance(raw, dict):
        properties = raw.get("properties")
        if isinstance(properties, dict):
            value = properties.get("name") or properties.get("href")
    if not value:
        raise TrainingUIAPIError(f"В исходном GeoJSON отсутствует CRS: {path}")
    try:
        return PyprojCRS.from_user_input(value)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(f"Некорректный CRS исходного GeoJSON {path}: {value}") from exc


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    repaired = make_valid(geometry) if not geometry.is_valid else geometry
    if isinstance(repaired, (Polygon, MultiPolygon)):
        return repaired
    polygons: list[Polygon] = []
    for part in getattr(repaired, "geoms", ()):
        if isinstance(part, Polygon):
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(part.geoms)
    return unary_union(polygons) if polygons else Polygon()


def _folder_file_hashes(folder: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(
        (item for item in folder.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(folder).as_posix().casefold(),
    ):
        result[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _tree_revision(values: dict[str, str]) -> str:
    return hashlib.sha256(_canonical_json(values).encode("utf-8")).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise TrainingUIAPIError("Не удалось определить Git-ревизию MLMarkup.")
    return result.stdout.strip()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CombinedDatasetBuild",
    "build_combined_dataset",
    "feature_hash",
    "folder_file_hashes",
    "tree_revision",
]

"""Виртуальные управляемые датасеты и их детерминированная материализация."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import rasterio
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import per_image_annotation_name
from mlsystem2.dataset_preparing.contracts import (
    DatasetClassDefinition,
    DatasetManifest,
    DatasetSourceRevision,
)

from ._config import TrainingUIAPIConfig
from ._datasets import _path_metadata, imagery_images_dir
from ._models import (
    DatasetClassRow,
    DatasetRow,
    ManagedDatasetSceneRow,
    ManagedDatasetSourceRow,
)
from .contracts import ManagedDatasetSourceInfo, TrainingUIAPIError


SOURCE_MANAGED = "managed"
_MANAGED_CACHE_FOLDER = "managed-datasets"
_MANAGED_DATASET_VERSION_ALGORITHM = 2
_MANAGED_MATERIALIZATION_ALGORITHM = 3


@dataclass(frozen=True, slots=True)
class ManagedSource:
    relation: ManagedDatasetSourceRow
    dataset: DatasetRow
    dataset_class: DatasetClassRow


@dataclass(frozen=True, slots=True)
class ManagedDatasetMaterialization:
    path: Path
    version: str
    updated_at: datetime | None
    manifest: DatasetManifest
    class_counts: dict[str, int]
    hard_negative_count: int


def managed_sources(session: Session, dataset_id: uuid.UUID) -> list[ManagedSource]:
    rows = session.execute(
        select(ManagedDatasetSourceRow, DatasetRow, DatasetClassRow)
        .join(DatasetRow, DatasetRow.id == ManagedDatasetSourceRow.source_dataset_id)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(ManagedDatasetSourceRow.managed_dataset_id == dataset_id)
        .order_by(ManagedDatasetSourceRow.object_type_id)
    ).all()
    return [ManagedSource(*row) for row in rows]


def managed_source_infos(session: Session, dataset_id: uuid.UUID) -> list[ManagedDatasetSourceInfo]:
    return [
        ManagedDatasetSourceInfo(
            dataset_key=item.dataset.key,
            dataset_name=item.dataset.name,
            class_key=item.dataset_class.key,
            class_name=item.dataset_class.name,
            priority=item.relation.priority,
            object_type_id=item.relation.object_type_id,
            object_type_slug=item.relation.object_type_slug,
            color=item.relation.color,
        )
        for item in managed_sources(session, dataset_id)
    ]


def managed_scenes(session: Session, dataset_id: uuid.UUID) -> list[ManagedDatasetSceneRow]:
    return list(
        session.scalars(
            select(ManagedDatasetSceneRow)
            .where(ManagedDatasetSceneRow.managed_dataset_id == dataset_id)
            .order_by(ManagedDatasetSceneRow.annotation_name)
        ).all()
    )


def managed_manifest(session: Session, dataset: DatasetRow) -> DatasetManifest:
    sources = managed_sources(session, dataset.id)
    if len(sources) < 2:
        raise TrainingUIAPIError(
            f"Управляемому датасету {dataset.name} требуется минимум два источника."
        )
    classes = [
        DatasetClassDefinition(
            id=item.relation.object_type_id,
            slug=item.relation.object_type_slug,
            name=item.relation.object_type_name,
            color=item.relation.color,
            priority=item.relation.priority,
        )
        for item in sources
    ]
    return DatasetManifest(
        schema_version=1,
        task="multiclass",
        combined=True,
        managed=True,
        classes=classes,
        sources=[
            DatasetSourceRevision(
                path=PurePosixPath(item.dataset.source_path).as_posix(),
                class_slug=item.relation.object_type_slug,
                dataset_key=item.dataset.key,
                priority=item.relation.priority,
                git_revision="pending",
                tree_revision="pending",
            )
            for item in sources
        ],
    )


def managed_dataset_version(
    session: Session,
    dataset: DatasetRow,
    source_root: Path,
) -> tuple[str, datetime | None]:
    values: list[dict[str, Any]] = []
    timestamps: list[datetime] = []
    root = Path(source_root).resolve()
    for item in managed_sources(session, dataset.id):
        source_path = _safe_source_path(root, item.dataset.source_path)
        updated_at, version = _path_metadata(source_path, root)
        if updated_at is not None:
            timestamps.append(updated_at)
        values.append(
            {
                "dataset_key": item.dataset.key,
                "dataset_revision": item.dataset.config_revision,
                "source_path": item.dataset.source_path,
                "source_version": version or _folder_fallback_revision(source_path),
                "priority": item.relation.priority,
                "object_type_id": item.relation.object_type_id,
                "object_type_slug": item.relation.object_type_slug,
                "object_type_name": item.relation.object_type_name,
                "color": item.relation.color,
            }
        )
    explicit_scenes = managed_scenes(session, dataset.id)
    payload = {
        "algorithm": _MANAGED_DATASET_VERSION_ALGORITHM,
        "dataset_key": dataset.key,
        "config_revision": dataset.config_revision,
        "sources": values,
        "scenes": [
            {
                "annotation_name": scene.annotation_name,
                "image_relative_path": scene.image_relative_path,
            }
            for scene in explicit_scenes
        ],
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"managed:{digest}", max(timestamps) if timestamps else None


def materialize_managed_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset: DatasetRow,
    *,
    source_root: Path,
    scope: str,
) -> ManagedDatasetMaterialization:
    if dataset.source_type != SOURCE_MANAGED:
        raise TrainingUIAPIError(f"Датасет не является управляемым: {dataset.key}")
    if scope not in {"live", "editor"}:
        raise ValueError("scope должен быть live или editor")
    class_row = session.get(DatasetClassRow, dataset.class_id)
    if class_row is None:
        raise TrainingUIAPIError(f"Класс управляемого датасета не найден: {dataset.key}")
    version, updated_at = managed_dataset_version(session, dataset, source_root)
    cache_digest = hashlib.sha256(
        (
            f"{version}:materialization-algorithm:"
            f"{_MANAGED_MATERIALIZATION_ALGORITHM}"
        ).encode("utf-8")
    ).hexdigest()
    cache_parent = (
        Path(config.stored_files_root).parent
        / _MANAGED_CACHE_FOLDER
        / scope
        / dataset.key
    )
    target = cache_parent / cache_digest
    manifest_path = target / ".mlsystem2-dataset.json"
    if manifest_path.is_file():
        return _materialization_from_cache(target, version, updated_at)

    manifest = managed_manifest(session, dataset)
    from ._combined_dataset import build_combined_dataset

    build = build_combined_dataset(
        manifest=manifest,
        repo_root=Path(source_root),
        images_root=imagery_images_dir(config.images_root, class_row.imagery_type),
        code_revision=_project_revision(config.project_root),
    )
    files = dict(build.files)
    image_paths = dict(build.image_paths)
    baseline_hashes = dict(build.manifest.baseline_hashes)
    images_root = imagery_images_dir(config.images_root, class_row.imagery_type).resolve()
    classes_payload = [item.model_dump(mode="json") for item in manifest.classes]
    for scene in managed_scenes(session, dataset.id):
        relative = PurePosixPath(scene.image_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise TrainingUIAPIError(
                f"Некорректный путь снимка управляемого датасета: {scene.image_relative_path}"
            )
        image_path = images_root.joinpath(*relative.parts).resolve()
        try:
            image_path.relative_to(images_root)
        except ValueError as exc:
            raise TrainingUIAPIError(
                f"Снимок управляемого датасета выходит за разрешённый каталог: {scene.image_relative_path}"
            ) from exc
        if not image_path.is_file():
            raise TrainingUIAPIError(
                f"Снимок управляемого датасета не найден: {scene.image_relative_path}"
            )
        expected_name = per_image_annotation_name(image_path)
        if expected_name.casefold() != scene.annotation_name.casefold():
            raise TrainingUIAPIError(
                "Имя разметки управляемого датасета не соответствует снимку: "
                f"{scene.annotation_name} != {expected_name}"
            )
        existing_image = image_paths.get(scene.annotation_name)
        if existing_image is not None and existing_image.resolve() != image_path:
            raise TrainingUIAPIError(
                f"Снимок {scene.annotation_name} неоднозначно задан в управляемом датасете."
            )
        image_paths[scene.annotation_name] = image_path
        if scene.annotation_name in files:
            continue
        try:
            with rasterio.open(image_path) as raster:
                if raster.crs is None:
                    raise TrainingUIAPIError(f"У TIFF отсутствует CRS: {image_path}")
                crs_name = raster.crs.to_string()
        except rasterio.errors.RasterioError as exc:
            raise TrainingUIAPIError(
                f"Не удалось открыть TIFF {image_path.name}: {exc}"
            ) from exc
        files[scene.annotation_name] = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_name}},
            "_mlsystem2_schema_version": manifest.schema_version,
            "_mlsystem2_task": manifest.task,
            "_mlsystem2_classes": classes_payload,
            "features": [],
        }
        baseline_hashes[scene.annotation_name] = {}
    resolved_manifest = build.manifest.model_copy(
        update={
            "scene_ids": sorted(Path(name).stem for name in files),
            "baseline_hashes": baseline_hashes,
        }
    )
    cache_parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_parent / f".{cache_digest}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for annotation_name, payload in files.items():
            _write_json(temporary / annotation_name, payload)
            image_path = image_paths[annotation_name]
            from ._dataset_editor import _footprint_geojson_payload
            from mlsystem2.dataset_preparing.api import footprint_name_for_annotation

            _write_json(
                temporary / footprint_name_for_annotation(annotation_name),
                _footprint_geojson_payload(image_path),
            )
        _write_json(
            temporary / ".mlsystem2-dataset.json",
            resolved_manifest.model_dump(mode="json"),
        )
        try:
            os.replace(temporary, target)
        except OSError:
            if not manifest_path.is_file():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return ManagedDatasetMaterialization(
        path=target,
        version=version,
        updated_at=updated_at,
        manifest=resolved_manifest,
        class_counts=build.class_counts,
        hard_negative_count=build.hard_negative_count,
    )


def invalidate_managed_cache(config: TrainingUIAPIConfig, dataset_key: str | None = None) -> None:
    root = Path(config.stored_files_root).parent / _MANAGED_CACHE_FOLDER
    if dataset_key is None:
        shutil.rmtree(root, ignore_errors=True)
        return
    for scope in ("live", "editor"):
        shutil.rmtree(root / scope / dataset_key, ignore_errors=True)


def _materialization_from_cache(
    target: Path,
    version: str,
    updated_at: datetime | None,
) -> ManagedDatasetMaterialization:
    try:
        payload = json.loads(
            (target / ".mlsystem2-dataset.json").read_text(encoding="utf-8-sig")
        )
        manifest = DatasetManifest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(f"Повреждён кэш управляемого датасета {target}: {exc}") from exc
    class_counts = {item.slug: 0 for item in manifest.classes}
    hard_negative_count = 0
    for path in target.glob("*.geojson"):
        if path.name.casefold().endswith("_footprint.geojson"):
            continue
        try:
            annotation = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        for feature in annotation.get("features") or []:
            properties = feature.get("properties") or {}
            if properties.get("_mlsystem2_role") == "hard_negative":
                hard_negative_count += 1
            else:
                slug = properties.get("_mlsystem2_class")
                if slug in class_counts:
                    class_counts[str(slug)] += 1
    return ManagedDatasetMaterialization(
        path=target,
        version=version,
        updated_at=updated_at,
        manifest=manifest,
        class_counts=class_counts,
        hard_negative_count=hard_negative_count,
    )


def _safe_source_path(root: Path, source_path: str) -> Path:
    relative = PurePosixPath(source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError(f"Некорректный путь источника: {source_path}")
    target = root.joinpath(*relative.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TrainingUIAPIError(f"Источник выходит за пределы MLMarkup: {source_path}") from exc
    if not target.is_dir():
        raise TrainingUIAPIError(f"Источник управляемого датасета недоступен: {source_path}")
    return target


def _folder_fallback_revision(path: Path) -> str:
    if not path.is_dir():
        return "missing"
    payload = {
        item.relative_to(path).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix().casefold(),
        )
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _project_revision(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    try:
        return hashlib.sha256(head.read_bytes()).hexdigest()
    except OSError:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "ManagedDatasetMaterialization",
    "ManagedSource",
    "SOURCE_MANAGED",
    "invalidate_managed_cache",
    "managed_dataset_version",
    "managed_manifest",
    "managed_source_infos",
    "managed_scenes",
    "managed_sources",
    "materialize_managed_dataset",
]

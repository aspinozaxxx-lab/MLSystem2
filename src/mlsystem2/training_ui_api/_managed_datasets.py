"""Виртуальные управляемые датасеты и их детерминированная материализация."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal

import rasterio
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import (
    footprint_name_for_annotation,
    per_image_annotation_name,
)
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
_MANAGED_MATERIALIZATION_ALGORITHM = 4
_MATERIALIZATION_SUMMARY = ".mlsystem2-materialization.json"
_MATERIALIZATION_REQUESTS = "_requests"
_MATERIALIZATION_LOCK_TIMEOUT_SECONDS = 2 * 60 * 60
_MATERIALIZATION_RETAIN_VERSIONS = 2
MaterializationStatus = Literal[
    "current",
    "queued",
    "building",
    "failed",
    "missing",
]
LOGGER = logging.getLogger(__name__)


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
    image_count: int


@dataclass(frozen=True, slots=True)
class ManagedDatasetCacheState:
    desired_version: str
    updated_at: datetime | None
    status: MaterializationStatus
    materialization: ManagedDatasetMaterialization | None
    last_ready: ManagedDatasetMaterialization | None = None
    error: str | None = None


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


def managed_dataset_cache_state(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset: DatasetRow,
    *,
    source_root: Path,
    scope: str,
    request_if_missing: bool = False,
    priority: Literal["normal", "urgent"] = "normal",
) -> ManagedDatasetCacheState:
    """Быстро проверить кэш, не выполняя материализацию в HTTP-процессе."""

    version, updated_at = managed_dataset_version(session, dataset, source_root)
    cache_parent, target, _digest = _managed_cache_paths(config, dataset.key, scope, version)
    if (target / ".mlsystem2-dataset.json").is_file():
        return ManagedDatasetCacheState(
            desired_version=version,
            updated_at=updated_at,
            status="current",
            materialization=_materialization_from_cache(
                target,
                version,
                updated_at,
                scan_missing_summary=False,
            ),
        )

    request_path = _materialization_request_path(config, dataset.key, scope)
    request_payload = _read_json_object(request_path)
    if request_if_missing:
        request_payload = _write_materialization_request(
            config,
            dataset_key=dataset.key,
            scope=scope,
            version=version,
            priority=priority,
            previous=request_payload,
        )
    lock_path = cache_parent / f".{target.name}.lock"
    if lock_path.is_file():
        status: MaterializationStatus = "building"
    elif request_payload is None:
        status = "missing"
    elif request_payload.get("last_error") and not _request_is_due(request_payload):
        status = "failed"
    else:
        status = "queued"
    return ManagedDatasetCacheState(
        desired_version=version,
        updated_at=updated_at,
        status=status,
        materialization=None,
        last_ready=_latest_ready_materialization(cache_parent, updated_at),
        error=(
            str(request_payload.get("last_error"))
            if request_payload and request_payload.get("last_error")
            else None
        ),
    )


def has_pending_managed_materialization(config: TrainingUIAPIConfig) -> bool:
    return bool(_pending_materialization_requests(config))


def process_next_managed_materialization(
    session: Session,
    config: TrainingUIAPIConfig,
) -> bool:
    """Собрать один запрошенный кэш. Вызывается только отдельным worker."""

    requests = _pending_materialization_requests(config)
    if not requests:
        return False
    request_path, payload = requests[0]
    dataset_key = str(payload.get("dataset_key") or "")
    scope = str(payload.get("scope") or "")
    dataset = session.scalar(
        select(DatasetRow).where(
            DatasetRow.key == dataset_key,
            DatasetRow.source_type == SOURCE_MANAGED,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if dataset is None or scope not in {"live", "editor"}:
        request_path.unlink(missing_ok=True)
        return True
    source_root = config.mlmarkup_root if scope == "live" else config.mlmarkup_editor_root
    started_at = time.perf_counter()
    try:
        materialized = materialize_managed_dataset(
            session,
            config,
            dataset,
            source_root=source_root,
            scope=scope,
        )
    except Exception as exc:  # noqa: BLE001
        attempts = int(payload.get("attempts") or 0) + 1
        delay = min(15 * 60, 60 * (2 ** min(attempts - 1, 4)))
        failed = {
            **payload,
            "attempts": attempts,
            "last_error": str(exc),
            "next_attempt_at": (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_json_atomic(request_path, failed)
        raise
    current_version, _current_updated_at = managed_dataset_version(
        session,
        dataset,
        source_root,
    )
    if current_version != materialized.version:
        _write_materialization_request(
            config,
            dataset_key=dataset_key,
            scope=scope,
            version=current_version,
            priority=("urgent" if payload.get("priority") == "urgent" else "normal"),
            previous=None,
        )
    else:
        request_path.unlink(missing_ok=True)
    LOGGER.info(
        "Управляемый датасет материализован: %s, scope=%s, снимков=%s, %.1f с",
        dataset_key,
        scope,
        materialized.image_count,
        time.perf_counter() - started_at,
    )
    return True


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
    cache_parent, target, cache_digest = _managed_cache_paths(
        config,
        dataset.key,
        scope,
        version,
    )
    manifest_path = target / ".mlsystem2-dataset.json"
    cache_parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_parent / f".{cache_digest}.lock"
    with _materialization_lock(lock_path, manifest_path):
        if manifest_path.is_file():
            return _materialization_from_cache(
                target,
                version,
                updated_at,
                scan_missing_summary=True,
            )
        started_at = time.perf_counter()
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
        source_footprints = _source_footprint_index(session, dataset, Path(source_root))
        reused_footprints = 0
        computed_footprints = 0
        temporary = cache_parent / f".{cache_digest}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            for annotation_name, payload in files.items():
                _write_json(temporary / annotation_name, payload)
                footprint_name = footprint_name_for_annotation(annotation_name)
                source_footprint = source_footprints.get(footprint_name.casefold())
                if source_footprint is not None:
                    shutil.copy2(source_footprint, temporary / footprint_name)
                    reused_footprints += 1
                else:
                    from ._dataset_editor import _footprint_geojson_payload

                    _write_json(
                        temporary / footprint_name,
                        _footprint_geojson_payload(image_paths[annotation_name]),
                    )
                    computed_footprints += 1
            _write_json(
                temporary / ".mlsystem2-dataset.json",
                resolved_manifest.model_dump(mode="json"),
            )
            _write_materialization_summary(
                temporary,
                version=version,
                class_counts=build.class_counts,
                hard_negative_count=build.hard_negative_count,
                image_count=len(files),
                duration_seconds=time.perf_counter() - started_at,
                reused_footprints=reused_footprints,
                computed_footprints=computed_footprints,
            )
            os.replace(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
        _prune_managed_cache(cache_parent, keep=target)
        return _materialization_from_cache(
            target,
            version,
            updated_at,
            scan_missing_summary=False,
        )


def invalidate_managed_cache(
    config: TrainingUIAPIConfig,
    dataset_key: str | None = None,
    *,
    remove_ready: bool = False,
) -> None:
    root = _managed_cache_root(config)
    if dataset_key is None:
        shutil.rmtree(root, ignore_errors=True)
        return
    for scope in ("live", "editor"):
        # Версия кэша включает source/config revision, поэтому старый готовый
        # снимок безопасно остаётся для быстрых сводок до окончания новой сборки.
        if remove_ready:
            shutil.rmtree(root / scope / dataset_key, ignore_errors=True)
        _materialization_request_path(config, dataset_key, scope).unlink(missing_ok=True)


def _materialization_from_cache(
    target: Path,
    version: str,
    updated_at: datetime | None,
    *,
    scan_missing_summary: bool,
) -> ManagedDatasetMaterialization:
    try:
        payload = json.loads((target / ".mlsystem2-dataset.json").read_text(encoding="utf-8-sig"))
        manifest = DatasetManifest.model_validate(payload)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(f"Повреждён кэш управляемого датасета {target}: {exc}") from exc
    summary = _read_json_object(target / _MATERIALIZATION_SUMMARY)
    if summary is None and scan_missing_summary:
        class_counts, hard_negative_count = _scan_materialization_counts(target, manifest)
        image_count = len(manifest.scene_ids)
        _write_materialization_summary(
            target,
            version=version,
            class_counts=class_counts,
            hard_negative_count=hard_negative_count,
            image_count=image_count,
            duration_seconds=None,
            reused_footprints=None,
            computed_footprints=None,
        )
    elif summary is not None:
        raw_counts = summary.get("class_counts")
        class_counts = {
            item.slug: max(0, int((raw_counts or {}).get(item.slug) or 0))
            for item in manifest.classes
        }
        hard_negative_count = max(0, int(summary.get("hard_negative_count") or 0))
        image_count = max(0, int(summary.get("image_count") or len(manifest.scene_ids)))
    else:
        class_counts = {item.slug: 0 for item in manifest.classes}
        hard_negative_count = 0
        image_count = len(manifest.scene_ids)
    return ManagedDatasetMaterialization(
        path=target,
        version=version,
        updated_at=updated_at,
        manifest=manifest,
        class_counts=class_counts,
        hard_negative_count=hard_negative_count,
        image_count=image_count,
    )


def _managed_cache_root(config: TrainingUIAPIConfig) -> Path:
    return Path(config.stored_files_root).parent / _MANAGED_CACHE_FOLDER


def _managed_cache_paths(
    config: TrainingUIAPIConfig,
    dataset_key: str,
    scope: str,
    version: str,
) -> tuple[Path, Path, str]:
    if scope not in {"live", "editor"}:
        raise ValueError("scope должен быть live или editor")
    cache_digest = hashlib.sha256(
        (f"{version}:materialization-algorithm:{_MANAGED_MATERIALIZATION_ALGORITHM}").encode(
            "utf-8"
        )
    ).hexdigest()
    parent = _managed_cache_root(config) / scope / dataset_key
    return parent, parent / cache_digest, cache_digest


def _materialization_request_path(
    config: TrainingUIAPIConfig,
    dataset_key: str,
    scope: str,
) -> Path:
    digest = hashlib.sha256(f"{scope}\0{dataset_key}".encode("utf-8")).hexdigest()
    return _managed_cache_root(config) / _MATERIALIZATION_REQUESTS / scope / f"{digest}.json"


def _write_materialization_request(
    config: TrainingUIAPIConfig,
    *,
    dataset_key: str,
    scope: str,
    version: str,
    priority: Literal["normal", "urgent"],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    same_version = previous is not None and previous.get("version") == version
    if same_version and (priority == "normal" or previous.get("priority") == "urgent"):
        return previous
    payload = {
        "dataset_key": dataset_key,
        "scope": scope,
        "version": version,
        "priority": (
            "urgent"
            if priority == "urgent" or (same_version and previous.get("priority") == "urgent")
            else "normal"
        ),
        "created_at": previous.get("created_at", now) if same_version and previous else now,
        "updated_at": now,
        "attempts": int(previous.get("attempts") or 0) if same_version and previous else 0,
        "last_error": previous.get("last_error") if same_version and previous else None,
        "next_attempt_at": previous.get("next_attempt_at") if same_version and previous else None,
    }
    path = _materialization_request_path(config, dataset_key, scope)
    _write_json_atomic(path, payload)
    return payload


def _pending_materialization_requests(
    config: TrainingUIAPIConfig,
) -> list[tuple[Path, dict[str, Any]]]:
    root = _managed_cache_root(config) / _MATERIALIZATION_REQUESTS
    result: list[tuple[Path, dict[str, Any]]] = []
    if not root.is_dir():
        return result
    for path in root.rglob("*.json"):
        payload = _read_json_object(path)
        if payload is None:
            path.unlink(missing_ok=True)
            continue
        if _request_is_due(payload):
            result.append((path, payload))
    result.sort(
        key=lambda item: (
            item[1].get("priority") != "urgent",
            str(item[1].get("created_at") or ""),
            item[0].as_posix(),
        )
    )
    return result


def _request_is_due(payload: dict[str, Any]) -> bool:
    raw = payload.get("next_attempt_at")
    if not raw:
        return True
    try:
        return datetime.fromisoformat(str(raw)) <= datetime.now(timezone.utc)
    except ValueError:
        return True


@contextmanager
def _materialization_lock(lock_path: Path, manifest_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if _stale_materialization_lock(lock_path):
        lock_path.unlink(missing_ok=True)
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        if manifest_path.is_file():
            yield
            return
        raise TrainingUIAPIError("Материализация управляемого датасета уже выполняется.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _stale_materialization_lock(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = _read_json_object(path) or {}
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if age >= _MATERIALIZATION_LOCK_TIMEOUT_SECONDS:
        return True
    if payload.get("host") != socket.gethostname():
        return False
    try:
        pid = int(payload.get("pid") or 0)
        if pid <= 0:
            return age >= 60
        os.kill(pid, 0)
    except (OSError, ValueError):
        return True
    return False


def _scan_materialization_counts(
    target: Path,
    manifest: DatasetManifest,
) -> tuple[dict[str, int], int]:
    class_counts = {item.slug: 0 for item in manifest.classes}
    hard_negative_count = 0
    for path in target.glob("*.geojson"):
        if path.name.casefold().endswith("_footprint.geojson"):
            continue
        annotation = _read_json_object(path)
        if annotation is None:
            continue
        for feature in annotation.get("features") or []:
            properties = feature.get("properties") or {}
            if properties.get("_mlsystem2_role") == "hard_negative":
                hard_negative_count += 1
            else:
                slug = properties.get("_mlsystem2_class")
                if slug in class_counts:
                    class_counts[str(slug)] += 1
    return class_counts, hard_negative_count


def _write_materialization_summary(
    target: Path,
    *,
    version: str,
    class_counts: dict[str, int],
    hard_negative_count: int,
    image_count: int,
    duration_seconds: float | None,
    reused_footprints: int | None,
    computed_footprints: int | None,
) -> None:
    _write_json_atomic(
        target / _MATERIALIZATION_SUMMARY,
        {
            "schema_version": 1,
            "version": version,
            "materialization_algorithm": _MANAGED_MATERIALIZATION_ALGORITHM,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "class_counts": class_counts,
            "hard_negative_count": hard_negative_count,
            "image_count": image_count,
            "duration_seconds": duration_seconds,
            "reused_footprints": reused_footprints,
            "computed_footprints": computed_footprints,
        },
    )


def _source_footprint_index(
    session: Session,
    dataset: DatasetRow,
    source_root: Path,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    root = source_root.resolve()
    for source in managed_sources(session, dataset.id):
        folder = _safe_source_path(root, source.dataset.source_path)
        for path in folder.glob("*_footprint.geojson"):
            result.setdefault(path.name.casefold(), path)
    return result


def _latest_ready_materialization(
    cache_parent: Path,
    updated_at: datetime | None,
) -> ManagedDatasetMaterialization | None:
    if not cache_parent.is_dir():
        return None
    candidates = [
        path
        for path in cache_parent.iterdir()
        if path.is_dir() and (path / ".mlsystem2-dataset.json").is_file()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for path in candidates:
        summary = _read_json_object(path / _MATERIALIZATION_SUMMARY) or {}
        version = str(summary.get("version") or f"stale:{path.name}")
        try:
            return _materialization_from_cache(
                path,
                version,
                updated_at,
                scan_missing_summary=False,
            )
        except TrainingUIAPIError:
            continue
    return None


def _prune_managed_cache(cache_parent: Path, *, keep: Path) -> None:
    candidates = [
        path
        for path in cache_parent.iterdir()
        if path.is_dir()
        and len(path.name) == 64
        and all(character in "0123456789abcdef" for character in path.name)
        and (path / ".mlsystem2-dataset.json").is_file()
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    retained = {keep, *candidates[:_MATERIALIZATION_RETAIN_VERSIONS]}
    for path in candidates:
        if path not in retained:
            shutil.rmtree(path, ignore_errors=True)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_json(temporary, payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    "ManagedDatasetCacheState",
    "ManagedDatasetMaterialization",
    "ManagedSource",
    "SOURCE_MANAGED",
    "has_pending_managed_materialization",
    "invalidate_managed_cache",
    "managed_dataset_cache_state",
    "managed_dataset_version",
    "managed_manifest",
    "managed_source_infos",
    "managed_scenes",
    "managed_sources",
    "materialize_managed_dataset",
    "process_next_managed_materialization",
]

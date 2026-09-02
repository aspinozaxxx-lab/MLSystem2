"""Git-backed редактор per-image датасетов MLMarkup."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote

import rasterio
from affine import Affine
from pyproj import CRS as PyprojCRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from mlsystem2.dataset_preparing.api import (
    footprint_name_for_annotation,
    is_per_image_footprint_name,
    load_dataset_manifest,
    per_image_annotation_name,
)

from ._config import TrainingUIAPIConfig
from ._combined_dataset import (
    build_combined_dataset,
    feature_hash,
    folder_file_hashes,
    tree_revision,
)
from ._dataset_catalog import (
    dataset_class_row,
    find_managed_dataset,
    list_managed_datasets,
)
from ._datasets import RASTER_SUFFIXES, build_image_index, build_per_image_index
from ._external_models import external_model_payload
from ._models import (
    DatasetClassRow,
    DatasetEditorDraftRow,
    DatasetRow,
    JobRow,
    ManagedDatasetSceneRow,
    ManagedDatasetSourceRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TrainingResultRow,
)
from ._managed_datasets import (
    SOURCE_MANAGED,
    invalidate_managed_cache,
    managed_dataset_cache_state,
    managed_sources,
)
from ._pseudolabel import _select_model
from ._queueing import DATASET_EDITOR_PSEUDO_OPERATION, next_queue_position
from ._test_samples import current_primary_training_result
from .contracts import (
    DatasetEditorDatasetInfo,
    DatasetEditorDatasetListResponse,
    DatasetEditorCopyResult,
    DatasetEditorDiscardDraftsResult,
    DatasetEditorDraftInfo,
    DatasetEditorDraftSummary,
    DatasetEditorObjectType,
    DatasetEditorMutationResult,
    DatasetEditorPublicationInfo,
    DatasetEditorPseudoMarkupInfo,
    DatasetEditorRasterBrowserResponse,
    DatasetEditorRasterFolderInfo,
    DatasetEditorRasterInfo,
    DatasetEditorRebuildChange,
    DatasetEditorRebuildPreview,
    DatasetEditorRebuildResult,
    DatasetEditorSceneDetail,
    DatasetEditorSceneInfo,
    DatasetEditorSceneListResponse,
    DatasetEditorUserDraftInfo,
    DatasetEditorUserDraftListResponse,
    DatasetInfo,
    JobSource,
    JobStatus,
    JobType,
    ResultStatus,
    TrainingUIAPIError,
)


_ROLE_PROPERTY = "_mlsystem2_role"
_CLASS_PROPERTY = "_mlsystem2_class"
_ROLES = {"positive", "hard_negative"}
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")
_VALID_FOOTPRINT_MAX_SIDE = 4096
_VALID_FOOTPRINT_SIMPLIFY_CELLS = 0.75
_URGENT_PRIORITY = "urgent"
_EDITOR_SYNC_TTL_SECONDS = 60.0
_EDITOR_PSEUDO_ALGORITHM_VERSION = 1
_MANAGED_OWNER_PROPERTY = "_mlsystem2_managed_dataset_key"
_EDITOR_RASTER_CRS = PyprojCRS.from_epsg(3857)


class DatasetEditorConflict(RuntimeError):
    """Blob revision устарела относительно origin."""


class DatasetEditorGitError(RuntimeError):
    """Git-клон редактора недоступен или операция Git завершилась ошибкой."""


def list_editor_datasets(
    session: Session,
    config: TrainingUIAPIConfig,
) -> DatasetEditorDatasetListResponse:
    with _editor_lock(config):
        _synchronize_editor_clone_if_stale(config)
        result: list[DatasetEditorDatasetInfo] = []
        for dataset in list_managed_datasets(
            session,
            config,
            include_custom=False,
            materialize_managed=False,
        ):
            try:
                if dataset.managed:
                    scene_count = _managed_editor_scene_count(
                        session,
                        config,
                        dataset.key,
                    )
                else:
                    source_dir = _editor_source_dir(config, dataset)
            except TrainingUIAPIError:
                continue
            if dataset.managed:
                row = session.scalar(
                    select(DatasetRow).where(
                        DatasetRow.key == dataset.key,
                        DatasetRow.deleted_at.is_(None),
                    )
                )
                if row is None:
                    continue
                try:
                    cache_state = managed_dataset_cache_state(
                        session,
                        config,
                        row,
                        source_root=config.mlmarkup_editor_root,
                        scope="editor",
                    )
                    dataset = dataset.model_copy(
                        update={
                            "materialization_status": cache_state.status,
                            "materialized_version": (
                                cache_state.materialization.version
                                if cache_state.materialization is not None
                                else None
                            ),
                            "materialization_error": cache_state.error,
                        }
                    )
                except TrainingUIAPIError as exc:
                    dataset = dataset.model_copy(
                        update={
                            "materialization_status": "failed",
                            "materialization_error": str(exc),
                        }
                    )
                primary = _editor_effective_training_result(
                    session,
                    dataset.key,
                    dataset.class_key or dataset.key,
                )
                result.append(
                    _editor_dataset_info(
                        dataset,
                        scene_count,
                        primary_training_result_id=(primary.id if primary is not None else None),
                    )
                )
                continue
            if source_dir.exists() and not source_dir.is_dir():
                continue
            if source_dir.is_dir() and _direct_files(source_dir, ".txt"):
                continue
            geojson_files = _direct_annotation_files(source_dir)
            has_dataset_subdirectories = source_dir.is_dir() and any(
                item.is_dir() and not item.name.startswith(".") for item in source_dir.iterdir()
            )
            if not geojson_files and has_dataset_subdirectories:
                continue
            if dataset.annotations_dir is None and not geojson_files:
                continue
            primary = _editor_effective_training_result(
                session,
                dataset.key,
                dataset.class_key or dataset.key,
            )
            result.append(
                _editor_dataset_info(
                    dataset,
                    len(geojson_files),
                    primary_training_result_id=(primary.id if primary is not None else None),
                )
            )
        result.sort(key=lambda item: (item.class_name.casefold(), item.dataset_name.casefold()))
        return DatasetEditorDatasetListResponse(datasets=result)


def list_editor_user_drafts(
    session: Session,
    *,
    username: str,
) -> DatasetEditorUserDraftListResponse:
    rows = session.execute(
        select(
            DatasetEditorDraftRow.dataset_key,
            DatasetEditorDraftRow.deleted,
            DatasetEditorDraftRow.updated_at,
            DatasetRow.name,
            DatasetClassRow.key,
            DatasetClassRow.name,
        )
        .join(DatasetRow, DatasetRow.key == DatasetEditorDraftRow.dataset_key)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(
            DatasetEditorDraftRow.username == username,
            DatasetRow.deleted_at.is_(None),
        )
    ).all()
    grouped: dict[str, DatasetEditorUserDraftInfo] = {}
    for dataset_key, deleted, updated_at, dataset_name, class_key, class_name in rows:
        current = grouped.get(dataset_key)
        if current is None:
            grouped[dataset_key] = DatasetEditorUserDraftInfo(
                dataset_key=dataset_key,
                class_key=class_key,
                class_name=class_name,
                dataset_name=dataset_name,
                scene_count=1,
                deleted_scene_count=int(deleted),
                updated_at=updated_at,
            )
            continue
        grouped[dataset_key] = current.model_copy(
            update={
                "scene_count": current.scene_count + 1,
                "deleted_scene_count": current.deleted_scene_count + int(deleted),
                "updated_at": max(current.updated_at, updated_at),
            }
        )
    drafts = sorted(
        grouped.values(),
        key=lambda item: (
            -item.updated_at.timestamp(),
            item.class_name.casefold(),
            item.dataset_name.casefold(),
        ),
    )
    return DatasetEditorUserDraftListResponse(drafts=drafts)


def list_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    username: str,
) -> DatasetEditorSceneListResponse:
    with _editor_lock(config):
        dataset, source_dir = _editor_dataset_context(
            session,
            config,
            dataset_key,
            allow_missing=True,
        )
        scenes = _attach_draft_summaries(
            session,
            dataset,
            _scene_infos(config, dataset, source_dir),
            username,
        )
        primary = _editor_effective_training_result(
            session,
            dataset.key,
            dataset.class_key or dataset.key,
        )
        return DatasetEditorSceneListResponse(
            dataset=_editor_dataset_info(
                dataset,
                len(scenes),
                primary_training_result_id=(primary.id if primary is not None else None),
            ),
            scenes=scenes,
        )


def editor_scene_detail(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
    *,
    username: str,
) -> DatasetEditorSceneDetail:
    with _editor_lock(config):
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
        annotation_path = _annotation_path(source_dir, annotation_name)
        image_path = _matched_image_path(dataset, source_dir, annotation_name)
        published_geojson = _reproject_editor_geojson(
            _read_geojson(annotation_path),
            image_path,
        )
        footprint = _valid_data_footprint(image_path)
        draft_row = _editor_draft_row(
            session,
            dataset.key,
            scene.annotation_name,
            username,
        )
        return DatasetEditorSceneDetail(
            scene=scene,
            geojson=_clip_geojson_to_footprint(published_geojson, footprint),
            valid_data_footprint=dict(mapping(footprint)),
            draft=(
                _draft_info(
                    draft_row,
                    dataset,
                    scene.revision,
                    geojson=_reproject_editor_geojson(
                        dict(draft_row.geojson),
                        image_path,
                    ),
                )
                if draft_row is not None
                else None
            ),
        )


def save_editor_draft(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
    *,
    base_revision: str,
    geojson: dict[str, Any],
    deleted: bool = False,
    username: str,
) -> DatasetEditorDraftInfo:
    """Сохранить проверенный черновик в БД без Git-коммита и инференса."""

    with _editor_lock(config):
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
        annotation_path = _annotation_path(source_dir, scene.annotation_name)
        image_path = _matched_image_path(dataset, source_dir, scene.annotation_name)
        previous_payload = _reproject_editor_geojson(
            _read_geojson(annotation_path),
            image_path,
        )
        normalized_geojson = _normalize_editor_geojson(
            geojson,
            previous_payload,
            dataset,
            image_path=image_path,
        )
        normalized_geojson = _clip_geojson_to_footprint(
            normalized_geojson,
            _valid_data_footprint(image_path),
        )
        _validate_editor_geojson(normalized_geojson, image_path, dataset)
        _validate_preserved_properties(previous_payload, normalized_geojson)
        row = _editor_draft_row(session, dataset.key, scene.annotation_name, username)
        if row is None:
            created_at = datetime.now(timezone.utc)
            row = DatasetEditorDraftRow(
                dataset_key=dataset.key,
                annotation_name=scene.annotation_name,
                username=username,
                base_revision=base_revision,
                geojson=normalized_geojson,
                deleted=deleted,
                created_at=created_at,
                updated_at=created_at,
            )
            session.add(row)
        else:
            row.base_revision = base_revision
            row.geojson = normalized_geojson
            row.deleted = deleted
            row.updated_at = datetime.now(timezone.utc)
        session.flush()
        return _draft_info(row, dataset, scene.revision)


def discard_editor_drafts(
    session: Session,
    dataset_key: str,
    *,
    username: str,
    annotation_name: str | None = None,
) -> DatasetEditorDiscardDraftsResult:
    conditions = [
        DatasetEditorDraftRow.dataset_key == dataset_key,
        DatasetEditorDraftRow.username == username,
    ]
    if annotation_name is not None:
        conditions.append(
            DatasetEditorDraftRow.annotation_name == _safe_annotation_name(annotation_name)
        )
    rows = session.scalars(select(DatasetEditorDraftRow).where(*conditions)).all()
    for row in rows:
        session.delete(row)
    session.flush()
    return DatasetEditorDiscardDraftsResult(deleted_count=len(rows))


def publish_editor_drafts(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    username: str,
) -> DatasetEditorMutationResult:
    rows = session.scalars(
        select(DatasetEditorDraftRow)
        .where(
            DatasetEditorDraftRow.dataset_key == dataset_key,
            DatasetEditorDraftRow.username == username,
        )
        .order_by(DatasetEditorDraftRow.annotation_name)
    ).all()
    if not rows:
        raise TrainingUIAPIError("Нет сохранённых черновиков для публикации")
    return publish_editor_scenes(
        session,
        config,
        dataset_key,
        scenes=[
            (row.annotation_name, row.base_revision, dict(row.geojson))
            for row in rows
            if not row.deleted
        ],
        deletions=[(row.annotation_name, row.base_revision) for row in rows if row.deleted],
        username=username,
    )


def editor_scene_pseudo_markup(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
    *,
    ensure: bool,
    retry: bool = False,
) -> DatasetEditorPseudoMarkupInfo:
    """Вернуть готовый фрагмент или идемпотентно поставить срочный инференс TIFF."""

    with _editor_lock(config):
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
        image_path = _matched_image_path(dataset, source_dir, annotation_name).resolve()

    class_key = dataset.class_key or dataset.key
    primary = _editor_effective_training_result(session, dataset.key, class_key)
    if primary is None:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            message="Для класса нет успешной сети.",
        )
    compatibility_error = _editor_pseudo_compatibility_error(dataset, primary)
    if compatibility_error is not None:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=primary.id,
            model_name=primary.model_name,
            message=compatibility_error,
        )

    full_result = _latest_covering_pseudo_result(
        session,
        dataset,
        primary.id,
        image_path,
    )
    if full_result is not None and full_result.geojson_file is not None:
        payload = _pseudo_geojson_for_image(Path(full_result.geojson_file.path), image_path)
        return DatasetEditorPseudoMarkupInfo(
            status="ready",
            source="dataset",
            training_result_id=primary.id,
            model_name=primary.model_name,
            object_count=len(payload["features"]),
            geojson=payload,
        )

    raster_revision = _raster_revision(image_path)
    if not ensure:
        job = _latest_editor_pseudo_job(
            session,
            dataset.key,
            scene.annotation_name,
            primary.id,
            raster_revision,
        )
        if job is not None:
            return _editor_pseudo_job_info(
                session,
                job,
                image_path=image_path,
                training_result=primary,
            )
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=primary.id,
            model_name=primary.model_name,
            message="Для снимка ещё нет псевдоразметки.",
        )

    selected = _select_model(
        session,
        config,
        class_key,
        required=False,
        preferred_training_result_id=primary.id,
    )
    if selected is None or selected.result.id != primary.id:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=primary.id,
            model_name=primary.model_name,
            message="Checkpoint текущей основной сети недоступен для инференса.",
        )
    if dataset.input_channels is not None and selected.input_channels != dataset.input_channels:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=primary.id,
            model_name=primary.model_name,
            message="Основная сеть несовместима с каналами снимков датасета.",
        )
    root = _dataset_images_root(dataset).resolve()
    image_relative = image_path.relative_to(root).with_suffix("").as_posix()
    source_job = session.get(JobRow, primary.job_id) if primary.job_id is not None else None
    tile_size = _positive_integer(
        (source_job.config or {}).get("tile_preparation.tile_size") if source_job else None,
        768,
    )
    inference_config_hash = _editor_pseudo_inference_config_hash(selected, tile_size)
    dedup_key = _editor_pseudo_dedup_key(
        dataset.key,
        scene.annotation_name,
        primary.id,
        raster_revision,
        inference_config_hash,
    )
    existing = session.scalar(select(JobRow).where(JobRow.dedup_key == dedup_key))
    if existing is not None:
        info = _editor_pseudo_job_info(
            session,
            existing,
            image_path=image_path,
            training_result=primary,
        )
        if info.status != "failed" or not retry:
            return info
        _reset_editor_pseudo_job(session, existing)
        return _editor_pseudo_job_info(
            session,
            existing,
            image_path=image_path,
            training_result=primary,
        )
    row = JobRow(
        type=JobType.INFERENCE.value,
        source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.INFERENCE, JobSource.MANUAL),
        dataset_key=dataset.key,
        dataset_version=dataset.version,
        dataset_name=dataset.name,
        training_dataset_name=selected.dataset_name,
        inference_dataset_name=f"{dataset.name}: {scene.image_name}",
        model_name=primary.model_name,
        architecture=primary.architecture,
        tile_size=tile_size,
        dedup_key=dedup_key,
        config={
            "operation": DATASET_EDITOR_PSEUDO_OPERATION,
            "priority": _URGENT_PRIORITY,
            "editor_pseudo": {
                "dataset_key": dataset.key,
                "annotation_name": scene.annotation_name,
                "image_relative": image_relative,
                "images_root": str(root),
                "raster_revision": raster_revision,
                "image_relative_path": image_path.relative_to(root).as_posix(),
                "inference_config_hash": inference_config_hash,
                "algorithm_version": _EDITOR_PSEUDO_ALGORITHM_VERSION,
                "class_id": class_key,
                "class_name": dataset.class_name or dataset.name,
                "training_result_id": str(primary.id),
                "model_name": primary.model_name,
                "task": primary.task,
                "object_types": list(primary.class_schema or []),
                "imagery_type": dataset.imagery_type.value
                if dataset.imagery_type is not None
                else None,
                "input_channels": selected.input_channels,
                "mlflow_run_id": primary.mlflow_run_id,
                "checkpoint_uri": selected.checkpoint.artifact_uri,
                "checkpoint_artifact_path": selected.checkpoint.artifact_path,
                "checkpoint_threshold": selected.checkpoint.threshold,
                "checkpoint_f1_score": selected.checkpoint.f1_score,
                "checkpoint_epoch": selected.checkpoint.epoch,
                "external_model": external_model_payload(selected.external_model),
                "inference_template_id": (
                    str(selected.inference_template_id)
                    if selected.inference_template_id is not None
                    else None
                ),
                "inference_template_config": selected.inference_template_config,
                "result_file_id": None,
                "error": None,
            },
        },
    )
    try:
        with session.begin_nested():
            session.add(row)
            session.flush()
    except IntegrityError:
        concurrent = session.scalar(select(JobRow).where(JobRow.dedup_key == dedup_key))
        if concurrent is None:
            raise
        return _editor_pseudo_job_info(
            session,
            concurrent,
            image_path=image_path,
            training_result=primary,
        )
    return DatasetEditorPseudoMarkupInfo(
        status="queued",
        source="scene",
        training_result_id=primary.id,
        model_name=primary.model_name,
        job_id=row.id,
        message="Срочный инференс по снимку поставлен в начало очереди.",
    )


def editor_pseudo_job_info(
    session: Session,
    config: TrainingUIAPIConfig,
    job_id: uuid.UUID,
) -> DatasetEditorPseudoMarkupInfo:
    """Лёгкий polling задания без Git и повторного разрешения каталога снимков."""

    row = session.get(JobRow, job_id)
    if row is None or (row.config or {}).get("operation") != DATASET_EDITOR_PSEUDO_OPERATION:
        raise TrainingUIAPIError("Задание псевдоразметки снимка не найдено")
    state = _editor_pseudo_state(row)
    try:
        training_result_id = uuid.UUID(str(state.get("training_result_id")))
    except (TypeError, ValueError) as exc:
        raise TrainingUIAPIError("Параметры задания псевдоразметки повреждены") from exc
    training_result = session.get(TrainingResultRow, training_result_id)
    if training_result is None:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            job_id=row.id,
            message="Основная сеть задания больше недоступна.",
        )
    current_primary = _editor_effective_training_result(
        session,
        str(row.dataset_key or ""),
        str(state.get("class_id") or ""),
    )
    if current_primary is None or current_primary.id != training_result.id:
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=training_result.id,
            model_name=training_result.model_name,
            job_id=row.id,
            message="Основная сеть класса была изменена.",
        )
    root = Path(str(state.get("images_root") or config.images_root)).resolve()
    relative = PurePosixPath(str(state.get("image_relative_path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError("Путь снимка в задании повреждён")
    image_path = root.joinpath(*relative.parts).resolve()
    _ensure_within(image_path, root, "Снимок задания выходит за пределы каталога")
    if not image_path.is_file() or _raster_revision(image_path) != state.get("raster_revision"):
        return DatasetEditorPseudoMarkupInfo(
            status="unavailable",
            training_result_id=training_result.id,
            model_name=training_result.model_name,
            job_id=row.id,
            message="Снимок был изменён после запуска инференса.",
        )
    return _editor_pseudo_job_info(
        session,
        row,
        image_path=image_path,
        training_result=training_result,
    )


def browse_editor_rasters(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    folder: str,
) -> DatasetEditorRasterBrowserResponse:
    dataset = _managed_editor_dataset(session, config, dataset_key)
    images_root = _dataset_images_root(dataset)
    relative_folder, folder_path = _safe_relative_directory(images_root, folder)
    folders = [
        DatasetEditorRasterFolderInfo(
            name=item.name,
            path=(relative_folder / item.name).as_posix(),
        )
        for item in sorted(folder_path.iterdir(), key=lambda path: path.name.casefold())
        if item.is_dir() and not item.name.startswith(".")
    ]
    rasters = [
        DatasetEditorRasterInfo(
            name=item.name,
            path=(relative_folder / item.name).as_posix(),
            annotation_name=per_image_annotation_name(item),
            size_bytes=item.stat().st_size,
        )
        for item in sorted(folder_path.iterdir(), key=lambda path: path.name.casefold())
        if item.is_file() and item.suffix.casefold() in RASTER_SUFFIXES
    ]
    parent = relative_folder.parent.as_posix() if relative_folder.parts else None
    if parent == ".":
        parent = ""
    return DatasetEditorRasterBrowserResponse(
        folder="" if relative_folder.as_posix() == "." else relative_folder.as_posix(),
        parent=parent,
        folders=folders,
        rasters=rasters,
    )


def resolve_editor_raster(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    image_path: str,
) -> Path:
    dataset = _managed_editor_dataset(session, config, dataset_key)
    return _safe_raster_path(_dataset_images_root(dataset), image_path)


def add_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    image_paths: list[str],
    folder_path: str | None,
    username: str,
) -> DatasetEditorMutationResult:
    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(
            session,
            config,
            dataset_key,
            allow_missing=True,
        )
        images_root = _dataset_images_root(dataset)
        if folder_path is not None:
            _relative, selected_folder = _safe_relative_directory(images_root, folder_path)
            selected_paths = [
                path
                for path in sorted(selected_folder.iterdir(), key=lambda item: item.name.casefold())
                if path.is_file() and path.suffix.casefold() in RASTER_SUFFIXES
            ]
        else:
            selected_paths = [
                _safe_raster_path(images_root, image_path) for image_path in image_paths
            ]
        if not selected_paths:
            raise TrainingUIAPIError("В выбранном источнике нет TIFF")
        names: dict[str, tuple[str, Path]] = {}
        for image_path in selected_paths:
            annotation_name = per_image_annotation_name(image_path)
            key = annotation_name.casefold()
            if key in names:
                raise TrainingUIAPIError(
                    f"Несколько TIFF дают одинаковое имя GeoJSON: {annotation_name}"
                )
            names[key] = (annotation_name, image_path)
        existing = {path.name.casefold() for path in _direct_annotation_files(source_dir)}
        collisions = sorted(name for key, (name, _path) in names.items() if key in existing)
        if collisions:
            if folder_path is None:
                raise DatasetEditorConflict(
                    "Снимки уже добавлены в датасет: " + ", ".join(collisions)
                )
            names = {key: value for key, value in names.items() if key not in existing}
            if not names:
                raise TrainingUIAPIError("Все TIFF из выбранной папки уже добавлены в датасет")
        if dataset.managed:
            return _add_managed_editor_scenes(
                session,
                config,
                dataset,
                names,
                username=username,
            )
        source_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        relative_files: list[PurePosixPath] = []
        try:
            for annotation_name, image_path in names.values():
                payload = _empty_annotation_payload(image_path, dataset)
                target = source_dir / annotation_name
                _write_geojson_atomic(target, payload)
                footprint_target = source_dir / footprint_name_for_annotation(annotation_name)
                _write_geojson_atomic(
                    footprint_target,
                    _footprint_geojson_payload(image_path),
                )
                created.extend((target, footprint_target))
                relative_files.extend(
                    (
                        _repo_relative(config, target),
                        _repo_relative(config, footprint_target),
                    )
                )
            _git(config, "add", "--", *(path.as_posix() for path in relative_files))
            commit = _commit(
                config,
                f"Добавить снимки в датасет {dataset.dataset_name or dataset.name}",
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions={path: None for path in relative_files},
            )
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            if relative_files:
                _git_optional(
                    config,
                    "restore",
                    "--staged",
                    "--worktree",
                    "--",
                    *(path.as_posix() for path in relative_files),
                )
            raise
        scenes = _scene_infos(config, dataset, source_dir)
        added_names = {name for name, _path in names.values()}
        return DatasetEditorMutationResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
            scenes=[scene for scene in scenes if scene.annotation_name in added_names],
        )


def save_editor_scene(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
    *,
    revision: str,
    geojson: dict[str, Any],
    username: str,
) -> DatasetEditorMutationResult:
    return publish_editor_scenes(
        session,
        config,
        dataset_key,
        scenes=[(annotation_name, revision, geojson)],
        username=username,
    )


def publish_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    scenes: list[tuple[str, str, dict[str, Any]]],
    deletions: list[tuple[str, str]] | None = None,
    username: str,
) -> DatasetEditorMutationResult:
    deletions = deletions or []
    if not scenes and not deletions:
        raise TrainingUIAPIError("Для публикации нужен хотя бы один снимок")
    normalized_names = [name.casefold() for name, _revision, _geojson in scenes]
    normalized_names.extend(name.casefold() for name, _revision in deletions)
    if len(normalized_names) != len(set(normalized_names)):
        raise TrainingUIAPIError("Список публикации содержит повторяющиеся снимки")

    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        if dataset.managed:
            return _publish_managed_editor_scenes(
                session,
                config,
                dataset,
                source_dir,
                scenes=scenes,
                deletions=deletions,
                username=username,
            )
        resolved: list[tuple[str, str, dict[str, Any], Path, PurePosixPath]] = []
        resolved_deletions: list[tuple[str, str, Path, PurePosixPath]] = []
        footprint_deletions: list[tuple[PurePosixPath, str | None]] = []
        conflicts: list[str] = []
        for annotation_name, revision, geojson in scenes:
            scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
            annotation_path = _annotation_path(source_dir, scene.annotation_name)
            relative_path = _repo_relative(config, annotation_path)
            current_revision = _blob_revision(config, "HEAD", relative_path)
            if current_revision != revision:
                conflicts.append(scene.annotation_name)
            resolved.append(
                (scene.annotation_name, revision, geojson, annotation_path, relative_path)
            )
        for annotation_name, revision in deletions:
            scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
            annotation_path = _annotation_path(source_dir, scene.annotation_name)
            relative_path = _repo_relative(config, annotation_path)
            current_revision = _blob_revision(config, "HEAD", relative_path)
            if current_revision != revision:
                conflicts.append(scene.annotation_name)
            resolved_deletions.append(
                (scene.annotation_name, revision, annotation_path, relative_path)
            )
            footprint_path = source_dir / footprint_name_for_annotation(scene.annotation_name)
            if footprint_path.is_file():
                footprint_relative = _repo_relative(config, footprint_path)
                footprint_deletions.append(
                    (
                        footprint_relative,
                        _blob_revision(config, "HEAD", footprint_relative),
                    )
                )
        if conflicts:
            raise DatasetEditorConflict(
                "Разметка уже изменена другим пользователем: " + ", ".join(conflicts)
            )

        prepared: list[tuple[str, str, dict[str, Any], Path, PurePosixPath]] = []
        footprint_updates: list[tuple[Path, PurePosixPath, str | None, dict[str, Any]]] = []
        for annotation_name, revision, geojson, annotation_path, relative_path in resolved:
            image_path = _matched_image_path(dataset, source_dir, annotation_name)
            previous_payload = _reproject_editor_geojson(
                _read_geojson(annotation_path),
                image_path,
            )
            normalized_geojson = _normalize_editor_geojson(
                geojson,
                previous_payload,
                dataset,
                image_path=image_path,
            )
            _validate_editor_geojson(normalized_geojson, image_path, dataset)
            _validate_preserved_properties(previous_payload, normalized_geojson)
            prepared.append(
                (
                    annotation_name,
                    revision,
                    normalized_geojson,
                    annotation_path,
                    relative_path,
                )
            )
            footprint_path = source_dir / footprint_name_for_annotation(annotation_name)
            footprint_relative = _repo_relative(config, footprint_path)
            footprint_updates.append(
                (
                    footprint_path,
                    footprint_relative,
                    _blob_revision(config, "HEAD", footprint_relative),
                    _footprint_geojson_payload(image_path),
                )
            )

        relative_paths = [item[4] for item in prepared]
        footprint_update_paths = [item[1] for item in footprint_updates]
        deletion_paths = [item[3] for item in resolved_deletions]
        footprint_deletion_paths = [item[0] for item in footprint_deletions]
        all_relative_paths = [
            *relative_paths,
            *footprint_update_paths,
            *deletion_paths,
            *footprint_deletion_paths,
        ]
        try:
            for _name, _revision, geojson, annotation_path, _relative_path in prepared:
                _write_geojson_atomic(annotation_path, geojson)
            for footprint_path, _relative, _revision, payload in footprint_updates:
                _write_geojson_atomic(footprint_path, payload)
            if relative_paths or footprint_update_paths:
                _git(
                    config,
                    "add",
                    "--",
                    *(path.as_posix() for path in [*relative_paths, *footprint_update_paths]),
                )
            if deletion_paths:
                _git(
                    config,
                    "rm",
                    "--",
                    *(path.as_posix() for path in [*deletion_paths, *footprint_deletion_paths]),
                )
            change_count = len(prepared) + len(resolved_deletions)
            if change_count == 1 and prepared:
                subject = f"Обновить разметку {prepared[0][0]}"
            elif change_count == 1:
                subject = f"Удалить снимок {resolved_deletions[0][0]}"
            else:
                subject = (
                    f"Обновить разметку датасета {dataset.dataset_name or dataset.name} "
                    f"({change_count} снимка)"
                )
            commit = _commit(config, subject, username)
            commit = _push_with_retry(
                config,
                expected_revisions={
                    **{item[4]: item[1] for item in prepared},
                    **{item[1]: item[2] for item in footprint_updates},
                    **{item[3]: item[1] for item in resolved_deletions},
                    **dict(footprint_deletions),
                },
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                *(path.as_posix() for path in all_relative_paths),
            )
            for footprint_path, _relative, revision, _payload in footprint_updates:
                if revision is None:
                    footprint_path.unlink(missing_ok=True)
            raise
        updated_scenes = {
            item.annotation_name: item for item in _scene_infos(config, dataset, source_dir)
        }
        if prepared:
            session.execute(
                delete(DatasetEditorDraftRow).where(
                    DatasetEditorDraftRow.dataset_key == dataset.key,
                    DatasetEditorDraftRow.username == username,
                    DatasetEditorDraftRow.annotation_name.in_([item[0] for item in prepared]),
                )
            )
        if resolved_deletions:
            session.execute(
                delete(DatasetEditorDraftRow).where(
                    DatasetEditorDraftRow.dataset_key == dataset.key,
                    DatasetEditorDraftRow.annotation_name.in_(
                        [item[0] for item in resolved_deletions]
                    ),
                )
            )
        session.flush()
        return DatasetEditorMutationResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
            scenes=[updated_scenes[item[0]] for item in prepared],
        )


def delete_editor_scene(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
    *,
    revision: str,
    username: str,
) -> DatasetEditorDraftInfo:
    """Пометить снимок на удаление в пользовательском черновике."""

    with _editor_lock(config):
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
        if scene.revision != revision:
            raise DatasetEditorConflict("Разметка уже изменена другим пользователем")
        geojson = _read_geojson(_annotation_path(source_dir, scene.annotation_name))
    return save_editor_draft(
        session,
        config,
        dataset_key,
        annotation_name,
        base_revision=revision,
        geojson=geojson,
        deleted=True,
        username=username,
    )


def copy_editor_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    name: str,
    username: str,
) -> DatasetEditorCopyResult:
    """Создать новый датасет из опубликованного состояния существующего."""

    copy_name = _dataset_copy_name(name)
    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset = _managed_editor_dataset(session, config, dataset_key)
        row = session.scalar(
            select(DatasetRow).where(
                DatasetRow.key == dataset_key,
                DatasetRow.deleted_at.is_(None),
            )
        )
        if row is None:
            raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")
        _ensure_dataset_copy_name_available(session, row.class_id, copy_name)

        if row.source_type == SOURCE_MANAGED:
            copied = _copy_managed_dataset_row(session, row, copy_name)
            commit = _git(config, "rev-parse", "HEAD").stdout.strip()
            return DatasetEditorCopyResult(
                commit=commit,
                publication_status=_publication_status(config, commit),
                dataset_key=copied.key,
                dataset_name=copied.name,
                source_path=copied.source_path,
                managed=True,
            )

        source_dir = _editor_source_dir(config, dataset)
        source_relative = _repo_relative(config, source_dir)
        if len(source_relative.parts) < 2:
            raise TrainingUIAPIError(
                "Копировать можно только отдельную папку датасета внутри папки класса."
            )
        target_relative = source_relative.parent / copy_name
        target_dir = config.mlmarkup_editor_root.joinpath(*target_relative.parts)
        _ensure_within(
            target_dir.resolve(),
            config.mlmarkup_editor_root.resolve(),
            "Папка копии выходит за пределы editor-клона",
        )
        _ensure_dataset_copy_source_available(session, target_relative, target_dir)

        source_tree_revision = _tree_object_revision(config, "HEAD", source_relative)
        if source_tree_revision is None or not source_dir.is_dir():
            raise TrainingUIAPIError(
                "Исходная папка датасета отсутствует или не зафиксирована в Git editor-клона."
            )

        try:
            shutil.copytree(source_dir, target_dir, copy_function=shutil.copy2)
            _git(config, "add", "--", target_relative.as_posix())
            commit = _commit(
                config,
                f"Создать копию датасета {dataset.name} как {copy_name}",
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions={},
                expected_tree_revisions={
                    source_relative: source_tree_revision,
                    target_relative: None,
                },
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                target_relative.as_posix(),
            )
            if target_dir.is_dir():
                shutil.rmtree(target_dir)
            raise

        copied = DatasetRow(
            key=str(uuid.uuid4()),
            class_id=row.class_id,
            name=copy_name,
            source_type=row.source_type,
            source_path=target_relative.as_posix(),
            model_name_stem=row.model_name_stem,
            config_revision=1,
            legacy_version=False,
        )
        session.add(copied)
        session.flush()
        return DatasetEditorCopyResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
            dataset_key=copied.key,
            dataset_name=copied.name,
            source_path=copied.source_path,
            managed=False,
        )


def delete_editor_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    username: str,
) -> DatasetEditorMutationResult:
    active_job = session.scalar(
        select(JobRow.id)
        .where(
            JobRow.dataset_key == dataset_key,
            JobRow.status.in_(("queued", "running", "paused")),
        )
        .limit(1)
    )
    if active_job is not None:
        raise TrainingUIAPIError(
            "Перед удалением датасета завершите или удалите его активные задания."
        )

    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset = _managed_editor_dataset(session, config, dataset_key)
        row = session.scalar(
            select(DatasetRow).where(
                DatasetRow.key == dataset_key,
                DatasetRow.deleted_at.is_(None),
            )
        )
        if row is None:
            raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")
        if row.source_type == SOURCE_MANAGED:
            class_row = session.get(DatasetClassRow, row.class_id)
            if class_row is not None and class_row.primary_dataset_id == row.id:
                class_row.primary_dataset_id = None
            row.deleted_at = datetime.now(timezone.utc)
            row.config_revision += 1
            session.execute(
                delete(DatasetEditorDraftRow).where(
                    DatasetEditorDraftRow.dataset_key == dataset_key
                )
            )
            invalidate_managed_cache(config, dataset_key, remove_ready=True)
            session.flush()
            commit = _git(config, "rev-parse", "HEAD").stdout.strip()
            return DatasetEditorMutationResult(
                commit=commit,
                publication_status=_publication_status(config, commit),
            )
        dependent = session.execute(
            select(DatasetRow.name, DatasetClassRow.name)
            .join(
                ManagedDatasetSourceRow,
                ManagedDatasetSourceRow.managed_dataset_id == DatasetRow.id,
            )
            .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
            .where(
                ManagedDatasetSourceRow.source_dataset_id == row.id,
                DatasetRow.deleted_at.is_(None),
            )
            .limit(1)
        ).first()
        if dependent is not None:
            raise TrainingUIAPIError(
                "Нельзя удалить исходный датасет, пока он входит в управляемый: "
                f"{dependent[1]} / {dependent[0]}."
            )
        source_dir = _editor_source_dir(config, dataset)
        source_relative = _repo_relative(config, source_dir)
        if len(source_relative.parts) < 2:
            raise TrainingUIAPIError(
                "Удалять можно только отдельную папку датасета внутри папки класса."
            )
        nested_sources = [
            candidate.source_path
            for candidate in session.scalars(
                select(DatasetRow).where(
                    DatasetRow.id != row.id,
                    DatasetRow.deleted_at.is_(None),
                )
            )
            if source_relative in PurePosixPath(candidate.source_path).parents
        ]
        if nested_sources:
            raise TrainingUIAPIError(
                "Папка содержит другие управляемые датасеты: "
                + ", ".join(sorted(nested_sources, key=str.casefold))
            )
        tree_revision = _tree_object_revision(config, "HEAD", source_relative)
        if tree_revision is None or not source_dir.is_dir():
            raise TrainingUIAPIError(
                "Папка датасета отсутствует или не зафиксирована в Git editor-клона."
            )

        _git(config, "rm", "-r", "--", source_relative.as_posix())
        try:
            commit = _commit(
                config,
                f"Удалить датасет {dataset.name}",
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions={},
                expected_tree_revisions={source_relative: tree_revision},
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                source_relative.as_posix(),
            )
            raise

        class_row = session.get(DatasetClassRow, row.class_id)
        if class_row is not None and class_row.primary_dataset_id == row.id:
            class_row.primary_dataset_id = None
        row.deleted_at = datetime.now(timezone.utc)
        row.config_revision += 1
        row.legacy_version = False
        session.execute(
            delete(DatasetEditorDraftRow).where(DatasetEditorDraftRow.dataset_key == dataset_key)
        )
        session.flush()
        return DatasetEditorMutationResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
        )


def _dataset_copy_name(value: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise TrainingUIAPIError("Название копии не может быть пустым")
    if normalized in {".", ".."} or any(char in normalized for char in '<>:"/\\|?*'):
        raise TrainingUIAPIError("Название копии содержит символы, недопустимые в имени папки.")
    if normalized.endswith((" ", ".")):
        raise TrainingUIAPIError("Название копии не должно оканчиваться пробелом или точкой.")
    reserved = {
        "aux",
        "clock$",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    if normalized.split(".", maxsplit=1)[0].casefold() in reserved:
        raise TrainingUIAPIError("Название копии зарезервировано операционной системой.")
    return normalized


def _ensure_dataset_copy_name_available(
    session: Session,
    class_id: uuid.UUID,
    name: str,
) -> None:
    names = session.scalars(select(DatasetRow.name).where(DatasetRow.class_id == class_id)).all()
    if any(existing.casefold() == name.casefold() for existing in names):
        raise TrainingUIAPIError(
            f"Датасет с названием «{name}» уже существует в классе или сохранён в истории."
        )


def _ensure_dataset_copy_source_available(
    session: Session,
    target_relative: PurePosixPath,
    target_dir: Path,
) -> None:
    target_key = target_relative.as_posix().casefold()
    source_paths = session.scalars(select(DatasetRow.source_path)).all()
    if any(value.casefold() == target_key for value in source_paths):
        raise TrainingUIAPIError(
            f"Путь копии уже использовался другим датасетом: {target_relative.as_posix()}"
        )
    if target_dir.exists():
        raise TrainingUIAPIError(
            f"Папка копии уже существует в MLMarkup: {target_relative.as_posix()}"
        )


def _copy_managed_dataset_row(
    session: Session,
    source: DatasetRow,
    name: str,
) -> DatasetRow:
    copied = DatasetRow(
        key=str(uuid.uuid4()),
        class_id=source.class_id,
        name=name,
        source_type=SOURCE_MANAGED,
        source_path=f"managed/{uuid.uuid4()}",
        model_name_stem=source.model_name_stem,
        config_revision=1,
        legacy_version=False,
    )
    session.add(copied)
    session.flush()

    relations = session.scalars(
        select(ManagedDatasetSourceRow).where(
            ManagedDatasetSourceRow.managed_dataset_id == source.id
        )
    ).all()
    session.add_all(
        [
            ManagedDatasetSourceRow(
                managed_dataset_id=copied.id,
                source_dataset_id=relation.source_dataset_id,
                priority=relation.priority,
                object_type_id=relation.object_type_id,
                object_type_slug=relation.object_type_slug,
                object_type_name=relation.object_type_name,
                color=relation.color,
            )
            for relation in relations
        ]
    )
    explicit_scenes = session.scalars(
        select(ManagedDatasetSceneRow).where(ManagedDatasetSceneRow.managed_dataset_id == source.id)
    ).all()
    session.add_all(
        [
            ManagedDatasetSceneRow(
                managed_dataset_id=copied.id,
                annotation_name=scene.annotation_name,
                image_relative_path=scene.image_relative_path,
            )
            for scene in explicit_scenes
        ]
    )
    session.flush()
    return copied


def editor_publication_info(
    config: TrainingUIAPIConfig,
    commit: str,
) -> DatasetEditorPublicationInfo:
    if _SHA_PATTERN.fullmatch(commit) is None:
        raise TrainingUIAPIError("Некорректный SHA коммита")
    with _editor_lock(config, restore_ownership=True):
        _fetch_editor_clone(config)
        live_commit = _live_commit(config)
        status = "publishing"
        if live_commit is not None:
            result = _git_optional(
                config,
                "merge-base",
                "--is-ancestor",
                commit,
                live_commit,
            )
            if result.returncode == 0:
                status = "published"
        return DatasetEditorPublicationInfo(
            commit=commit,
            live_commit=live_commit,
            status=status,
        )


def preview_editor_dataset_rebuild(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetEditorRebuildPreview:
    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        manifest = load_dataset_manifest(source_dir)
        if manifest is None or not manifest.combined:
            raise TrainingUIAPIError("Пересборка доступна только датасету с combined-манифестом.")
        build = build_combined_dataset(
            manifest=manifest,
            repo_root=config.mlmarkup_editor_root,
            images_root=_dataset_images_root(dataset),
            code_revision=_project_git_head(config.project_root),
        )
        current_payloads = _target_geojson_payloads(source_dir)
        local_changes = _local_rebuild_changes(manifest, current_payloads)
        source_changes = _source_rebuild_changes(
            manifest,
            config.mlmarkup_editor_root,
        )
        conflicts = _rebuild_conflicts(
            manifest,
            current_payloads,
            build.files,
            local_changes,
        )
        token = uuid.uuid4().hex
        state = {
            "dataset_key": dataset_key,
            "target_tree": tree_revision(folder_file_hashes(source_dir)),
            "target_git_tree": _tree_object_revision(
                config,
                "HEAD",
                _repo_relative(config, source_dir),
            ),
            "source_trees": {
                source.path: tree_revision(
                    folder_file_hashes(
                        config.mlmarkup_editor_root.joinpath(*PurePosixPath(source.path).parts)
                    )
                )
                for source in manifest.sources
            },
            "source_git_trees": {
                source.path: _tree_object_revision(
                    config,
                    "HEAD",
                    PurePosixPath(source.path),
                )
                for source in manifest.sources
            },
            "files": build.files,
            "manifest": build.manifest.model_dump(mode="json"),
            "class_counts": build.class_counts,
            "hard_negative_count": build.hard_negative_count,
            "warnings": list(build.warnings),
            "local_changes": [item.model_dump(mode="json") for item in local_changes],
            "conflicts": [item.model_dump(mode="json") for item in conflicts],
        }
        preview_path = _rebuild_preview_path(config, token)
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(preview_path, state)
        return DatasetEditorRebuildPreview(
            preview_token=token,
            dataset_key=dataset_key,
            source_status=("stale" if source_changes else "current"),
            source_changes=source_changes,
            local_changes=local_changes,
            conflicts=conflicts,
            replacement_scene_count=len(build.files),
            replacement_class_counts=build.class_counts,
            replacement_hard_negative_count=build.hard_negative_count,
            warnings=list(build.warnings),
        )


def rebuild_editor_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    preview_token: str,
    mode: str,
    username: str,
) -> DatasetEditorRebuildResult:
    if mode not in {"merge", "replace"}:
        raise TrainingUIAPIError("mode должен быть merge или replace")
    if not re.fullmatch(r"[0-9a-f]{32}", preview_token):
        raise DatasetEditorConflict("Некорректный или устаревший preview_token")
    with _editor_lock(config, restore_ownership=True):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        preview_path = _rebuild_preview_path(config, preview_token)
        try:
            state = json.loads(preview_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetEditorConflict("Preview пересборки не найден или устарел") from exc
        if state.get("dataset_key") != dataset_key:
            raise DatasetEditorConflict("preview_token создан для другого датасета")
        if tree_revision(folder_file_hashes(source_dir)) != state.get("target_tree"):
            raise DatasetEditorConflict("Target-датасет изменился после preview")
        manifest = load_dataset_manifest(source_dir)
        if manifest is None:
            raise DatasetEditorConflict("Манифест target-датасета изменился после preview")
        for source in manifest.sources:
            source_path = config.mlmarkup_editor_root.joinpath(*PurePosixPath(source.path).parts)
            current_tree = tree_revision(folder_file_hashes(source_path))
            if current_tree != (state.get("source_trees") or {}).get(source.path):
                raise DatasetEditorConflict(
                    f"Исходная папка {source.path} изменилась после preview"
                )
        candidate_files = {
            str(name): payload for name, payload in (state.get("files") or {}).items()
        }
        candidate_manifest = state.get("manifest")
        if not isinstance(candidate_manifest, dict):
            raise DatasetEditorConflict("Preview пересборки повреждён")
        output_files = (
            candidate_files
            if mode == "replace"
            else _merge_rebuild_payloads(
                manifest,
                _target_geojson_payloads(source_dir),
                candidate_files,
            )
        )
        tracked_paths = _rebuild_tracked_paths(
            config,
            source_dir,
            manifest,
            candidate_files,
        )
        expected_revisions = {path: _blob_revision(config, "HEAD", path) for path in tracked_paths}
        _replace_dataset_files_atomically(
            source_dir,
            output_files,
            candidate_manifest,
            dataset,
        )
        target_relative = _repo_relative(config, source_dir)
        try:
            _git(config, "add", "-A", "--", target_relative.as_posix())
            commit = _commit(
                config,
                (f"Пересобрать датасет {dataset.dataset_name or dataset.name} в режиме {mode}"),
                username,
            )
            commit = _push_with_retry(
                config,
                expected_revisions=expected_revisions,
                expected_tree_revisions={
                    _repo_relative(config, source_dir): state.get("target_git_tree"),
                    **{
                        PurePosixPath(source.path): tree
                        for source, tree in (
                            (
                                source,
                                (state.get("source_git_trees") or {}).get(source.path),
                            )
                            for source in manifest.sources
                        )
                        if isinstance(tree, str)
                    },
                },
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                target_relative.as_posix(),
            )
            raise
        preview_path.unlink(missing_ok=True)
        scenes = _scene_infos(config, dataset, source_dir)
        return DatasetEditorRebuildResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
            scenes=scenes,
            mode=mode,
            conflicts=[
                DatasetEditorRebuildChange.model_validate(item)
                for item in state.get("conflicts") or []
            ],
            warnings=[str(item) for item in state.get("warnings") or []],
        )


def _target_geojson_payloads(source_dir: Path) -> dict[str, dict[str, Any]]:
    return {path.name: _read_geojson(path) for path in _direct_annotation_files(source_dir)}


def _source_rebuild_changes(
    manifest,
    repo_root: Path,
) -> list[str]:
    changes: list[str] = []
    for source in manifest.sources:
        folder = repo_root.joinpath(*PurePosixPath(source.path).parts)
        current = folder_file_hashes(folder)
        previous = dict(source.file_hashes or {})
        if not previous:
            if tree_revision(current) != source.tree_revision:
                changes.append(f"Изменена исходная папка: {source.path}")
            continue
        for name in sorted(set(previous) | set(current)):
            if name not in previous:
                changes.append(f"Добавлен исходный файл: {source.path}/{name}")
            elif name not in current:
                changes.append(f"Удалён исходный файл: {source.path}/{name}")
            elif previous[name] != current[name]:
                changes.append(f"Изменён исходный файл: {source.path}/{name}")
    return changes


def _local_rebuild_changes(
    manifest,
    current_payloads: dict[str, dict[str, Any]],
) -> list[DatasetEditorRebuildChange]:
    changes: list[DatasetEditorRebuildChange] = []
    baseline = dict(manifest.baseline_hashes or {})
    for annotation_name in sorted(set(baseline) | set(current_payloads)):
        if annotation_name not in current_payloads:
            changes.append(
                DatasetEditorRebuildChange(
                    kind="deleted",
                    annotation_name=annotation_name,
                    detail="Локально удалён весь per-image файл.",
                )
            )
            continue
        current = _features_by_origin(current_payloads[annotation_name])
        base = baseline.get(annotation_name, {})
        if annotation_name not in baseline:
            changes.append(
                DatasetEditorRebuildChange(
                    kind="added",
                    annotation_name=annotation_name,
                    detail="Локально добавлен per-image файл.",
                )
            )
            continue
        for origin_key in sorted(set(base) | set(current)):
            if origin_key not in base:
                changes.append(
                    DatasetEditorRebuildChange(
                        kind="added",
                        annotation_name=annotation_name,
                        origin_key=origin_key,
                    )
                )
            elif origin_key not in current:
                changes.append(
                    DatasetEditorRebuildChange(
                        kind="deleted",
                        annotation_name=annotation_name,
                        origin_key=origin_key,
                    )
                )
            elif feature_hash(current[origin_key]) != base[origin_key]:
                changes.append(
                    DatasetEditorRebuildChange(
                        kind="edited",
                        annotation_name=annotation_name,
                        origin_key=origin_key,
                    )
                )
    return changes


def _rebuild_conflicts(
    manifest,
    current_payloads: dict[str, dict[str, Any]],
    candidate_payloads: dict[str, dict[str, Any]],
    local_changes: list[DatasetEditorRebuildChange],
) -> list[DatasetEditorRebuildChange]:
    del current_payloads
    baseline = dict(manifest.baseline_hashes or {})
    candidate = {
        name: {
            origin: feature_hash(feature)
            for origin, feature in _features_by_origin(payload).items()
        }
        for name, payload in candidate_payloads.items()
    }
    conflicts: list[DatasetEditorRebuildChange] = []
    for change in local_changes:
        base_file = baseline.get(change.annotation_name, {})
        candidate_file = candidate.get(change.annotation_name, {})
        if change.origin_key is None:
            source_changed = candidate_file != base_file
        else:
            source_changed = candidate_file.get(change.origin_key) != base_file.get(
                change.origin_key
            )
        if source_changed:
            conflicts.append(
                change.model_copy(
                    update={
                        "detail": (
                            "Источник и ручная версия изменили один объект; "
                            "в merge побеждает ручная версия."
                        )
                    }
                )
            )
    return conflicts


def _merge_rebuild_payloads(
    manifest,
    current_payloads: dict[str, dict[str, Any]],
    candidate_payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    baseline = dict(manifest.baseline_hashes or {})
    output: dict[str, dict[str, Any]] = {}
    for annotation_name in sorted(set(current_payloads) | set(candidate_payloads) | set(baseline)):
        current_payload = current_payloads.get(annotation_name)
        candidate_payload = candidate_payloads.get(annotation_name)
        base = baseline.get(annotation_name)
        if current_payload is None and base is not None:
            continue
        if current_payload is not None and base is None:
            output[annotation_name] = current_payload
            continue
        if current_payload is None:
            if candidate_payload is not None:
                output[annotation_name] = candidate_payload
            continue
        current_features = _features_by_origin(current_payload)
        candidate_features = (
            _features_by_origin(candidate_payload) if candidate_payload is not None else {}
        )
        base_hashes = base or {}
        merged_features = dict(candidate_features)
        for origin_key in set(base_hashes) | set(current_features):
            if origin_key not in current_features:
                merged_features.pop(origin_key, None)
            elif origin_key not in base_hashes:
                merged_features[origin_key] = current_features[origin_key]
            elif feature_hash(current_features[origin_key]) != base_hashes[origin_key]:
                merged_features[origin_key] = current_features[origin_key]
        template = dict(candidate_payload or current_payload)
        template["_mlsystem2_schema_version"] = manifest.schema_version
        template["_mlsystem2_task"] = manifest.task
        template["_mlsystem2_classes"] = [item.model_dump(mode="json") for item in manifest.classes]
        template["features"] = sorted(
            merged_features.values(),
            key=lambda feature: str(feature.get("id") or ""),
        )
        if template["features"]:
            output[annotation_name] = template
    return output


def _features_by_origin(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for feature in payload.get("features") or []:
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        origin_key = properties.get("_mlsystem2_origin_key")
        if not isinstance(origin_key, str) or not origin_key:
            feature_id = feature.get("id")
            origin_key = f"manual:{feature_id}" if feature_id is not None else feature_hash(feature)
        output[origin_key] = feature
    return output


def _replace_dataset_files_atomically(
    source_dir: Path,
    payloads: dict[str, dict[str, Any]],
    manifest_payload: dict[str, Any],
    dataset: DatasetInfo,
) -> None:
    parent = source_dir.parent
    temporary = parent / f".{source_dir.name}.rebuild-{uuid.uuid4().hex}"
    backup = parent / f".{source_dir.name}.backup-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False)
    try:
        if source_dir.is_dir():
            for item in source_dir.iterdir():
                if item.name == ".mlsystem2-dataset.json" or item.suffix.casefold() == ".geojson":
                    continue
                destination = temporary / item.name
                if item.is_dir():
                    shutil.copytree(item, destination)
                else:
                    shutil.copy2(item, destination)
        for name, payload in payloads.items():
            annotation_name = _safe_annotation_name(name)
            image_path = _image_path_for_annotation(dataset, annotation_name)
            _write_geojson_atomic(temporary / annotation_name, payload)
            _write_geojson_atomic(
                temporary / footprint_name_for_annotation(annotation_name),
                _footprint_geojson_payload(image_path),
            )
        _write_json_atomic(temporary / ".mlsystem2-dataset.json", manifest_payload)
        if source_dir.exists():
            os.replace(source_dir, backup)
        try:
            os.replace(temporary, source_dir)
        except Exception:
            if backup.exists() and not source_dir.exists():
                os.replace(backup, source_dir)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _rebuild_tracked_paths(
    config: TrainingUIAPIConfig,
    source_dir: Path,
    manifest,
    candidate_payloads: dict[str, dict[str, Any]],
) -> set[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    for folder in [
        source_dir,
        *[
            config.mlmarkup_editor_root.joinpath(*PurePosixPath(item.path).parts)
            for item in manifest.sources
        ],
    ]:
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file():
                    paths.add(_repo_relative(config, path))
    target_relative = _repo_relative(config, source_dir)
    paths.add(target_relative / ".mlsystem2-dataset.json")
    for name in candidate_payloads:
        annotation_name = _safe_annotation_name(name)
        paths.add(target_relative / annotation_name)
        paths.add(target_relative / footprint_name_for_annotation(annotation_name))
    return paths


def _rebuild_preview_path(config: TrainingUIAPIConfig, token: str) -> Path:
    return config.mlmarkup_editor_root.parent / ".mlsystem2-rebuild-previews" / f"{token}.json"


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _project_git_head(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _managed_editor_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetInfo:
    dataset = find_managed_dataset(
        session,
        config,
        dataset_key,
        materialize_managed=False,
    )
    if dataset is None or dataset.is_custom or dataset.source_path is None:
        raise TrainingUIAPIError(f"Датасет редактора не найден: {dataset_key}")
    if dataset.images_dir is None or dataset.imagery_type is None:
        raise TrainingUIAPIError("Для датасета не настроен каталог снимков")
    return dataset


def _managed_editor_scene_count(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> int:
    row = session.scalar(
        select(DatasetRow).where(
            DatasetRow.key == dataset_key,
            DatasetRow.source_type == SOURCE_MANAGED,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if row is None:
        raise TrainingUIAPIError(f"Управляемый датасет редактора не найден: {dataset_key}")
    annotation_names = {
        scene.annotation_name.casefold()
        for scene in session.scalars(
            select(ManagedDatasetSceneRow).where(
                ManagedDatasetSceneRow.managed_dataset_id == row.id
            )
        ).all()
    }
    editor_root = config.mlmarkup_editor_root.resolve()
    for source in managed_sources(session, row.id):
        relative = PurePosixPath(source.dataset.source_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise TrainingUIAPIError(
                f"Некорректный source_path исходного датасета: {source.dataset.source_path}"
            )
        source_dir = editor_root.joinpath(*relative.parts).resolve()
        _ensure_within(
            source_dir,
            editor_root,
            "Источник управляемого датасета выходит за пределы editor-клона",
        )
        if not source_dir.is_dir():
            raise TrainingUIAPIError(
                "Источник управляемого датасета отсутствует в editor-клоне: "
                f"{source.dataset.source_path}"
            )
        if _direct_files(source_dir, ".txt"):
            raise TrainingUIAPIError(
                "Источник управляемого датасета не является per-image: "
                f"{source.dataset.source_path}"
            )
        annotation_names.update(
            path.name.casefold() for path in _direct_annotation_files(source_dir)
        )
    return len(annotation_names)


def _editor_dataset_context(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    allow_missing: bool = False,
) -> tuple[DatasetInfo, Path]:
    dataset = _managed_editor_dataset(session, config, dataset_key)
    if dataset.managed:
        row = session.scalar(
            select(DatasetRow).where(
                DatasetRow.key == dataset_key,
                DatasetRow.deleted_at.is_(None),
            )
        )
        if row is None:
            raise TrainingUIAPIError(f"Датасет редактора не найден: {dataset_key}")
        cache_state = managed_dataset_cache_state(
            session,
            config,
            row,
            source_root=config.mlmarkup_editor_root,
            scope="editor",
            request_if_missing=True,
            priority="urgent",
        )
        materialized = cache_state.materialization
        if materialized is None:
            message = cache_state.error or "Материализация поставлена в очередь."
            raise TrainingUIAPIError("Управляемый датасет подготавливается в фоне. " + message)
        dataset = dataset.model_copy(
            update={
                "path": str(materialized.path),
                "annotations_dir": str(materialized.path),
                "version": materialized.version,
                "updated_at": materialized.updated_at,
                "source_available": True,
                "materialization_status": "current",
                "materialized_version": materialized.version,
                "materialization_error": None,
            }
        )
        return dataset, materialized.path
    source_dir = _editor_source_dir(config, dataset)
    if source_dir.exists() and not source_dir.is_dir():
        raise TrainingUIAPIError("Источник датасета не является папкой")
    if not source_dir.exists() and not allow_missing:
        raise TrainingUIAPIError("Источник датасета отсутствует в editor-клоне")
    if source_dir.is_dir() and _direct_files(source_dir, ".txt"):
        raise TrainingUIAPIError("Редактор поддерживает только per-image датасеты без TXT")
    return dataset, source_dir


def _editor_pseudo_compatibility_error(
    dataset: DatasetInfo,
    result: TrainingResultRow,
) -> str | None:
    if dataset.task != result.task:
        return "Тип задачи основной сети не совпадает с типом датасета."
    if dataset.task == "multiclass":
        expected = [(item.id, item.slug) for item in dataset.object_types]
        actual = [
            (int(item.get("id") or 0), str(item.get("slug") or ""))
            for item in result.class_schema or []
            if isinstance(item, dict)
        ]
        if actual != expected:
            return "Схема типов основной сети не совпадает со схемой датасета."
    return None


def _editor_effective_training_result(
    session: Session,
    dataset_key: str,
    class_key: str,
) -> TrainingResultRow | None:
    """Выбрать сеть для подсказки редактора без неявного назначения звезды."""

    class_row = dataset_class_row(session, class_key or dataset_key)
    if class_row is not None and class_row.primary_training_result_id is not None:
        selected = session.get(TrainingResultRow, class_row.primary_training_result_id)
        if selected is not None and selected.status == ResultStatus.OK.value:
            return selected
    local_result = session.scalar(
        select(TrainingResultRow)
        .where(
            (
                (TrainingResultRow.dataset_key == dataset_key)
                | (TrainingResultRow.class_key == dataset_key)
            ),
            TrainingResultRow.status == ResultStatus.OK.value,
        )
        .order_by(
            TrainingResultRow.trained_at.desc().nullslast(),
            TrainingResultRow.created_at.desc(),
            TrainingResultRow.id.desc(),
        )
        .limit(1)
    )
    if local_result is not None:
        return local_result
    return current_primary_training_result(session, class_key or dataset_key)


def _latest_covering_pseudo_result(
    session: Session,
    dataset: DatasetInfo,
    training_result_id: uuid.UUID,
    image_path: Path,
) -> PseudoMarkupResultRow | None:
    rows = session.scalars(
        select(PseudoMarkupResultRow)
        .where(
            PseudoMarkupResultRow.training_result_id == training_result_id,
            PseudoMarkupResultRow.status == ResultStatus.OK.value,
            PseudoMarkupResultRow.geojson_file_id.is_not(None),
        )
        .options(
            selectinload(PseudoMarkupResultRow.scenes_file),
            selectinload(PseudoMarkupResultRow.geojson_file),
        )
        .order_by(
            PseudoMarkupResultRow.updated_at.desc(),
            PseudoMarkupResultRow.created_at.desc(),
        )
    ).all()
    return next(
        (
            row
            for row in rows
            if row.geojson_file is not None
            and Path(row.geojson_file.path).is_file()
            and _pseudo_result_covers_image(session, row, dataset, image_path)
        ),
        None,
    )


def _pseudo_result_covers_image(
    session: Session,
    result: PseudoMarkupResultRow,
    dataset: DatasetInfo,
    image_path: Path,
) -> bool:
    # После миграции legacy -> per-image исходный TXT результата может быть уже удалён.
    # Совпадающая управляемая ревизия означает, что результат построен для состава
    # именно этого датасета, включая сцены без единого предсказанного объекта.
    if (
        result.dataset_key == dataset.key
        and result.dataset_version is not None
        and result.dataset_version == dataset.version
    ):
        return True
    if result.scenes_file is None:
        return False
    try:
        entries = (
            result.scenes_file.path
            and Path(result.scenes_file.path).read_text(encoding="utf-8-sig").splitlines()
        )
    except OSError:
        return False
    resolved_image = image_path.resolve()
    expected = {_scene_reference_key(resolved_image.as_posix())}
    source_job = session.get(JobRow, result.job_id) if result.job_id is not None else None
    source_root = (source_job.config or {}).get("images_root") if source_job is not None else None
    root = (
        Path(str(source_root)).resolve() if source_root else _dataset_images_root(dataset).resolve()
    )
    try:
        expected.add(_scene_reference_key(resolved_image.relative_to(root).as_posix()))
    except ValueError:
        pass
    normalized = {
        _scene_reference_key(line)
        for line in entries
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not expected.isdisjoint(normalized):
        return True

    # Старые полные псевдоразметки могли сохранять в scenes_file только имя файла
    # TIFF, хотя per-image GeoJSON уже кодировал подпапку в имени. Такой результат
    # можно переиспользовать лишь при однозначном сопоставлении имени внутри
    # того же корня снимков: иначе одинаковые имена из разных папок смешают сцены.
    return _unique_basename_reference_matches(normalized, root, resolved_image)


def _unique_basename_reference_matches(
    normalized_references: set[str],
    images_root: Path,
    image_path: Path,
) -> bool:
    resolved_image = image_path.resolve()
    basename = _scene_reference_key(resolved_image.name)
    if basename not in {reference for reference in normalized_references if "/" not in reference}:
        return False
    candidates = {
        candidate.resolve()
        for candidate in build_image_index(images_root).get(basename.lower(), [])
    }
    return candidates == {resolved_image}


def _scene_reference_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.suffix.casefold() in RASTER_SUFFIXES:
        path = path.with_suffix("")
    return path.as_posix().casefold()


def _raster_revision(image_path: Path) -> str:
    status = image_path.stat()
    payload = f"{image_path.resolve()}\0{status.st_size}\0{status.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _editor_pseudo_inference_config_hash(selected: Any, tile_size: int) -> str:
    payload = {
        "algorithm_version": _EDITOR_PSEUDO_ALGORITHM_VERSION,
        "inference_template_id": (
            str(selected.inference_template_id)
            if selected.inference_template_id is not None
            else None
        ),
        "inference_template_config": selected.inference_template_config,
        "checkpoint_threshold": selected.checkpoint.threshold,
        "tile_size": tile_size,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _editor_pseudo_dedup_key(
    dataset_key: str,
    annotation_name: str,
    training_result_id: uuid.UUID,
    raster_revision: str,
    inference_config_hash: str,
) -> str:
    serialized = "\0".join(
        (
            dataset_key,
            annotation_name,
            str(training_result_id),
            raster_revision,
            inference_config_hash,
        )
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _latest_editor_pseudo_job(
    session: Session,
    dataset_key: str,
    annotation_name: str,
    training_result_id: uuid.UUID,
    raster_revision: str,
) -> JobRow | None:
    rows = session.scalars(
        select(JobRow)
        .where(
            JobRow.type == JobType.INFERENCE.value,
            JobRow.dataset_key == dataset_key,
        )
        .order_by(JobRow.created_at.desc(), JobRow.id.desc())
    ).all()
    for row in rows:
        if (row.config or {}).get("operation") != DATASET_EDITOR_PSEUDO_OPERATION:
            continue
        state = _editor_pseudo_state(row)
        if (
            state.get("annotation_name") == annotation_name
            and state.get("training_result_id") == str(training_result_id)
            and state.get("raster_revision") == raster_revision
        ):
            return row
    return None


def _editor_pseudo_state(row: JobRow) -> dict[str, Any]:
    state = (row.config or {}).get("editor_pseudo")
    return state if isinstance(state, dict) else {}


def _editor_pseudo_job_info(
    session: Session,
    row: JobRow,
    *,
    image_path: Path,
    training_result: TrainingResultRow,
) -> DatasetEditorPseudoMarkupInfo:
    state = _editor_pseudo_state(row)
    ready_file = _editor_pseudo_result_file(session, state)
    if row.status == JobStatus.COMPLETED.value and ready_file is not None:
        payload = _pseudo_geojson_for_image(Path(ready_file.path), image_path)
        return DatasetEditorPseudoMarkupInfo(
            status="ready",
            source="scene",
            training_result_id=training_result.id,
            model_name=training_result.model_name,
            job_id=row.id,
            object_count=len(payload["features"]),
            geojson=payload,
        )
    if row.status in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        current, total = _editor_pseudo_progress(row)
        return DatasetEditorPseudoMarkupInfo(
            status="queued" if row.status == JobStatus.QUEUED.value else "running",
            source="scene",
            training_result_id=training_result.id,
            model_name=training_result.model_name,
            job_id=row.id,
            progress_current=current,
            progress_total=total,
            message=(
                "Срочный инференс поставлен в очередь."
                if row.status == JobStatus.QUEUED.value
                else "Выполняется срочный инференс по снимку."
            ),
        )
    return DatasetEditorPseudoMarkupInfo(
        status="failed",
        source="scene",
        training_result_id=training_result.id,
        model_name=training_result.model_name,
        job_id=row.id,
        message=str(state.get("error") or row.error or "Инференс по снимку завершился ошибкой."),
        can_retry=True,
    )


def _reset_editor_pseudo_job(session: Session, row: JobRow) -> None:
    state = dict(_editor_pseudo_state(row))
    state["result_file_id"] = None
    state["error"] = None
    row.config = {**(row.config or {}), "editor_pseudo": state}
    row.status = JobStatus.QUEUED.value
    row.queue_position = next_queue_position(
        session,
        JobType.INFERENCE,
        JobSource.MANUAL,
    )
    row.process_pid = None
    row.tmp_path = None
    row.error = None
    row.started_at = None
    row.finished_at = None
    row.created_at = datetime.now(timezone.utc)
    session.flush()


def _editor_pseudo_result_file(
    session: Session,
    state: dict[str, Any],
) -> StoredFileRow | None:
    try:
        file_id = uuid.UUID(str(state.get("result_file_id")))
    except (TypeError, ValueError):
        return None
    row = session.get(StoredFileRow, file_id)
    return row if row is not None and Path(row.path).is_file() else None


def _editor_pseudo_progress(row: JobRow) -> tuple[int | None, int | None]:
    if row.tmp_path is None:
        return None, None
    path = Path(row.tmp_path) / "scratch" / "progress.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        current = int(payload.get("current")) if payload.get("current") is not None else None
        total = int(payload.get("total")) if payload.get("total") is not None else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, None
    return current, total


def _pseudo_geojson_for_image(path: Path, image_path: Path) -> dict[str, Any]:
    payload = _read_geojson(path)
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise TrainingUIAPIError("Псевдоразметка должна быть FeatureCollection.")
    try:
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise TrainingUIAPIError("У TIFF отсутствует CRS.")
            raster_crs = PyprojCRS.from_user_input(source.crs)
    except rasterio.errors.RasterioError as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF: {exc}") from exc
    footprint = _valid_data_footprint(image_path)
    to_wgs84 = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    footprint_wgs84 = transform_geometry(to_wgs84.transform, footprint)
    raw_crs = payload.get("crs")
    source_crs = _geojson_crs(payload) if raw_crs else PyprojCRS.from_epsg(4326)
    to_wgs84_geometry = (
        Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
        if source_crs != PyprojCRS.from_epsg(4326)
        else None
    )
    clipped: list[dict[str, Any]] = []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            continue
        try:
            geometry = shape(feature.get("geometry"))
            if to_wgs84_geometry is not None:
                geometry = transform_geometry(to_wgs84_geometry.transform, geometry)
            geometry = _polygonal_geometry(geometry.intersection(footprint_wgs84))
        except Exception:
            continue
        if geometry.is_empty:
            continue
        clipped.append({**feature, "geometry": dict(mapping(geometry))})
    return {
        **payload,
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": clipped,
    }


def _positive_integer(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _editor_source_dir(config: TrainingUIAPIConfig, dataset: DatasetInfo) -> Path:
    if dataset.source_path is None:
        raise TrainingUIAPIError("У датасета отсутствует source_path")
    relative = PurePosixPath(dataset.source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError("Некорректный source_path датасета")
    root = config.mlmarkup_editor_root.resolve()
    target = root.joinpath(*relative.parts).resolve()
    _ensure_within(target, root, "Источник датасета выходит за пределы editor-клона")
    return target


def _editor_dataset_info(
    dataset: DatasetInfo,
    scene_count: int,
    *,
    primary_training_result_id: uuid.UUID | None = None,
) -> DatasetEditorDatasetInfo:
    if dataset.imagery_type is None:
        raise TrainingUIAPIError("У датасета не задан тип снимков")
    return DatasetEditorDatasetInfo(
        key=dataset.key,
        name=dataset.name,
        class_key=dataset.class_key or dataset.key,
        class_name=dataset.class_name or dataset.name,
        dataset_name=dataset.dataset_name or "main",
        imagery_type=dataset.imagery_type.value,
        scene_count=scene_count,
        task=dataset.task,
        object_types=[
            DatasetEditorObjectType(
                id=item.id,
                slug=item.slug,
                name=item.name,
                color=item.color,
                priority=item.priority,
            )
            for item in dataset.object_types
        ],
        combined=dataset.combined,
        managed=dataset.managed,
        managed_sources=list(dataset.managed_sources),
        source_status=dataset.source_status,
        source_changes=list(dataset.source_changes),
        class_counts=dict(dataset.class_counts),
        hard_negative_count=dataset.hard_negative_count,
        primary_training_result_id=primary_training_result_id,
        materialization_status=dataset.materialization_status,
        materialized_version=dataset.materialized_version,
        materialization_error=dataset.materialization_error,
    )


def _scene_infos(
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    source_dir: Path,
) -> list[DatasetEditorSceneInfo]:
    if not source_dir.is_dir():
        return []
    annotation_paths = _direct_annotation_files(source_dir)
    revisions = (
        {}
        if dataset.managed
        else _blob_revisions(config, "HEAD", _repo_relative(config, source_dir))
    )
    result: list[DatasetEditorSceneInfo] = []
    for path in annotation_paths:
        relative_path = None if dataset.managed else _repo_relative(config, path)
        revision = _file_revision(path) if dataset.managed else revisions.get(relative_path)
        if revision is None:
            raise DatasetEditorGitError(f"GeoJSON не зафиксирован в Git: {path.name}")
        result.append(
            _scene_info_for_annotation(
                config,
                dataset,
                source_dir,
                path.name,
                revision=revision,
            )
        )
    result.sort(key=lambda item: item.scene_id.casefold())
    return result


def _add_managed_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    names: dict[str, tuple[str, Path]],
    *,
    username: str,
) -> DatasetEditorMutationResult:
    row = session.scalar(
        select(DatasetRow).where(
            DatasetRow.key == dataset.key,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if row is None:
        raise TrainingUIAPIError(f"Управляемый датасет не найден: {dataset.key}")
    del username  # Добавление пустой сцены меняет только состав в БД, публикации ещё нет.
    images_root = _dataset_images_root(dataset)
    for annotation_name, image_path in names.values():
        try:
            relative = image_path.resolve().relative_to(images_root).as_posix()
        except ValueError as exc:
            raise TrainingUIAPIError(
                f"Снимок выходит за каталог управляемого датасета: {image_path}"
            ) from exc
        session.add(
            ManagedDatasetSceneRow(
                managed_dataset_id=row.id,
                annotation_name=annotation_name,
                image_relative_path=relative,
            )
        )
    row.config_revision += 1
    session.flush()
    invalidate_managed_cache(config, dataset.key)
    managed_dataset_cache_state(
        session,
        config,
        row,
        source_root=config.mlmarkup_editor_root,
        scope="editor",
        request_if_missing=True,
        priority="urgent",
    )
    commit = _git(config, "rev-parse", "HEAD").stdout.strip()
    return DatasetEditorMutationResult(
        commit=commit,
        publication_status=_publication_status(config, commit),
        scenes=[],
    )


def _publish_managed_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    materialized_dir: Path,
    *,
    scenes: list[tuple[str, str, dict[str, Any]]],
    deletions: list[tuple[str, str]],
    username: str,
) -> DatasetEditorMutationResult:
    row = session.scalar(
        select(DatasetRow).where(
            DatasetRow.key == dataset.key,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if row is None:
        raise TrainingUIAPIError(f"Управляемый датасет не найден: {dataset.key}")
    source_specs = managed_sources(session, row.id)
    source_by_key = {item.dataset.key: item for item in source_specs}
    source_by_slug = {item.relation.object_type_slug: item for item in source_specs}

    normalized: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    conflicts: list[str] = []
    for annotation_name, revision, geojson in scenes:
        safe_name = _safe_annotation_name(annotation_name)
        current_path = materialized_dir / safe_name
        if _file_revision(current_path) != revision:
            conflicts.append(safe_name)
            continue
        image_path = _matched_image_path(dataset, materialized_dir, safe_name)
        previous = _reproject_editor_geojson(_read_geojson(current_path), image_path)
        candidate = _normalize_editor_geojson(
            geojson,
            previous,
            dataset,
            image_path=image_path,
        )
        _validate_editor_geojson(candidate, image_path, dataset)
        _validate_preserved_properties(previous, candidate)
        normalized.append((safe_name, previous, candidate))
    for annotation_name, revision in deletions:
        safe_name = _safe_annotation_name(annotation_name)
        if _file_revision(materialized_dir / safe_name) != revision:
            conflicts.append(safe_name)
    if conflicts:
        raise DatasetEditorConflict(
            "Разметка уже изменилась вместе с исходными датасетами: " + ", ".join(conflicts)
        )

    mutations: dict[tuple[str, str], tuple[Path, dict[str, Any], str | None]] = {}
    created_paths: list[Path] = []

    def source_payload(
        source_key: str, annotation_name: str
    ) -> tuple[Path, dict[str, Any], str | None]:
        cache_key = (source_key, annotation_name)
        cached = mutations.get(cache_key)
        if cached is not None:
            return cached
        source = source_by_key[source_key]
        folder = config.mlmarkup_editor_root.joinpath(
            *PurePosixPath(source.dataset.source_path).parts
        )
        path = folder / annotation_name
        relative = _repo_relative(config, path)
        revision = _blob_revision(config, "HEAD", relative)
        if path.is_file():
            payload = _read_geojson(path)
        else:
            source_info = find_managed_dataset(session, config, source_key)
            if source_info is None:
                raise TrainingUIAPIError(f"Исходный датасет не найден: {source_key}")
            image_path = _image_path_for_annotation(dataset, annotation_name)
            folder.mkdir(parents=True, exist_ok=True)
            payload = _empty_annotation_payload(image_path, source_info)
            payload[_MANAGED_OWNER_PROPERTY] = dataset.key
            created_paths.append(path)
        mutations[cache_key] = (path, payload, revision)
        return mutations[cache_key]

    for annotation_name, previous, candidate in normalized:
        previous_by_origin = _features_by_origin(previous)
        candidate_by_origin = _features_by_origin(candidate)
        previous_crs = _geojson_crs(previous)
        candidate_crs = _geojson_crs(candidate)
        for origin_key in sorted(set(previous_by_origin) | set(candidate_by_origin)):
            old_feature = previous_by_origin.get(origin_key)
            new_feature = candidate_by_origin.get(origin_key)
            if old_feature is not None and new_feature is not None:
                if _managed_editor_features_equal(old_feature, new_feature):
                    continue
            old_properties = (
                old_feature.get("properties") or {} if isinstance(old_feature, dict) else {}
            )
            old_source_key = old_properties.get("_mlsystem2_source_dataset_key")
            old_source = source_by_key.get(str(old_source_key))
            old_role = str(old_properties.get(_ROLE_PROPERTY) or "positive")
            if old_source is None:
                old_source = source_by_slug.get(str(old_properties.get(_CLASS_PROPERTY)))
            old_sources = (
                [old_source]
                if old_source is not None
                else (
                    source_specs if old_feature is not None and old_role == "hard_negative" else []
                )
            )
            new_sources = []
            if new_feature is not None:
                new_properties = new_feature.get("properties") or {}
                if new_properties.get(_ROLE_PROPERTY) == "positive":
                    new_source = source_by_slug.get(str(new_properties.get(_CLASS_PROPERTY)))
                    if new_source is not None:
                        new_sources = [new_source]
                else:
                    new_source = source_by_slug.get(str(new_properties.get(_CLASS_PROPERTY)))
                    if new_source is not None:
                        new_sources = [new_source]
                    elif new_properties.get(_CLASS_PROPERTY) is None:
                        new_sources = list(source_specs)
                if not new_sources:
                    raise TrainingUIAPIError(
                        f"Не найден исходный датасет для объекта {origin_key}."
                    )

            for source in old_sources:
                path, payload, revision = source_payload(
                    source.dataset.key,
                    annotation_name,
                )
                removed = _remove_managed_source_feature(
                    payload,
                    source_identity=old_properties.get("_mlsystem2_source_identity"),
                    source_origin_key=old_properties.get("_mlsystem2_source_origin_key"),
                    fallback_geometry=(old_feature or {}).get("geometry"),
                    fallback_crs=previous_crs,
                    fallback_role=old_role,
                )
                mutations[(source.dataset.key, annotation_name)] = (
                    path,
                    removed,
                    revision,
                )
            for source in new_sources:
                path, payload, revision = source_payload(
                    source.dataset.key,
                    annotation_name,
                )
                added = _append_managed_source_feature(
                    payload,
                    new_feature,
                    source_crs=candidate_crs,
                    origin_key=origin_key,
                )
                mutations[(source.dataset.key, annotation_name)] = (
                    path,
                    added,
                    revision,
                )

    explicit_deletion_names = {
        name
        for name in session.scalars(
            select(ManagedDatasetSceneRow.annotation_name).where(
                ManagedDatasetSceneRow.managed_dataset_id == row.id,
                ManagedDatasetSceneRow.annotation_name.in_([item[0] for item in deletions]),
            )
        ).all()
    }
    deletion_paths: dict[PurePosixPath, str | None] = {}
    for annotation_name, _revision in deletions:
        safe_name = _safe_annotation_name(annotation_name)
        for source in source_specs:
            folder = config.mlmarkup_editor_root.joinpath(
                *PurePosixPath(source.dataset.source_path).parts
            )
            for name in (safe_name, footprint_name_for_annotation(safe_name)):
                path = folder / name
                if not path.is_file():
                    continue
                relative = _repo_relative(config, path)
                deletion_paths[relative] = _blob_revision(config, "HEAD", relative)

    for key, (path, payload, revision) in list(mutations.items()):
        features = payload.get("features")
        if (
            isinstance(features, list)
            and not features
            and payload.get(_MANAGED_OWNER_PROPERTY) == dataset.key
        ):
            if revision is not None:
                relative = _repo_relative(config, path)
                deletion_paths[relative] = revision
                footprint = path.parent / footprint_name_for_annotation(path.name)
                if footprint.is_file():
                    footprint_relative = _repo_relative(config, footprint)
                    deletion_paths[footprint_relative] = _blob_revision(
                        config,
                        "HEAD",
                        footprint_relative,
                    )
            mutations.pop(key)

    expected_revisions: dict[PurePosixPath, str | None] = {}
    written_paths: list[PurePosixPath] = []
    try:
        for (_source_key, annotation_name), (path, payload, revision) in mutations.items():
            _write_geojson_atomic(path, payload)
            relative = _repo_relative(config, path)
            expected_revisions[relative] = revision
            written_paths.append(relative)
            footprint = path.parent / footprint_name_for_annotation(annotation_name)
            if not footprint.is_file():
                _write_geojson_atomic(
                    footprint,
                    _footprint_geojson_payload(
                        _image_path_for_annotation(dataset, annotation_name)
                    ),
                )
                footprint_relative = _repo_relative(config, footprint)
                expected_revisions[footprint_relative] = None
                written_paths.append(footprint_relative)
                created_paths.append(footprint)
        if written_paths:
            _git(config, "add", "--", *(path.as_posix() for path in written_paths))
        if deletion_paths:
            _git(config, "rm", "--", *(path.as_posix() for path in deletion_paths))
            expected_revisions.update(deletion_paths)
        if not written_paths and not deletion_paths and not explicit_deletion_names:
            raise TrainingUIAPIError("В публикации нет фактических изменений исходных датасетов.")
        if written_paths or deletion_paths:
            commit = _commit(
                config,
                f"Опубликовать правки управляемого датасета {dataset.dataset_name or dataset.name}",
                username,
            )
            commit = _push_with_retry(config, expected_revisions=expected_revisions)
        else:
            commit = _git(config, "rev-parse", "HEAD").stdout.strip()
    except Exception:
        for path in created_paths:
            if _repo_relative(config, path) in {
                relative for relative, revision in expected_revisions.items() if revision is None
            }:
                path.unlink(missing_ok=True)
        restore_paths = [*written_paths, *deletion_paths]
        if restore_paths:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                *(path.as_posix() for path in restore_paths),
            )
        raise

    if explicit_deletion_names:
        session.execute(
            delete(ManagedDatasetSceneRow).where(
                ManagedDatasetSceneRow.managed_dataset_id == row.id,
                ManagedDatasetSceneRow.annotation_name.in_(explicit_deletion_names),
            )
        )
        row.config_revision += 1
        session.flush()
    invalidate_managed_cache(config, dataset.key)
    managed_dataset_cache_state(
        session,
        config,
        row,
        source_root=config.mlmarkup_editor_root,
        scope="editor",
        request_if_missing=True,
        priority="urgent",
    )
    if normalized:
        session.execute(
            delete(DatasetEditorDraftRow).where(
                DatasetEditorDraftRow.dataset_key == dataset.key,
                DatasetEditorDraftRow.username == username,
                DatasetEditorDraftRow.annotation_name.in_([item[0] for item in normalized]),
            )
        )
    if deletions:
        session.execute(
            delete(DatasetEditorDraftRow).where(
                DatasetEditorDraftRow.dataset_key == dataset.key,
                DatasetEditorDraftRow.annotation_name.in_([item[0] for item in deletions]),
            )
        )
    session.flush()
    return DatasetEditorMutationResult(
        commit=commit,
        publication_status=_publication_status(config, commit),
        scenes=[],
    )


def _remove_managed_source_feature(
    payload: dict[str, Any],
    *,
    source_identity: Any,
    source_origin_key: Any,
    fallback_geometry: Any,
    fallback_crs: PyprojCRS,
    fallback_role: str | None = None,
) -> dict[str, Any]:
    features = [item for item in payload.get("features") or [] if isinstance(item, dict)]
    for index, feature in enumerate(features, start=1):
        properties = feature.get("properties") or {}
        identity = _managed_source_identity(feature, properties, index)
        origin = properties.get("_mlsystem2_origin_key")
        if (isinstance(source_identity, str) and identity == source_identity) or (
            isinstance(source_origin_key, str) and source_origin_key and origin == source_origin_key
        ):
            return {**payload, "features": [*features[: index - 1], *features[index:]]}
    if fallback_geometry is None:
        return payload
    target_crs = _geojson_crs(payload)
    geometry = shape(fallback_geometry)
    if fallback_crs != target_crs:
        transformer = Transformer.from_crs(fallback_crs, target_crs, always_xy=True)
        geometry = transform_geometry(transformer.transform, geometry)
    output: list[dict[str, Any]] = []
    for feature in features:
        properties = feature.get("properties") or {}
        role = str(properties.get(_ROLE_PROPERTY) or "positive")
        if fallback_role is not None and role != fallback_role:
            output.append(feature)
            continue
        try:
            current = _polygonal_geometry(shape(feature.get("geometry")))
            remainder = _polygonal_geometry(current.difference(geometry))
        except Exception:
            output.append(feature)
            continue
        if remainder.is_empty or remainder.area <= 0:
            continue
        output.append({**feature, "geometry": dict(mapping(remainder))})
    return {**payload, "features": output}


def _managed_editor_features_equal(
    previous: dict[str, Any],
    candidate: dict[str, Any],
) -> bool:
    previous_properties = previous.get("properties") or {}
    candidate_properties = candidate.get("properties") or {}
    if previous_properties.get(_ROLE_PROPERTY) != candidate_properties.get(_ROLE_PROPERTY):
        return False
    if previous_properties.get(_CLASS_PROPERTY) != candidate_properties.get(_CLASS_PROPERTY):
        return False
    try:
        return shape(previous.get("geometry")).equals(shape(candidate.get("geometry")))
    except Exception:  # noqa: BLE001
        return False


def _append_managed_source_feature(
    payload: dict[str, Any],
    feature: dict[str, Any],
    *,
    source_crs: PyprojCRS,
    origin_key: str,
) -> dict[str, Any]:
    target_crs = _geojson_crs(payload)
    geometry = _polygonal_geometry(shape(feature.get("geometry")))
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        geometry = _polygonal_geometry(transform_geometry(transformer.transform, geometry))
    properties = dict(feature.get("properties") or {})
    properties = {
        key: value for key, value in properties.items() if not key.startswith("_mlsystem2_")
    }
    source_properties = feature.get("properties") or {}
    role = str(source_properties.get(_ROLE_PROPERTY) or "positive")
    properties[_ROLE_PROPERTY] = role
    source_origin = source_properties.get("_mlsystem2_source_origin_key")
    properties["_mlsystem2_origin_key"] = (
        source_origin if isinstance(source_origin, str) and source_origin else origin_key
    )
    added = {
        "type": "Feature",
        "id": feature.get("id") or str(uuid.uuid4()),
        "properties": properties,
        "geometry": dict(mapping(geometry)),
    }
    return {**payload, "features": [*(payload.get("features") or []), added]}


def _managed_source_identity(
    feature: dict[str, Any],
    properties: dict[str, Any],
    index: int,
) -> str:
    clean_properties = {
        key: value for key, value in properties.items() if not key.startswith("_mlsystem2_")
    }
    content_payload = json.dumps(
        {
            "geometry": feature.get("geometry"),
            "properties": clean_properties,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content = hashlib.sha256(content_payload.encode("utf-8")).hexdigest()
    if feature.get("id") is not None:
        return (
            "feature-id:"
            + json.dumps(feature["id"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + f"|content:{content}"
        )
    for key in ("id", "fid"):
        if properties.get(key) is not None:
            return (
                f"property-{key}:"
                + json.dumps(
                    properties[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + f"|content:{content}"
            )
    return f"index:{index}|content:{content}"


def _scene_info_for_annotation(
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    source_dir: Path,
    annotation_name: str,
    *,
    revision: str | None = None,
) -> DatasetEditorSceneInfo:
    annotation_path = _annotation_path(source_dir, annotation_name)
    if revision is None:
        if dataset.managed:
            revision = _file_revision(annotation_path)
        else:
            relative_annotation = _repo_relative(config, annotation_path)
            revision = _blob_revision(config, "HEAD", relative_annotation)
    if revision is None:
        raise DatasetEditorGitError(f"GeoJSON не зафиксирован в Git: {annotation_path.name}")
    positive, hard_negative, class_counts = _editor_counts(
        _read_geojson(annotation_path),
        dataset,
    )
    root = _dataset_images_root(dataset)
    image_path = _matched_image_path(dataset, source_dir, annotation_path.name).resolve()
    image_relative = image_path.relative_to(root).as_posix()
    return DatasetEditorSceneInfo(
        scene_id=image_path.relative_to(root).with_suffix("").as_posix(),
        annotation_name=annotation_path.name,
        image_name=image_path.name,
        raster_url=(
            "/api/v1/dataset-editor/datasets/"
            f"{quote(dataset.key, safe='')}/raster/{quote(image_relative, safe='/')}"
        ),
        total_count=positive + hard_negative,
        positive_count=positive,
        hard_negative_count=hard_negative,
        revision=revision,
        class_counts=class_counts,
    )


def _scene_by_annotation(
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    source_dir: Path,
    annotation_name: str,
) -> DatasetEditorSceneInfo:
    safe_name = _safe_annotation_name(annotation_name)
    if not (source_dir / safe_name).is_file():
        raise TrainingUIAPIError(f"Снимок датасета не найден: {safe_name}")
    return _scene_info_for_annotation(config, dataset, source_dir, safe_name)


def _matched_image_path(
    dataset: DatasetInfo,
    source_dir: Path,
    annotation_name: str,
) -> Path:
    safe_name = _safe_annotation_name(annotation_name)
    if not (source_dir / safe_name).is_file():
        raise TrainingUIAPIError(f"GeoJSON датасета не найден: {safe_name}")
    return _image_path_for_annotation(dataset, safe_name)


def _image_path_for_annotation(
    dataset: DatasetInfo,
    annotation_name: str,
) -> Path:
    safe_name = _safe_annotation_name(annotation_name)
    candidates = build_per_image_index(_dataset_images_root(dataset)).get(
        safe_name.casefold(),
        [],
    )
    if not candidates:
        raise TrainingUIAPIError(f"Для GeoJSON не найден TIFF: {safe_name}")
    if len(candidates) > 1:
        raise TrainingUIAPIError(f"Имя GeoJSON неоднозначно сопоставлено с TIFF: {safe_name}")
    return candidates[0]


def _attach_draft_summaries(
    session: Session,
    dataset: DatasetInfo,
    scenes: list[DatasetEditorSceneInfo],
    username: str,
) -> list[DatasetEditorSceneInfo]:
    rows = session.scalars(
        select(DatasetEditorDraftRow).where(
            DatasetEditorDraftRow.dataset_key == dataset.key,
            DatasetEditorDraftRow.username == username,
        )
    ).all()
    drafts = {row.annotation_name: row for row in rows}
    return [
        scene.model_copy(
            update={
                "draft": (
                    _draft_summary(drafts[scene.annotation_name], dataset, scene.revision)
                    if scene.annotation_name in drafts
                    else None
                )
            }
        )
        for scene in scenes
    ]


def _editor_draft_row(
    session: Session,
    dataset_key: str,
    annotation_name: str,
    username: str,
) -> DatasetEditorDraftRow | None:
    return session.scalar(
        select(DatasetEditorDraftRow).where(
            DatasetEditorDraftRow.dataset_key == dataset_key,
            DatasetEditorDraftRow.annotation_name == annotation_name,
            DatasetEditorDraftRow.username == username,
        )
    )


def _draft_summary(
    row: DatasetEditorDraftRow,
    dataset: DatasetInfo,
    current_revision: str,
) -> DatasetEditorDraftSummary:
    positive, hard_negative, class_counts = _editor_counts(dict(row.geojson), dataset)
    return DatasetEditorDraftSummary(
        annotation_name=row.annotation_name,
        base_revision=row.base_revision,
        deleted=row.deleted,
        stale=row.base_revision != current_revision,
        total_count=positive + hard_negative,
        positive_count=positive,
        hard_negative_count=hard_negative,
        class_counts=class_counts,
        updated_at=row.updated_at,
    )


def _draft_info(
    row: DatasetEditorDraftRow,
    dataset: DatasetInfo,
    current_revision: str,
    *,
    geojson: dict[str, Any] | None = None,
) -> DatasetEditorDraftInfo:
    return DatasetEditorDraftInfo(
        **_draft_summary(row, dataset, current_revision).model_dump(),
        geojson=dict(row.geojson) if geojson is None else geojson,
    )


def _dataset_images_root(dataset: DatasetInfo) -> Path:
    if dataset.images_dir is None:
        raise TrainingUIAPIError("Каталог снимков датасета не настроен")
    root = Path(dataset.images_dir).resolve()
    if not root.is_dir():
        raise TrainingUIAPIError("Каталог снимков датасета недоступен")
    return root


def _safe_relative_directory(root: Path, value: str) -> tuple[PurePosixPath, Path]:
    normalized = value.strip().replace("\\", "/").strip("/")
    relative = PurePosixPath(normalized) if normalized else PurePosixPath()
    if relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError("Папка снимков выходит за пределы разрешённого каталога")
    target = root.joinpath(*relative.parts).resolve()
    _ensure_within(target, root, "Папка снимков выходит за пределы разрешённого каталога")
    if not target.is_dir():
        raise TrainingUIAPIError(f"Папка снимков не найдена: {value}")
    return relative, target


def _safe_raster_path(root: Path, value: str) -> Path:
    normalized = value.strip().replace("\\", "/").strip("/")
    relative = PurePosixPath(normalized)
    if not normalized or relative.is_absolute() or ".." in relative.parts:
        raise TrainingUIAPIError("Некорректный путь TIFF")
    target = root.joinpath(*relative.parts).resolve()
    _ensure_within(target, root, "TIFF выходит за пределы разрешённого каталога")
    if not target.is_file() or target.suffix.casefold() not in RASTER_SUFFIXES:
        raise TrainingUIAPIError(f"TIFF не найден: {value}")
    return target


def _annotation_path(source_dir: Path, annotation_name: str) -> Path:
    return source_dir / _safe_annotation_name(annotation_name)


def _safe_annotation_name(value: str) -> str:
    name = Path(value).name
    if (
        name != value
        or Path(name).suffix.casefold() != ".geojson"
        or is_per_image_footprint_name(name)
    ):
        raise TrainingUIAPIError("Некорректное имя GeoJSON")
    return name


def _empty_annotation_payload(
    image_path: Path,
    dataset: DatasetInfo,
) -> dict[str, Any]:
    crs_name = _editor_raster_crs(image_path).to_string()
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": [],
    }
    if dataset.task == "multiclass":
        payload.update(
            {
                "_mlsystem2_schema_version": 1,
                "_mlsystem2_task": "multiclass",
                "_mlsystem2_classes": [
                    item.model_dump(mode="json") for item in dataset.object_types
                ],
            }
        )
    return payload


def _footprint_geojson_payload(image_path: Path) -> dict[str, Any]:
    crs_name = _editor_raster_crs(image_path).to_string()
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": [
            {
                "type": "Feature",
                "properties": {"_mlsystem2_type": "valid_data_footprint"},
                "geometry": dict(mapping(_valid_data_footprint(image_path))),
            }
        ],
    }


def _editor_raster_crs(image_path: Path) -> PyprojCRS:
    try:
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise TrainingUIAPIError(f"У TIFF отсутствует CRS: {image_path.name}")
            raster_crs = PyprojCRS.from_user_input(source.crs)
    except rasterio.errors.RasterioError as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF {image_path.name}: {exc}") from exc
    if raster_crs != _EDITOR_RASTER_CRS:
        raise TrainingUIAPIError(
            f"CRS TIFF должен быть EPSG:3857: {image_path.name}; получено {raster_crs.to_string()}"
        )
    return raster_crs


def _editor_crs_payload(crs: PyprojCRS) -> dict[str, Any]:
    return {"type": "name", "properties": {"name": crs.to_string()}}


def _reproject_editor_geojson(
    payload: dict[str, Any],
    image_path: Path,
) -> dict[str, Any]:
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise TrainingUIAPIError("GeoJSON должен быть FeatureCollection со списком features")
    source_crs = _geojson_crs(payload)
    raster_crs = _editor_raster_crs(image_path)
    if source_crs == raster_crs:
        return {**payload, "crs": _editor_crs_payload(raster_crs)}

    transformer = Transformer.from_crs(source_crs, raster_crs, always_xy=True)
    transformed_features: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise TrainingUIAPIError(f"Объект {index} не является GeoJSON Feature")
        try:
            geometry = shape(feature.get("geometry"))
            geometry = transform_geometry(transformer.transform, geometry)
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(
                f"Не удалось преобразовать CRS объекта {index}: {exc}"
            ) from exc
        transformed_features.append({**feature, "geometry": dict(mapping(geometry))})
    return {
        **payload,
        "crs": _editor_crs_payload(raster_crs),
        "features": transformed_features,
    }


def _validate_editor_geojson(
    payload: dict[str, Any],
    image_path: Path,
    dataset: DatasetInfo,
) -> None:
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise TrainingUIAPIError("GeoJSON должен быть FeatureCollection со списком features")
    geojson_crs = _geojson_crs(payload)
    raster_crs = _editor_raster_crs(image_path)
    footprint = _valid_data_footprint(image_path)
    if geojson_crs != raster_crs:
        raise TrainingUIAPIError(
            f"CRS GeoJSON ({geojson_crs.to_string()}) не совпадает с CRS TIFF "
            f"({raster_crs.to_string()})"
        )
    known_slugs = {item.slug for item in dataset.object_types}
    if dataset.task == "multiclass":
        if payload.get("_mlsystem2_schema_version") != 1:
            raise TrainingUIAPIError("Некорректная версия схемы multiclass-разметки")
        if payload.get("_mlsystem2_task") != "multiclass":
            raise TrainingUIAPIError("Некорректный task multiclass-разметки")
        actual_classes = payload.get("_mlsystem2_classes")
        expected_classes = [item.model_dump(mode="json") for item in dataset.object_types]
        if actual_classes != expected_classes:
            raise TrainingUIAPIError("Схема классов GeoJSON не совпадает с датасетом")
    for index, feature in enumerate(payload["features"], start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise TrainingUIAPIError(f"Объект {index} не является GeoJSON Feature")
        properties = feature.get("properties")
        if not isinstance(properties, dict) or properties.get(_ROLE_PROPERTY) not in _ROLES:
            raise TrainingUIAPIError(
                f"У объекта {index} должна быть явная роль positive или hard_negative"
            )
        role = str(properties[_ROLE_PROPERTY])
        class_slug = properties.get(_CLASS_PROPERTY)
        if dataset.task == "multiclass" and role == "positive":
            if class_slug not in known_slugs:
                raise TrainingUIAPIError(
                    f"У positive-объекта {index} должен быть один из классов датасета"
                )
        elif dataset.task == "multiclass" and role == "hard_negative" and dataset.managed:
            if class_slug is not None and class_slug not in known_slugs:
                raise TrainingUIAPIError(
                    f"У hard negative объекта {index} указан неизвестный исходный класс"
                )
        elif _CLASS_PROPERTY in properties:
            raise TrainingUIAPIError(
                f"У hard negative или binary-объекта {index} не должно быть класса"
            )
        try:
            geometry = shape(feature.get("geometry"))
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(f"Некорректная геометрия объекта {index}: {exc}") from exc
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise TrainingUIAPIError(f"Объект {index} должен быть Polygon или MultiPolygon")
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise TrainingUIAPIError(f"Геометрия объекта {index} пуста или невалидна")
        if not _footprint_covers_geometry(footprint, geometry):
            raise TrainingUIAPIError(
                f"Геометрия объекта {index} выходит за реальный footprint TIFF"
            )


def _valid_data_footprint(image_path: Path) -> BaseGeometry:
    try:
        status = image_path.stat()
    except OSError as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать TIFF {image_path.name}: {exc}") from exc
    return _cached_valid_data_footprint(
        str(image_path.resolve()),
        status.st_mtime_ns,
        status.st_size,
    )


def _footprint_covers_geometry(
    footprint: BaseGeometry,
    geometry: BaseGeometry,
) -> bool:
    if footprint.covers(geometry):
        return True
    outside_area = geometry.difference(footprint).area
    numerical_tolerance = max(geometry.area, 1.0) * 1e-12
    return outside_area <= numerical_tolerance


@lru_cache(maxsize=64)
def _cached_valid_data_footprint(
    image_path: str,
    _modified_ns: int,
    _size_bytes: int,
) -> BaseGeometry:
    try:
        with rasterio.open(image_path) as source:
            if source.width <= 0 or source.height <= 0:
                raise TrainingUIAPIError(f"TIFF не содержит пикселей: {Path(image_path).name}")
            scale = min(
                1.0,
                _VALID_FOOTPRINT_MAX_SIDE / max(source.width, source.height),
            )
            sample_width = max(1, int(round(source.width * scale)))
            sample_height = max(1, int(round(source.height * scale)))
            valid_mask = (
                source.dataset_mask(
                    out_shape=(sample_height, sample_width),
                    resampling=Resampling.nearest,
                )
                > 0
            )
            if not bool(valid_mask.any()):
                raise TrainingUIAPIError(
                    f"TIFF не содержит валидных пикселей: {Path(image_path).name}"
                )
            mask_transform = source.transform * Affine.scale(
                source.width / sample_width,
                source.height / sample_height,
            )
            if bool(valid_mask.all()):
                footprint: BaseGeometry = Polygon(
                    (
                        source.transform * (0, 0),
                        source.transform * (source.width, 0),
                        source.transform * (source.width, source.height),
                        source.transform * (0, source.height),
                    )
                )
            else:
                parts = [
                    shape(geometry)
                    for geometry, value in shapes(
                        valid_mask.astype("uint8", copy=False),
                        mask=valid_mask,
                        transform=mask_transform,
                    )
                    if int(value) == 1
                ]
                footprint = _polygonal_geometry(unary_union(parts))
                tolerance = (
                    max(
                        abs(mask_transform.a),
                        abs(mask_transform.b),
                        abs(mask_transform.d),
                        abs(mask_transform.e),
                    )
                    * _VALID_FOOTPRINT_SIMPLIFY_CELLS
                )
                if tolerance > 0:
                    footprint = _polygonal_geometry(
                        footprint.simplify(tolerance, preserve_topology=True)
                    )
    except TrainingUIAPIError:
        raise
    except (OSError, rasterio.errors.RasterioError) as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF {Path(image_path).name}: {exc}") from exc
    if footprint.is_empty or footprint.area <= 0:
        raise TrainingUIAPIError(
            f"Не удалось построить footprint валидных данных: {Path(image_path).name}"
        )
    return footprint


def _clip_geojson_to_footprint(
    payload: dict[str, Any],
    footprint: BaseGeometry,
) -> dict[str, Any]:
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise TrainingUIAPIError("GeoJSON должен быть FeatureCollection со списком features")
    clipped_features: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise TrainingUIAPIError(f"Объект {index} не является GeoJSON Feature")
        try:
            geometry = shape(feature.get("geometry"))
            if isinstance(geometry, (Polygon, MultiPolygon)):
                geometry = _polygonal_geometry(geometry)
            geometry = _polygonal_geometry(geometry.intersection(footprint))
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(
                f"Не удалось обрезать геометрию объекта {index}: {exc}"
            ) from exc
        if geometry.is_empty:
            continue
        clipped_features.append({**feature, "geometry": dict(mapping(geometry))})
    return {**payload, "features": clipped_features}


def _polygonal_geometry(geometry: BaseGeometry) -> BaseGeometry:
    repaired = make_valid(geometry) if not geometry.is_valid else geometry
    if isinstance(repaired, (Polygon, MultiPolygon)):
        return repaired
    polygons: list[Polygon] = []
    for part in getattr(repaired, "geoms", ()):
        if isinstance(part, Polygon):
            polygons.append(part)
        elif isinstance(part, MultiPolygon):
            polygons.extend(part.geoms)
    if not polygons:
        return Polygon()
    merged = unary_union(polygons)
    return make_valid(merged) if not merged.is_valid else merged


def _validate_preserved_properties(
    previous: dict[str, Any],
    updated: dict[str, Any],
) -> None:
    previous_by_id = {
        json.dumps(feature.get("id"), ensure_ascii=False, sort_keys=True): feature
        for feature in previous.get("features", [])
        if isinstance(feature, dict) and "id" in feature
    }
    previous_property_sets = {
        _non_system_properties(feature)
        for feature in previous.get("features", [])
        if isinstance(feature, dict)
    }
    for feature in updated.get("features", []):
        if not isinstance(feature, dict):
            continue
        identity = (
            json.dumps(feature.get("id"), ensure_ascii=False, sort_keys=True)
            if "id" in feature
            else None
        )
        if identity is not None and identity in previous_by_id:
            if _non_system_properties(feature) != _non_system_properties(previous_by_id[identity]):
                raise TrainingUIAPIError("Существующие свойства объектов изменять нельзя")
        properties = _non_system_properties(feature)
        if properties != "{}" and properties not in previous_property_sets:
            raise TrainingUIAPIError("Редактор не поддерживает произвольные атрибуты объектов")


def _non_system_properties(feature: dict[str, Any]) -> str:
    properties = feature.get("properties")
    cleaned = (
        {key: value for key, value in properties.items() if not key.startswith("_mlsystem2_")}
        if isinstance(properties, dict)
        else {}
    )
    return json.dumps(cleaned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _geojson_crs(payload: dict[str, Any]) -> PyprojCRS:
    raw_crs = payload.get("crs")
    value: Any = raw_crs
    if isinstance(raw_crs, dict):
        properties = raw_crs.get("properties")
        if isinstance(properties, dict):
            value = properties.get("name") or properties.get("href")
        if not value:
            value = raw_crs.get("name")
    if not value:
        raise TrainingUIAPIError("В GeoJSON должен быть явно указан CRS снимка")
    try:
        return PyprojCRS.from_user_input(value)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(f"Некорректный CRS GeoJSON: {value}") from exc


def _role_counts(payload: dict[str, Any]) -> tuple[int, int]:
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise TrainingUIAPIError("GeoJSON разметки должен быть FeatureCollection")
    positive = 0
    hard_negative = 0
    for feature in features:
        if not isinstance(feature, dict):
            raise TrainingUIAPIError("GeoJSON содержит некорректный Feature")
        properties = feature.get("properties")
        role = (
            properties.get(_ROLE_PROPERTY, "positive")
            if isinstance(properties, dict)
            else "positive"
        )
        if role == "positive":
            positive += 1
        elif role == "hard_negative":
            hard_negative += 1
        else:
            raise TrainingUIAPIError(f"Неизвестная роль объекта: {role}")
    return positive, hard_negative


def _editor_counts(
    payload: dict[str, Any],
    dataset: DatasetInfo,
) -> tuple[int, int, dict[str, int]]:
    positive, hard_negative = _role_counts(payload)
    class_counts = {item.slug: 0 for item in dataset.object_types}
    for feature in payload.get("features", []):
        properties = feature.get("properties") if isinstance(feature, dict) else None
        if not isinstance(properties, dict) or properties.get(_ROLE_PROPERTY) != "positive":
            continue
        slug = properties.get(_CLASS_PROPERTY)
        if slug in class_counts:
            class_counts[str(slug)] += 1
    return positive, hard_negative, class_counts


def _normalize_editor_geojson(
    payload: dict[str, Any],
    previous: dict[str, Any],
    dataset: DatasetInfo,
    *,
    image_path: Path,
) -> dict[str, Any]:
    incoming_crs = _geojson_crs(payload)
    raster_crs = _editor_raster_crs(image_path)
    existing_feature_transformer = (
        Transformer.from_crs(incoming_crs, raster_crs, always_xy=True)
        if incoming_crs != raster_crs
        else None
    )
    normalized = {
        **payload,
        "crs": _editor_crs_payload(raster_crs),
    }
    if dataset.task == "multiclass":
        normalized["_mlsystem2_schema_version"] = 1
        normalized["_mlsystem2_task"] = "multiclass"
        normalized["_mlsystem2_classes"] = [
            item.model_dump(mode="json") for item in dataset.object_types
        ]
    previous_by_id = {
        str(feature.get("id")): feature
        for feature in previous.get("features", [])
        if isinstance(feature, dict) and feature.get("id") is not None
    }
    features: list[dict[str, Any]] = []
    for raw_feature in payload.get("features", []):
        if not isinstance(raw_feature, dict):
            features.append(raw_feature)
            continue
        feature = dict(raw_feature)
        feature_id = feature.get("id")
        if feature_id is None or not str(feature_id):
            feature_id = str(uuid.uuid4())
            feature["id"] = feature_id
        previous_feature = previous_by_id.get(str(feature_id))
        if previous_feature is not None and existing_feature_transformer is not None:
            try:
                geometry = shape(feature.get("geometry"))
                geometry = transform_geometry(
                    existing_feature_transformer.transform,
                    geometry,
                )
            except Exception as exc:  # noqa: BLE001
                raise TrainingUIAPIError(
                    f"Не удалось преобразовать CRS объекта {feature_id}: {exc}"
                ) from exc
            feature["geometry"] = dict(mapping(geometry))
        properties = dict(feature.get("properties") or {})
        if previous_feature is not None:
            previous_properties = previous_feature.get("properties")
            if isinstance(previous_properties, dict):
                for key, value in previous_properties.items():
                    if key.startswith("_mlsystem2_") and key not in {
                        _ROLE_PROPERTY,
                        _CLASS_PROPERTY,
                    }:
                        properties[key] = value
        else:
            properties.setdefault("_mlsystem2_origin_key", f"manual:{feature_id}")
            properties.setdefault("_mlsystem2_source_path", "manual")
        if properties.get(_ROLE_PROPERTY) == "hard_negative" and not dataset.managed:
            properties.pop(_CLASS_PROPERTY, None)
        feature["properties"] = properties
        features.append(feature)
    normalized["features"] = features
    return normalized


def _read_geojson(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise TrainingUIAPIError(f"Не удалось прочитать GeoJSON {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrainingUIAPIError(f"GeoJSON {path.name} должен быть объектом")
    return payload


def _write_geojson_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_revision(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _direct_files(path: Path, suffix: str) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.casefold() == suffix.casefold()
        ),
        key=lambda item: item.name.casefold(),
    )


def _direct_annotation_files(path: Path) -> list[Path]:
    return [
        item
        for item in _direct_files(path, ".geojson")
        if not is_per_image_footprint_name(item.name)
    ]


def _repo_relative(config: TrainingUIAPIConfig, path: Path) -> PurePosixPath:
    root = config.mlmarkup_editor_root.resolve()
    try:
        return PurePosixPath(path.resolve().relative_to(root).as_posix())
    except ValueError as exc:
        raise TrainingUIAPIError("Файл выходит за пределы editor-клона") from exc


def _ensure_within(path: Path, root: Path, message: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TrainingUIAPIError(message) from exc


@contextmanager
def _editor_lock(
    config: TrainingUIAPIConfig,
    *,
    restore_ownership: bool = False,
) -> Iterator[None]:
    lock_path = config.mlmarkup_editor_root.parent / ".mlmarkup-editor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                try:
                    if restore_ownership:
                        _restore_editor_clone_ownership(config)
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                try:
                    if restore_ownership:
                        _restore_editor_clone_ownership(config)
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _restore_editor_clone_ownership(config: TrainingUIAPIConfig) -> None:
    """Вернуть всему клону владельца его корневого каталога."""

    if os.name == "nt" or not hasattr(os, "chown"):
        return
    root = config.mlmarkup_editor_root
    try:
        root_status = root.stat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DatasetEditorGitError(
            "Не удалось определить владельца editor-клона MLMarkup"
        ) from exc

    owner = (root_status.st_uid, root_status.st_gid)

    def restore(path: Path) -> None:
        try:
            status = path.lstat()
            if (status.st_uid, status.st_gid) != owner:
                os.chown(path, *owner, follow_symlinks=False)
        except FileNotFoundError:
            return

    try:
        restore(root)
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(directory)
            for name in (*directory_names, *file_names):
                restore(base / name)
    except OSError as exc:
        raise DatasetEditorGitError(
            "Не удалось восстановить владельца editor-клона MLMarkup"
        ) from exc


def _synchronize_editor_clone(config: TrainingUIAPIConfig) -> None:
    _ensure_editor_clone(config)
    status = _git(config, "status", "--porcelain").stdout.strip()
    if status:
        raise DatasetEditorGitError("Editor-клон содержит незавершённые изменения")
    _fetch_editor_clone(config)
    _git(
        config,
        "merge",
        "--ff-only",
        f"origin/{config.mlmarkup_editor_branch}",
    )
    _mark_editor_clone_synchronized(config)


def _synchronize_editor_clone_if_stale(config: TrainingUIAPIConfig) -> None:
    """Обновить клон не чаще TTL; polling и чтение сцен Git не запускают."""

    _ensure_editor_clone(config)
    stamp_path = _editor_sync_stamp_path(config)
    try:
        age_seconds = time.time() - stamp_path.stat().st_mtime
    except OSError:
        age_seconds = _EDITOR_SYNC_TTL_SECONDS
    if 0 <= age_seconds < _EDITOR_SYNC_TTL_SECONDS:
        return
    try:
        _synchronize_editor_clone(config)
    finally:
        _restore_editor_clone_ownership(config)


def _editor_sync_stamp_path(config: TrainingUIAPIConfig) -> Path:
    return config.mlmarkup_editor_root.parent / ".mlmarkup-editor-sync"


def _mark_editor_clone_synchronized(config: TrainingUIAPIConfig) -> None:
    path = _editor_sync_stamp_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(f"{time.time_ns()}\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fetch_editor_clone(config: TrainingUIAPIConfig) -> None:
    _ensure_editor_clone(config)
    _git(config, "fetch", "--prune", "origin", config.mlmarkup_editor_branch)


def _ensure_editor_clone(config: TrainingUIAPIConfig) -> None:
    root = config.mlmarkup_editor_root
    if not root.is_dir() or not (root / ".git").exists():
        raise DatasetEditorGitError(f"Editor-клон MLMarkup не найден: {root}")
    if not config.mlmarkup_editor_branch.strip():
        raise DatasetEditorGitError("Не задана ветка editor-клона")


def _git(
    config: TrainingUIAPIConfig,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    result = _git_optional(config, *arguments)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DatasetEditorGitError(
            f"Git-операция {' '.join(arguments[:2])} завершилась ошибкой: {detail}"
        )
    return result


def _git_optional(
    config: TrainingUIAPIConfig,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={config.mlmarkup_editor_root.resolve()}",
                "-C",
                str(config.mlmarkup_editor_root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        raise DatasetEditorGitError(f"Не удалось запустить Git: {exc}") from exc


def _commit(config: TrainingUIAPIConfig, subject: str, username: str) -> str:
    safe_username = " ".join(username.replace("\r", " ").replace("\n", " ").split())
    author_name = safe_username or "unknown"
    author_email = (
        "dataset-editor+"
        f"{hashlib.sha256(author_name.encode('utf-8')).hexdigest()[:16]}"
        "@mlsystem2.local"
    )
    _git(
        config,
        "-c",
        f"user.name={author_name}",
        "-c",
        f"user.email={author_email}",
        "commit",
        "--author",
        f"{author_name} <{author_email}>",
        "-m",
        subject,
        "-m",
        f"MLSystem2-User: {author_name}",
    )
    return _git(config, "rev-parse", "HEAD").stdout.strip()


def _push_with_retry(
    config: TrainingUIAPIConfig,
    *,
    expected_revisions: dict[PurePosixPath, str | None],
    expected_tree_revisions: dict[PurePosixPath, str | None] | None = None,
) -> str:
    branch = config.mlmarkup_editor_branch
    first = _git_optional(config, "push", "origin", f"HEAD:{branch}")
    if first.returncode == 0:
        return _git(config, "rev-parse", "HEAD").stdout.strip()
    _fetch_editor_clone(config)
    remote_ref = f"origin/{branch}"
    changed = [
        path.as_posix()
        for path, expected in expected_revisions.items()
        if _blob_revision(config, remote_ref, path) != expected
    ]
    tree_changes = [
        path.as_posix()
        for path, expected in (expected_tree_revisions or {}).items()
        if _tree_object_revision(config, remote_ref, path) != expected
    ]
    changed.extend(tree_changes)
    if changed:
        _discard_local_commit(config)
        raise DatasetEditorConflict(
            "Целевая разметка изменилась во время сохранения: " + ", ".join(changed)
        )
    rebase = _git_optional(config, "rebase", remote_ref)
    if rebase.returncode != 0:
        _git_optional(config, "rebase", "--abort")
        _discard_local_commit(config)
        raise DatasetEditorGitError("Не удалось перебазировать editor-коммит на origin")
    retry = _git_optional(config, "push", "origin", f"HEAD:{branch}")
    if retry.returncode != 0:
        detail = (retry.stderr or retry.stdout).strip()
        _discard_local_commit(config)
        raise DatasetEditorGitError(f"Не удалось отправить editor-коммит: {detail}")
    return _git(config, "rev-parse", "HEAD").stdout.strip()


def _discard_local_commit(config: TrainingUIAPIConfig) -> None:
    branch = config.mlmarkup_editor_branch
    remote_ref = f"origin/{branch}"
    _git(config, "switch", "--detach", remote_ref)
    _git(config, "branch", "--force", branch, remote_ref)
    _git(config, "switch", branch)


def _blob_revision(
    config: TrainingUIAPIConfig,
    ref: str,
    relative_path: PurePosixPath,
) -> str | None:
    result = _git_optional(config, "rev-parse", f"{ref}:{relative_path.as_posix()}")
    return result.stdout.strip() if result.returncode == 0 else None


def _blob_revisions(
    config: TrainingUIAPIConfig,
    ref: str,
    relative_directory: PurePosixPath,
) -> dict[PurePosixPath, str]:
    """Получить SHA всех blob каталога одним процессом Git."""

    result = _git(
        config,
        "ls-tree",
        "-rz",
        ref,
        "--",
        relative_directory.as_posix(),
    )
    revisions: dict[PurePosixPath, str] = {}
    for line in result.stdout.split("\0"):
        header, separator, raw_path = line.partition("\t")
        parts = header.split()
        if not separator or len(parts) != 3 or parts[1] != "blob":
            continue
        revisions[PurePosixPath(raw_path)] = parts[2]
    return revisions


def _tree_object_revision(
    config: TrainingUIAPIConfig,
    ref: str,
    relative_path: PurePosixPath,
) -> str | None:
    result = _git_optional(config, "rev-parse", f"{ref}:{relative_path.as_posix()}")
    return result.stdout.strip() if result.returncode == 0 else None


def _publication_status(
    config: TrainingUIAPIConfig,
    commit: str,
) -> str:
    live_commit = _live_commit(config)
    if live_commit is None:
        return "publishing"
    result = _git_optional(config, "merge-base", "--is-ancestor", commit, live_commit)
    return "published" if result.returncode == 0 else "publishing"


def _live_commit(config: TrainingUIAPIConfig) -> str | None:
    try:
        value = config.mlmarkup_release_marker.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, IndexError):
        return None
    return value if _SHA_PATTERN.fullmatch(value) is not None else None


__all__ = [
    "DatasetEditorConflict",
    "DatasetEditorGitError",
    "add_editor_scenes",
    "browse_editor_rasters",
    "copy_editor_dataset",
    "delete_editor_dataset",
    "delete_editor_scene",
    "editor_publication_info",
    "editor_scene_detail",
    "list_editor_datasets",
    "list_editor_scenes",
    "publish_editor_scenes",
    "preview_editor_dataset_rebuild",
    "rebuild_editor_dataset",
    "resolve_editor_raster",
    "save_editor_scene",
]

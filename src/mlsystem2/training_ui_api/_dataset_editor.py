"""Git-backed редактор per-image датасетов MLMarkup."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote

import rasterio
from affine import Affine
from pyproj import CRS as PyprojCRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import per_image_annotation_name, resolve_scene_images
from mlsystem2.dataset_preparing.contracts import SceneImageResolutionRequest

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_dataset, list_managed_datasets
from ._datasets import RASTER_SUFFIXES
from .contracts import (
    DatasetEditorDatasetInfo,
    DatasetEditorDatasetListResponse,
    DatasetEditorMutationResult,
    DatasetEditorPublicationInfo,
    DatasetEditorRasterBrowserResponse,
    DatasetEditorRasterFolderInfo,
    DatasetEditorRasterInfo,
    DatasetEditorSceneDetail,
    DatasetEditorSceneInfo,
    DatasetEditorSceneListResponse,
    DatasetInfo,
    TrainingUIAPIError,
)


_ROLE_PROPERTY = "_mlsystem2_role"
_ROLES = {"positive", "hard_negative"}
_SHA_PATTERN = re.compile(r"[0-9a-fA-F]{40,64}")
_SERVICE_AUTHOR_NAME = "MLSystem2 Dataset Editor"
_SERVICE_AUTHOR_EMAIL = "mlsystem2-dataset-editor@localhost"
_VALID_FOOTPRINT_MAX_SIDE = 4096
_VALID_FOOTPRINT_SIMPLIFY_CELLS = 0.75


class DatasetEditorConflict(RuntimeError):
    """Blob revision устарела относительно origin."""


class DatasetEditorGitError(RuntimeError):
    """Git-клон редактора недоступен или операция Git завершилась ошибкой."""


def list_editor_datasets(
    session: Session,
    config: TrainingUIAPIConfig,
) -> DatasetEditorDatasetListResponse:
    with _editor_lock(config):
        _synchronize_editor_clone(config)
        result: list[DatasetEditorDatasetInfo] = []
        for dataset in list_managed_datasets(session, config, include_custom=False):
            try:
                source_dir = _editor_source_dir(config, dataset)
            except TrainingUIAPIError:
                continue
            if source_dir.exists() and not source_dir.is_dir():
                continue
            if source_dir.is_dir() and _direct_files(source_dir, ".txt"):
                continue
            geojson_files = _direct_files(source_dir, ".geojson")
            has_dataset_subdirectories = source_dir.is_dir() and any(
                item.is_dir() and not item.name.startswith(".") for item in source_dir.iterdir()
            )
            if not geojson_files and has_dataset_subdirectories:
                continue
            if dataset.annotations_dir is None and not geojson_files:
                continue
            result.append(_editor_dataset_info(dataset, len(geojson_files)))
        result.sort(key=lambda item: (item.class_name.casefold(), item.dataset_name.casefold()))
        return DatasetEditorDatasetListResponse(datasets=result)


def list_editor_scenes(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetEditorSceneListResponse:
    with _editor_lock(config):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(
            session,
            config,
            dataset_key,
            allow_missing=True,
        )
        scenes = _scene_infos(config, dataset, source_dir)
        return DatasetEditorSceneListResponse(
            dataset=_editor_dataset_info(dataset, len(scenes)),
            scenes=scenes,
        )


def editor_scene_detail(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    annotation_name: str,
) -> DatasetEditorSceneDetail:
    with _editor_lock(config):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        scene = _scene_by_annotation(config, dataset, source_dir, annotation_name)
        annotation_path = _annotation_path(source_dir, annotation_name)
        image_path = _matched_image_path(dataset, source_dir, annotation_name)
        footprint = _valid_data_footprint(image_path)
        return DatasetEditorSceneDetail(
            scene=scene,
            geojson=_clip_geojson_to_footprint(_read_geojson(annotation_path), footprint),
            valid_data_footprint=dict(mapping(footprint)),
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
    with _editor_lock(config):
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
                for path in sorted(
                    selected_folder.iterdir(), key=lambda item: item.name.casefold()
                )
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
        existing = {path.name.casefold() for path in _direct_files(source_dir, ".geojson")}
        collisions = sorted(name for key, (name, _path) in names.items() if key in existing)
        if collisions:
            raise DatasetEditorConflict(
                "Снимки уже добавлены в датасет: " + ", ".join(collisions)
            )
        source_dir.mkdir(parents=True, exist_ok=True)
        created: list[Path] = []
        relative_files: list[PurePosixPath] = []
        try:
            for annotation_name, image_path in names.values():
                payload = _empty_annotation_payload(image_path)
                target = source_dir / annotation_name
                _write_geojson_atomic(target, payload)
                created.append(target)
                relative_files.append(_repo_relative(config, target))
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
    username: str,
) -> DatasetEditorMutationResult:
    if not scenes:
        raise TrainingUIAPIError("Для публикации нужен хотя бы один снимок")
    normalized_names = [name.casefold() for name, _revision, _geojson in scenes]
    if len(normalized_names) != len(set(normalized_names)):
        raise TrainingUIAPIError("Список публикации содержит повторяющиеся снимки")

    with _editor_lock(config):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        resolved: list[tuple[str, str, dict[str, Any], Path, PurePosixPath]] = []
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
        if conflicts:
            raise DatasetEditorConflict(
                "Разметка уже изменена другим пользователем: " + ", ".join(conflicts)
            )

        prepared: list[tuple[str, str, dict[str, Any], Path, PurePosixPath]] = []
        for annotation_name, revision, geojson, annotation_path, relative_path in resolved:
            image_path = _matched_image_path(dataset, source_dir, annotation_name)
            _validate_editor_geojson(geojson, image_path)
            previous_payload = _read_geojson(annotation_path)
            _validate_preserved_properties(previous_payload, geojson)
            prepared.append(
                (annotation_name, revision, geojson, annotation_path, relative_path)
            )

        relative_paths = [item[4] for item in prepared]
        try:
            for _name, _revision, geojson, annotation_path, _relative_path in prepared:
                _write_geojson_atomic(annotation_path, geojson)
            _git(config, "add", "--", *(path.as_posix() for path in relative_paths))
            subject = (
                f"Обновить разметку {prepared[0][0]}"
                if len(prepared) == 1
                else (
                    f"Обновить разметку датасета {dataset.dataset_name or dataset.name} "
                    f"({len(prepared)} снимка)"
                )
            )
            commit = _commit(config, subject, username)
            commit = _push_with_retry(
                config,
                expected_revisions={item[4]: item[1] for item in prepared},
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                *(path.as_posix() for path in relative_paths),
            )
            raise
        updated_scenes = {
            item.annotation_name: item for item in _scene_infos(config, dataset, source_dir)
        }
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
) -> DatasetEditorMutationResult:
    with _editor_lock(config):
        _synchronize_editor_clone(config)
        dataset, source_dir = _editor_dataset_context(session, config, dataset_key)
        annotation_path = _annotation_path(source_dir, annotation_name)
        relative_path = _repo_relative(config, annotation_path)
        current_revision = _blob_revision(config, "HEAD", relative_path)
        if current_revision != revision:
            raise DatasetEditorConflict("Разметка уже изменена другим пользователем")
        _git(config, "rm", "--", relative_path.as_posix())
        try:
            commit = _commit(config, f"Удалить снимок {annotation_name}", username)
            commit = _push_with_retry(
                config,
                expected_revisions={relative_path: revision},
            )
        except Exception:
            _git_optional(
                config,
                "restore",
                "--staged",
                "--worktree",
                "--",
                relative_path.as_posix(),
            )
            raise
        return DatasetEditorMutationResult(
            commit=commit,
            publication_status=_publication_status(config, commit),
        )


def editor_publication_info(
    config: TrainingUIAPIConfig,
    commit: str,
) -> DatasetEditorPublicationInfo:
    if _SHA_PATTERN.fullmatch(commit) is None:
        raise TrainingUIAPIError("Некорректный SHA коммита")
    with _editor_lock(config):
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


def _managed_editor_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetInfo:
    dataset = find_managed_dataset(session, config, dataset_key)
    if dataset is None or dataset.is_custom or dataset.source_path is None:
        raise TrainingUIAPIError(f"Датасет редактора не найден: {dataset_key}")
    if dataset.images_dir is None or dataset.imagery_type is None:
        raise TrainingUIAPIError("Для датасета не настроен каталог снимков")
    return dataset


def _editor_dataset_context(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
    *,
    allow_missing: bool = False,
) -> tuple[DatasetInfo, Path]:
    dataset = _managed_editor_dataset(session, config, dataset_key)
    source_dir = _editor_source_dir(config, dataset)
    if source_dir.exists() and not source_dir.is_dir():
        raise TrainingUIAPIError("Источник датасета не является папкой")
    if not source_dir.exists() and not allow_missing:
        raise TrainingUIAPIError("Источник датасета отсутствует в editor-клоне")
    if source_dir.is_dir() and _direct_files(source_dir, ".txt"):
        raise TrainingUIAPIError("Редактор поддерживает только per-image датасеты без TXT")
    return dataset, source_dir


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


def _editor_dataset_info(dataset: DatasetInfo, scene_count: int) -> DatasetEditorDatasetInfo:
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
    )


def _scene_infos(
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    source_dir: Path,
) -> list[DatasetEditorSceneInfo]:
    if not source_dir.is_dir():
        return []
    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(_dataset_images_root(dataset)),
            annotations_dir=str(source_dir),
        )
    )
    if resolution.missing_scenes:
        raise TrainingUIAPIError(
            "Для GeoJSON не найдены TIFF: " + ", ".join(resolution.missing_scenes)
        )
    if resolution.ambiguous_scenes:
        raise TrainingUIAPIError(
            "Имена GeoJSON неоднозначно сопоставлены с TIFF: "
            + ", ".join(sorted(resolution.ambiguous_scenes))
        )
    root = _dataset_images_root(dataset)
    result: list[DatasetEditorSceneInfo] = []
    for item in resolution.images:
        annotation_path = Path(item.annotation_file or "")
        relative_annotation = _repo_relative(config, annotation_path)
        revision = _blob_revision(config, "HEAD", relative_annotation)
        if revision is None:
            raise DatasetEditorGitError(
                f"GeoJSON не зафиксирован в Git: {annotation_path.name}"
            )
        positive, hard_negative = _role_counts(_read_geojson(annotation_path))
        image_path = Path(item.image_path).resolve()
        image_relative = image_path.relative_to(root).as_posix()
        result.append(
            DatasetEditorSceneInfo(
                scene_id=item.scene_id,
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
            )
        )
    result.sort(key=lambda item: item.scene_id.casefold())
    return result


def _scene_by_annotation(
    config: TrainingUIAPIConfig,
    dataset: DatasetInfo,
    source_dir: Path,
    annotation_name: str,
) -> DatasetEditorSceneInfo:
    safe_name = _safe_annotation_name(annotation_name)
    scene = next(
        (item for item in _scene_infos(config, dataset, source_dir) if item.annotation_name == safe_name),
        None,
    )
    if scene is None:
        raise TrainingUIAPIError(f"Снимок датасета не найден: {safe_name}")
    return scene


def _matched_image_path(
    dataset: DatasetInfo,
    source_dir: Path,
    annotation_name: str,
) -> Path:
    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(_dataset_images_root(dataset)),
            annotations_dir=str(source_dir),
        )
    )
    safe_name = _safe_annotation_name(annotation_name)
    item = next(
        (
            candidate
            for candidate in resolution.images
            if Path(candidate.annotation_file or "").name == safe_name
        ),
        None,
    )
    if item is None:
        raise TrainingUIAPIError(f"Для GeoJSON не найден TIFF: {safe_name}")
    return Path(item.image_path)


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
    if name != value or Path(name).suffix.casefold() != ".geojson":
        raise TrainingUIAPIError("Некорректное имя GeoJSON")
    return name


def _empty_annotation_payload(image_path: Path) -> dict[str, Any]:
    try:
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise TrainingUIAPIError(f"У TIFF отсутствует CRS: {image_path.name}")
            crs_name = source.crs.to_string()
    except rasterio.errors.RasterioError as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF {image_path.name}: {exc}") from exc
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs_name}},
        "features": [],
    }


def _validate_editor_geojson(payload: dict[str, Any], image_path: Path) -> None:
    if payload.get("type") != "FeatureCollection" or not isinstance(
        payload.get("features"), list
    ):
        raise TrainingUIAPIError("GeoJSON должен быть FeatureCollection со списком features")
    geojson_crs = _geojson_crs(payload)
    try:
        with rasterio.open(image_path) as source:
            if source.crs is None:
                raise TrainingUIAPIError("У TIFF отсутствует CRS")
            raster_crs = PyprojCRS.from_user_input(source.crs)
    except rasterio.errors.RasterioError as exc:
        raise TrainingUIAPIError(f"Не удалось открыть TIFF: {exc}") from exc
    footprint = _valid_data_footprint(image_path)
    if geojson_crs != raster_crs:
        raise TrainingUIAPIError(
            f"CRS GeoJSON ({geojson_crs.to_string()}) не совпадает с CRS TIFF "
            f"({raster_crs.to_string()})"
        )
    for index, feature in enumerate(payload["features"], start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise TrainingUIAPIError(f"Объект {index} не является GeoJSON Feature")
        properties = feature.get("properties")
        if not isinstance(properties, dict) or properties.get(_ROLE_PROPERTY) not in _ROLES:
            raise TrainingUIAPIError(
                f"У объекта {index} должна быть явная роль positive или hard_negative"
            )
        try:
            geometry = shape(feature.get("geometry"))
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(f"Некорректная геометрия объекта {index}: {exc}") from exc
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            raise TrainingUIAPIError(f"Объект {index} должен быть Polygon или MultiPolygon")
        if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
            raise TrainingUIAPIError(f"Геометрия объекта {index} пуста или невалидна")
        if not footprint.covers(geometry):
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
            valid_mask = source.dataset_mask(
                out_shape=(sample_height, sample_width),
                resampling=Resampling.nearest,
            ) > 0
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
                tolerance = max(
                    abs(mask_transform.a),
                    abs(mask_transform.b),
                    abs(mask_transform.d),
                    abs(mask_transform.e),
                ) * _VALID_FOOTPRINT_SIMPLIFY_CELLS
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
            geometry = _polygonal_geometry(geometry.intersection(footprint))
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(f"Не удалось обрезать геометрию объекта {index}: {exc}") from exc
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
            if _non_system_properties(feature) != _non_system_properties(
                previous_by_id[identity]
            ):
                raise TrainingUIAPIError("Существующие свойства объектов изменять нельзя")
        properties = _non_system_properties(feature)
        if properties != "{}" and properties not in previous_property_sets:
            raise TrainingUIAPIError("Редактор не поддерживает произвольные атрибуты объектов")


def _non_system_properties(feature: dict[str, Any]) -> str:
    properties = feature.get("properties")
    cleaned = (
        {key: value for key, value in properties.items() if key != _ROLE_PROPERTY}
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
        role = properties.get(_ROLE_PROPERTY, "positive") if isinstance(properties, dict) else "positive"
        if role == "positive":
            positive += 1
        elif role == "hard_negative":
            hard_negative += 1
        else:
            raise TrainingUIAPIError(f"Неизвестная роль объекта: {role}")
    return positive, hard_negative


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
def _editor_lock(config: TrainingUIAPIConfig) -> Iterator[None]:
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
    _git(
        config,
        "-c",
        f"user.name={_SERVICE_AUTHOR_NAME}",
        "-c",
        f"user.email={_SERVICE_AUTHOR_EMAIL}",
        "commit",
        "-m",
        subject,
        "-m",
        f"MLSystem2-User: {safe_username or 'unknown'}",
    )
    return _git(config, "rev-parse", "HEAD").stdout.strip()


def _push_with_retry(
    config: TrainingUIAPIConfig,
    *,
    expected_revisions: dict[PurePosixPath, str | None],
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
    "delete_editor_scene",
    "editor_publication_info",
    "editor_scene_detail",
    "list_editor_datasets",
    "list_editor_scenes",
    "publish_editor_scenes",
    "resolve_editor_raster",
    "save_editor_scene",
]

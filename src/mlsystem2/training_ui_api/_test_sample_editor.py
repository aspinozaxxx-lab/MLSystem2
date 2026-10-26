"""Хранилище тестовых разметок для общего редактора датасетов, без изменения MLMarkup."""

from __future__ import annotations

import copy
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import rasterize
from shapely.geometry import mapping, shape
from sqlalchemy import select

from ._dataset_catalog import dataset_class_row, find_managed_dataset
from ._dataset_editor import (
    DatasetEditorConflict, DatasetEditorImportClippingRequired,
    _clip_geojson_to_footprint, _dataset_images_root, _draft_info, _draft_summary, _empty_annotation_payload,
    _file_revision, _footprint_covers_geometry, _image_path_for_annotation,
    _normalize_editor_geojson, _read_geojson, _safe_annotation_name, _safe_raster_path,
    _reproject_editor_geojson,
    _safe_relative_directory, _validate_editor_geojson, _validate_preserved_properties,
    _write_geojson_atomic,
)
from ._datasets import imagery_images_dir
from ._markup_export import _write_png
from ._markup_lock import test_sample_storage_lock
from ._models import DatasetEditorDraftRow, TestSampleRow, TestSampleTileRow
from ._raster_valid_data import editor_clipping_footprint
from ._test_samples import (
    _clear_test_sample_tile_f1, _mark_test_sample_evaluation_stale,
    _refresh_training_metrics_after_primary_change, _sample_root, _sample_row, _utc_now,
)
from .contracts import (
    DatasetEditorDatasetInfo, DatasetEditorDatasetListResponse, DatasetEditorMutationResult,
    DatasetEditorRasterBrowserResponse, DatasetEditorRasterFolderInfo, DatasetEditorRasterInfo,
    DatasetEditorSceneDetail, DatasetEditorSceneInfo, DatasetEditorSceneListResponse,
    DatasetInfo, DatasetObjectType, ImageryType, TrainingUIAPIError,
)


def _draft_key(sample_id):
    return f"test-sample:{sample_id}"


def _dataset(session, config, sample):
    source = find_managed_dataset(session, config, sample.dataset_key)
    class_row = dataset_class_row(session, sample.class_key)
    imagery = source.imagery_type if source else (class_row.imagery_type if class_row else None)
    if imagery is None:
        raise TrainingUIAPIError("Для класса тестовой разметки не указан тип снимков.")
    imagery = ImageryType(imagery)
    return DatasetInfo(
        key=_draft_key(sample.id), name=sample.name, dataset_name=sample.name,
        class_key=sample.class_key, class_name=sample.class_name,
        imagery_type=imagery, images_dir=str(imagery_images_dir(config.images_root, imagery.value)),
        task=sample.task, object_types=[DatasetObjectType.model_validate(item) for item in sample.class_schema or []],
    )


def _drafts(session, sample_id, username):
    return list(session.scalars(select(DatasetEditorDraftRow).where(
        DatasetEditorDraftRow.dataset_key == _draft_key(sample_id),
        DatasetEditorDraftRow.username == username,
    )).all())


def _tile_name(tile):
    return f"tile_{tile.tile_index:03d}.geojson"


def _tile(sample, name):
    safe = _safe_annotation_name(name)
    return next((tile for tile in sample.tiles if _tile_name(tile) == safe), None)


def sample_editor_raster(session, config, sample_id, name, username):
    sample = _sample_row(session, sample_id)
    tile = _tile(sample, name)
    if tile is not None:
        path = _sample_root(config, sample.id) / Path(_tile_name(tile)).with_suffix(".tif")
        if not path.is_file():
            raise TrainingUIAPIError("Снимок тестовой разметки недоступен.")
        return path
    if any(d.annotation_name == name and d.base_revision == "new" for d in _drafts(session, sample_id, username)):
        return _image_path_for_annotation(_dataset(session, config, sample), name)
    raise TrainingUIAPIError("Снимок не входит в эту тестовую разметку или ваши черновики.")


def sample_editor_scenes(session, config, sample_id, username):
    with test_sample_storage_lock(config, sample_id):
        sample = _sample_row(session, sample_id)
        dataset = _dataset(session, config, sample)
        root = _sample_root(config, sample_id)
        drafts = {draft.annotation_name: draft for draft in _drafts(session, sample_id, username)}
        scenes = []
        for tile in sample.tiles:
            name = _tile_name(tile)
            revision = _file_revision(root / name)
            draft = drafts.pop(name, None)
            scenes.append(DatasetEditorSceneInfo(
                scene_id=name, annotation_name=name, image_name=tile.source_name,
                raster_url=f"/api/v1/test-samples/{sample_id}/editor/rasters/{name}",
                revision=revision or "missing", total_count=tile.object_count,
                positive_count=tile.object_count, hard_negative_count=0,
                class_counts=dict(tile.class_object_counts or {}),
                draft=_draft_summary(draft, dataset, revision) if draft else None,
            ))
        for name, draft in drafts.items():
            scenes.append(DatasetEditorSceneInfo(
                scene_id=name, annotation_name=name, image_name=name,
                raster_url=f"/api/v1/test-samples/{sample_id}/editor/rasters/{name}",
                revision="new", total_count=0, positive_count=0, hard_negative_count=0,
                draft=_draft_summary(draft, dataset, "new"),
            ))
        info = DatasetEditorDatasetInfo(
            key=dataset.key, name=sample.name, dataset_name=sample.name,
            class_key=sample.class_key, class_name=sample.class_name,
            imagery_type=dataset.imagery_type.value, scene_count=len(scenes),
            task=sample.task, object_types=sample.class_schema or [], source_status="current",
        )
        return DatasetEditorSceneListResponse(dataset=info, scenes=scenes)


def sample_editor_catalog(session, config, sample_id, username):
    return DatasetEditorDatasetListResponse(datasets=[sample_editor_scenes(session, config, sample_id, username).dataset])


def sample_editor_detail(session, config, sample_id, name, username):
    with test_sample_storage_lock(config, sample_id):
        sample = _sample_row(session, sample_id)
        dataset = _dataset(session, config, sample)
        scenes = sample_editor_scenes(session, config, sample_id, username).scenes
        scene = next((scene for scene in scenes if scene.annotation_name == name), None)
        if scene is None:
            raise TrainingUIAPIError("Снимок тестовой разметки не найден.")
        image = sample_editor_raster(session, config, sample_id, name, username)
        payload = (_read_geojson(_sample_root(config, sample_id) / name)
                   if _tile(sample, name) else _empty_annotation_payload(image, dataset))
        footprint = editor_clipping_footprint(image)
        draft = next((draft for draft in _drafts(session, sample_id, username) if draft.annotation_name == name), None)
        return DatasetEditorSceneDetail(
            scene=scene, geojson=payload, valid_data_footprint=dict(mapping(footprint)),
            draft=_draft_info(draft, dataset, scene.revision) if draft else None,
        )


def _normalize(payload, previous, dataset, image, *, clip=True, allow_invalid=False):
    candidate = copy.deepcopy(payload)
    if dataset.task == "binary" and candidate.get("_mlsystem2_task", "binary") != "binary":
        raise TrainingUIAPIError("Многоклассовая разметка не подходит бинарной тестовой выборке.")
    for feature in candidate.get("features", []):
        if isinstance(feature, dict):
            if feature.get("properties") is None:
                feature["properties"] = {}
            properties = feature.setdefault("properties", {})
            if not isinstance(properties, dict):
                raise TrainingUIAPIError("Свойства объекта должны быть JSON-объектом.")
            properties.setdefault("_mlsystem2_role", "positive")
            if properties["_mlsystem2_role"] != "positive":
                raise TrainingUIAPIError("В тестовой разметке допустимы только эталонные объекты, без отрицательных примеров и зон обучения.")
    identifiers = [str(feature["id"]) for feature in candidate.get("features", []) if isinstance(feature, dict) and "id" in feature]
    if len(set(identifiers)) != len(identifiers):
        raise TrainingUIAPIError("ID объектов не должны повторяться.")
    candidate = _reproject_editor_geojson(candidate, image)
    normalized = _normalize_editor_geojson(candidate, previous, dataset, image_path=image)
    if clip:
        normalized = _clip_geojson_to_footprint(normalized, editor_clipping_footprint(image))
    _validate_editor_geojson(normalized, image, dataset,
                             check_footprint=clip and not allow_invalid, allow_repair=allow_invalid)
    return normalized


def sample_editor_save_draft(session, config, sample_id, name, request, username):
    with test_sample_storage_lock(config, sample_id):
        detail = sample_editor_detail(session, config, sample_id, name, username)
        if detail.scene.revision != request.base_revision:
            raise DatasetEditorConflict("Разметка изменилась. Обновите снимок перед сохранением.")
        sample = _sample_row(session, sample_id)
        dataset = _dataset(session, config, sample)
        image = sample_editor_raster(session, config, sample_id, name, username)
        normalized = _normalize(request.geojson, detail.geojson, dataset, image, allow_invalid=True)
        _validate_preserved_properties(detail.geojson, normalized)
        row = next((draft for draft in _drafts(session, sample_id, username) if draft.annotation_name == name), None)
        if row is None:
            row = DatasetEditorDraftRow(dataset_key=_draft_key(sample_id), annotation_name=name, username=username)
            session.add(row)
        row.base_revision = request.base_revision
        row.geojson = normalized
        row.deleted = request.deleted
        row.updated_at = _utc_now()
        session.flush()
        return _draft_info(row, dataset, detail.scene.revision)


def sample_editor_discard(session, sample_id, username, name=None):
    _sample_row(session, sample_id)
    rows = [row for row in _drafts(session, sample_id, username) if name is None or row.annotation_name == name]
    for row in rows:
        session.delete(row)
    session.flush()
    return {"deleted_count": len(rows)}


def sample_editor_browse(session, config, sample_id, folder):
    sample = _sample_row(session, sample_id)
    dataset = _dataset(session, config, sample)
    added = {tile.source_name: _tile_name(tile) for tile in sample.tiles}
    relative, directory = _safe_relative_directory(_dataset_images_root(dataset), folder)
    from mlsystem2.dataset_preparing.api import per_image_annotation_name

    return DatasetEditorRasterBrowserResponse(
        folder=relative.as_posix() if relative.parts else "", parent=("" if relative.parent.as_posix() == "." else relative.parent.as_posix()) if relative.parts else None,
        folders=[DatasetEditorRasterFolderInfo(name=p.name, path=(relative / p.name).as_posix())
                 for p in sorted(directory.iterdir()) if p.is_dir() and not p.name.startswith(".")],
        rasters=[DatasetEditorRasterInfo(name=p.name, path=(relative / p.name).as_posix(),
                 annotation_name=added.get((relative / p.name).as_posix(), per_image_annotation_name(p)), size_bytes=p.stat().st_size)
                 for p in sorted(directory.iterdir()) if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}],
    )


def sample_editor_import(session, config, sample_id, request, username):
    if len(request.model_dump_json().encode("utf-8")) > 50 * 1024 * 1024:
        raise TrainingUIAPIError("Общий размер GeoJSON не должен превышать 50 МиБ.")
    sample = _sample_row(session, sample_id)
    dataset = _dataset(session, config, sample)
    prepared = []
    clipping = []
    for scene in request.scenes:
        name = _safe_annotation_name(scene.annotation_name)
        tile = _tile(sample, name)
        image = (_sample_root(config, sample_id) / Path(name).with_suffix(".tif")
                 if tile else _image_path_for_annotation(dataset, name))
        if not tile:
            relative = image.relative_to(_dataset_images_root(dataset)).as_posix()
            existing = next((tile for tile in sample.tiles if tile.source_name == relative), None)
            if existing:
                name = _tile_name(existing)
                tile = existing
                image = _sample_root(config, sample_id) / Path(name).with_suffix(".tif")
        previous = _read_geojson(_sample_root(config, sample_id) / name) if tile else _empty_annotation_payload(image, dataset)
        candidate = _normalize(scene.geojson, previous, dataset, image, clip=False)
        outside = sum(not _footprint_covers_geometry(editor_clipping_footprint(image), shape(feature["geometry"]))
                      for feature in candidate["features"])
        if outside:
            clipping.append(f"{name}: объектов за контуром — {outside}.")
        if request.clip_to_footprint:
            candidate = _normalize(candidate, previous, dataset, image)
        revision = _file_revision(_sample_root(config, sample_id) / name) if tile else "new"
        prepared.append((name, revision, candidate))
    if clipping and not request.clip_to_footprint:
        raise DatasetEditorImportClippingRequired("\n".join(clipping))
    if len({name for name, _, _ in prepared}) != len(prepared):
        raise TrainingUIAPIError("Несколько файлов относятся к одному снимку.")
    own = {draft.annotation_name: draft for draft in _drafts(session, sample_id, username)}
    if any(name in own for name, _, _ in prepared):
        raise DatasetEditorConflict("Сначала опубликуйте или отмените существующий черновик снимка.")
    for name, revision, payload in prepared:
        session.add(DatasetEditorDraftRow(dataset_key=_draft_key(sample_id), annotation_name=name,
                    username=username, base_revision=revision, geojson=payload, deleted=False))
    session.flush()
    return sample_editor_scenes(session, config, sample_id, username)


def _copy_image(source_path, target_path, imagery):
    with rasterio.open(source_path) as source:
        count = 4 if imagery == ImageryType.KANOPUS else 3
        alpha = source.count == 4 and source.colorinterp[-1] == ColorInterp.alpha
        if count == 4 and ColorInterp.alpha in source.colorinterp:
            raise TrainingUIAPIError("Канопусу нужен канал NIR, а не канал прозрачности ортофото.")
        if source.crs is None or source.crs.to_epsg() != 3857:
            raise TrainingUIAPIError("Для тестовой разметки нужен подготовленный TIFF в EPSG:3857.")
        if any(dtype != "uint8" for dtype in source.dtypes) or not (source.count == count or count == 3 and alpha):
            raise TrainingUIAPIError("TIFF должен содержать uint8: 4 канала Канопуса или RGB ортофото (допустима прозрачность).")
        profile = source.profile.copy()
        profile.update(driver="GTiff", count=count, tiled=True, blockxsize=256, blockysize=256,
                       compress="deflate", BIGTIFF="IF_SAFER", ALPHA="NO")
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(target_path, "w", **profile) as output:
            for _, window in source.block_windows(1):
                output.write(source.read(list(range(1, count + 1)), window=window), window=window)
                mask = source.dataset_mask(window=window)
                if alpha:
                    mask = np.minimum(mask, source.read(4, window=window))
                output.write_mask(mask, window=window)
            factors = [factor for factor in (2, 4, 8, 16) if min(source.shape) // factor >= 1]
            if factors:
                output.build_overviews(factors, Resampling.average)
            output.descriptions = source.descriptions[:count]
            output.colorinterp = source.colorinterp[:count]


def _write_mask(root, tile, payload, schema):
    path = root / f"tile_{tile.tile_index:03d}_mask.png"
    path.unlink(missing_ok=True)
    ids = {item["slug"]: item["id"] for item in schema}
    with rasterio.open(root / f"tile_{tile.tile_index:03d}.tif") as raster:
        shapes = [(feature["geometry"], ids.get(feature.get("properties", {}).get("_mlsystem2_class"), 1))
                  for feature in payload["features"]]
        mask = (rasterize(shapes, out_shape=raster.shape, transform=raster.transform, dtype="uint8")
                if shapes else np.zeros(raster.shape, dtype="uint8"))
        mask[raster.dataset_mask() == 0] = 0
    _write_png(path, mask[None])
    # Производные изображения следующего запроса строятся по новой разметке.
    for preview in root.glob(f"tile_{tile.tile_index:03d}*"):
        if preview.suffix.lower() in {".png", ".jpg", ".json"} and preview != path:
            preview.unlink()


@contextmanager
def sample_editor_publish(session, config, sample_id, username, *, images=None):
    """Публиковать файлы и БД вместе; при ошибке вернуть прежнюю версию."""
    with test_sample_storage_lock(config, sample_id, write=True):
        session.execute(select(TestSampleRow).where(TestSampleRow.id == sample_id).with_for_update()
                        .execution_options(populate_existing=True))
        sample = _sample_row(session, sample_id)
        session.expire(sample, ["tiles"])
        dataset = _dataset(session, config, sample)
        root = _sample_root(config, sample_id)
        stage = root.with_name(f".editing-{sample_id}-{uuid.uuid4().hex}")
        backup = root.with_name(f".previous-{sample_id}-{uuid.uuid4().hex}")
        swapped = False
        committed = False
        try:
            shutil.copytree(root, stage, copy_function=os.link)
            changes = [(draft.annotation_name, draft.base_revision, draft.geojson, draft.deleted)
                       for draft in _drafts(session, sample_id, username)] if images is None else images
            next_index = max((tile.tile_index for tile in sample.tiles), default=0) + 1
            for name, revision, payload, deleted in changes:
                tile = _tile(sample, name)
                if tile:
                    if _file_revision(root / name) != revision:
                        raise DatasetEditorConflict("Разметка изменена другим пользователем. Обновите снимок.")
                    image = stage / Path(name).with_suffix(".tif")
                else:
                    if revision != "new":
                        raise DatasetEditorConflict("Снимок тестовой разметки удалён.")
                    source = _image_path_for_annotation(dataset, name)
                    relative = source.relative_to(_dataset_images_root(dataset)).as_posix()
                    if any(tile.source_name == relative for tile in sample.tiles):
                        raise DatasetEditorConflict("Снимок уже добавлен в тестовую разметку.")
                    if deleted:
                        continue
                    tile = TestSampleTileRow(tile_index=next_index, source_name=relative,
                        territory=source.parent.name, object_count=0, enabled=True)
                    next_index += 1
                    sample.tiles.append(tile)
                    image = stage / Path(_tile_name(tile)).with_suffix(".tif")
                    _copy_image(source, image, dataset.imagery_type)
                if deleted:
                    sample.tiles.remove(tile)
                    for path in stage.glob(f"tile_{tile.tile_index:03d}.*"):
                        path.unlink()
                    for path in stage.glob(f"tile_{tile.tile_index:03d}_*"):
                        path.unlink()
                    continue
                normalized = _normalize(payload, {}, dataset, image)
                _write_geojson_atomic(stage / _tile_name(tile), normalized)
                _write_mask(stage, tile, normalized, sample.class_schema or [])
                tile.object_count = len(normalized["features"])
                tile.class_object_counts = {item["slug"]: sum(
                    f["properties"].get("_mlsystem2_class") == item["slug"] for f in normalized["features"]
                ) for item in sample.class_schema or []}
                tile.updated_at = _utc_now()
                with rasterio.open(image) as raster:
                    sample.tile_width, sample.tile_height = raster.width, raster.height
            if changes:
                sample.image_count = len(sample.tiles)
                sample.actual_object_count = sum(tile.object_count for tile in sample.tiles)
                sample.territory_count = len({tile.territory for tile in sample.tiles})
                sample.content_revision += 1
                sample.updated_at = _utc_now()
                _clear_test_sample_tile_f1(sample)
                _mark_test_sample_evaluation_stale(session, sample, "Тестовая разметка отредактирована. Пересчитайте F1 вручную.")
                if sample.is_primary:
                    _refresh_training_metrics_after_primary_change(session, sample.class_key, config,
                        reason="Основная тестовая разметка отредактирована. Пересчитайте F1 вручную.")
                root.replace(backup)
                try:
                    stage.replace(root)
                except BaseException:
                    backup.replace(root)
                    raise
                swapped = True
            if images is None:
                sample_editor_discard(session, sample_id, username)
            session.flush()
            yield DatasetEditorMutationResult(commit=str(sample.content_revision), publication_status="published",
                scenes=sample_editor_scenes(session, config, sample_id, username).scenes)
            committed = True
        except BaseException:
            if swapped:
                shutil.rmtree(root)
                backup.replace(root)
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            if committed and backup.exists():
                shutil.rmtree(backup)


def sample_editor_add_images(session, config, sample_id, request):
    dataset = _dataset(session, config, _sample_row(session, sample_id))
    root = _dataset_images_root(dataset)
    from mlsystem2.dataset_preparing.api import per_image_annotation_name

    if request.folder_path is not None:
        _, directory = _safe_relative_directory(root, request.folder_path)
        paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}]
    else:
        paths = [_safe_raster_path(root, value) for value in request.image_paths]
    if not paths or len(paths) > 100:
        raise TrainingUIAPIError("Выберите от 1 до 100 снимков.")
    return [(per_image_annotation_name(path), "new", _empty_annotation_payload(path, dataset), False) for path in paths]

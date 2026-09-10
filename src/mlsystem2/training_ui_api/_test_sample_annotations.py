"""Новая редакция тестовой выборки с объединением соседних объектов."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping, shape
from shapely.ops import unary_union
from sqlalchemy import select

from ._markup_export import GeneratedMarkupFiles, GeneratedMarkupTile, _geojson_crs
from ._models import TestSampleRow
from ._test_samples import (
    _build_test_sample_thumbnails,
    _detail,
    _ensure_test_sample_preview,
    _new_test_sample_row,
    _sample_root,
    _sample_row,
    _utc_now,
    _validated_tile_indices,
    queue_test_sample_evaluation,
)
from .contracts import JobSource, TrainingUIAPIError


class TestSampleRevisionConflict(TrainingUIAPIError):
    """Исходная выборка изменена после подготовки запроса."""


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _merged_document(document, groups):
    if document.get("type") != "FeatureCollection":
        raise TrainingUIAPIError("Разметка тайла должна быть FeatureCollection.")
    features = document.get("features", [])
    identifiers = [feature.get("id") for feature in features]
    if any(not isinstance(value, (str, int)) for value in identifiers):
        raise TrainingUIAPIError("Для объединения нужны стабильные ID всех объектов тайла.")
    if len(set(identifiers)) != len(identifiers):
        raise TrainingUIAPIError("ID объектов исходного тайла повторяются.")
    by_id = dict(zip(identifiers, features))
    removed = set()
    merged = []
    for group in groups:
        if any(identifier not in by_id for identifier in group):
            raise TrainingUIAPIError("Объект группы объединения не найден в исходном тайле.")
        selected = [by_id[identifier] for identifier in group]
        geometries = [shape(feature["geometry"]) for feature in selected]
        if any(g.geom_type not in {"Polygon", "MultiPolygon"} or not g.is_valid
               or g.is_empty or not np.isfinite(g.area) or g.area <= 0 for g in geometries):
            raise TrainingUIAPIError("Объединять можно только валидные непустые полигоны.")
        connected = {0}
        while True:
            neighbors = {index for index, geometry in enumerate(geometries)
                         if index not in connected and any(
                             geometry.boundary.intersection(geometries[other].boundary).length > 0
                             for other in connected)}
            if not neighbors:
                break
            connected.update(neighbors)
        if len(connected) != len(geometries):
            raise TrainingUIAPIError("Объединяемые участки должны иметь общие границы.")
        properties = [dict(feature.get("properties") or {}) for feature in selected]
        if any(p.get("_mlsystem2_role", "positive") != "positive" for p in properties):
            raise TrainingUIAPIError("Объединять можно только положительную разметку.")
        # Происхождение каждого исходного ID фиксируется в отдельном журнале редакции.
        ordinary = [{k: v for k, v in p.items() if not k.startswith("_mlsystem2_")}
                    for p in properties]
        if any(p != ordinary[0] for p in ordinary):
            raise TrainingUIAPIError("У объединяемых объектов различаются пользовательские свойства.")
        geometry = unary_union(geometries)
        if not geometry.is_valid or geometry.is_empty:
            raise TrainingUIAPIError("Объединение не дало валидной полигональной геометрии.")
        new_id = hashlib.sha256(json.dumps(group, ensure_ascii=False).encode()).hexdigest()[:24]
        if new_id in by_id:
            raise TrainingUIAPIError("ID объединённого объекта совпал с существующим.")
        merged.append({"type": "Feature", "id": new_id, "geometry": mapping(geometry),
                       "properties": {**ordinary[0], "_mlsystem2_role": "positive"}})
        removed.update(group)
    result = copy.deepcopy(document)
    result["features"] = [feature for feature in result["features"] if feature["id"] not in removed]
    result["features"].extend(merged)
    return result


def _check_unchanged_mask(before, after, tif_path, mask_path):
    with rasterio.open(tif_path) as raster:
        if raster.crs is None or _geojson_crs(before) != raster.crs:
            raise TrainingUIAPIError("CRS разметки не совпадает с CRS сохранённого снимка.")
        masks = [rasterize([(feature["geometry"], 1) for feature in document["features"]],
                           out_shape=raster.shape, transform=raster.transform, dtype="uint8")
                 for document in (before, after)]
    with rasterio.open(mask_path) as source:
        saved = source.read(1)
    if not np.array_equal(masks[0], masks[1]) or not np.array_equal(saved > 0, masks[1] > 0):
        raise TrainingUIAPIError("Объединение изменяет пиксельную маску; редакция отклонена.")


@contextmanager
def merge_test_sample_annotations(session, sample_id, request, config, *, actor):
    """Публиковать файлы новой редакции только вместе с транзакцией вызывающего маршрута."""
    # Межпроцессная сериализация одинаковых запросов; источник никогда не перезаписывается.
    session.execute(select(TestSampleRow.id).where(TestSampleRow.id == sample_id).with_for_update())
    source = _sample_row(session, sample_id)
    if source.content_revision != request.expected_revision:
        raise TestSampleRevisionConflict("Выборка изменилась. Обновите её и подготовьте запрос заново.")
    if source.task != "binary":
        raise TrainingUIAPIError("Объединение пока поддерживает только бинарные тестовые выборки.")
    _validated_tile_indices(source, [tile.tile_index for tile in request.tiles])
    fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
    new_id = uuid.uuid5(sample_id, fingerprint)
    existing = session.get(TestSampleRow, new_id)
    if existing is not None:
        yield _detail(session, existing, config)
        return

    source_root = _sample_root(config, sample_id)
    final_root = _sample_root(config, new_id)
    building_root = final_root.with_name(f".building-{new_id}-{uuid.uuid4().hex}")
    changes = {tile.tile_index: tile for tile in request.tiles}
    published = False
    try:
        building_root.mkdir(parents=True, exist_ok=False)
        generated_tiles = []
        audit_tiles = []
        for tile in source.tiles:
            base = f"tile_{tile.tile_index:03d}"
            # Копируются только неизменяемые исходники, производные превью пересоздаются.
            for suffix in (".tif", ".geojson", "_mask.png"):
                shutil.copy2(source_root / f"{base}{suffix}", building_root / f"{base}{suffix}")
            tiff_sha256 = _sha256(source_root / f"{base}.tif")
            if tiff_sha256 != _sha256(building_root / f"{base}.tif"):
                raise TrainingUIAPIError("Контрольная сумма скопированного снимка не совпала.")
            count = tile.object_count
            if tile.tile_index in changes:
                path = building_root / f"{base}.geojson"
                before = json.loads(path.read_text(encoding="utf-8"))
                groups = changes[tile.tile_index].groups
                after = _merged_document(before, groups)
                _check_unchanged_mask(before, after, building_root / f"{base}.tif",
                                      building_root / f"{base}_mask.png")
                path.write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
                count = len(after["features"])
            audit_tiles.append({"tile_index": tile.tile_index,
                                "groups": changes[tile.tile_index].groups
                                if tile.tile_index in changes else [],
                                "before_count": tile.object_count, "after_count": count,
                                "changed_pixels": 0, "tiff_sha256": tiff_sha256,
                                "before_geojson_sha256": _sha256(source_root / f"{base}.geojson"),
                                "after_geojson_sha256": _sha256(building_root / f"{base}.geojson")})
            preview = _ensure_test_sample_preview(building_root, tile.tile_index)
            generated_tiles.append(GeneratedMarkupTile(
                index=tile.tile_index, source_name=tile.source_name, territory=tile.territory,
                object_count=count, class_object_counts={}, preview_filename=preview.name))
        generated = GeneratedMarkupFiles(
            **{key: getattr(source, key) for key in (
                "dataset_key", "dataset_name", "dataset_short_name", "dataset_version",
                "class_key", "class_name", "task", "tile_width", "tile_height",
                "requested_object_count", "exclude_boundary_objects", "territory_count")},
            actual_object_count=sum(tile.object_count for tile in generated_tiles),
            class_schema=(), class_object_counts={}, warnings=tuple(source.warnings or []),
            tiles=tuple(generated_tiles))
        _build_test_sample_thumbnails(building_root, generated.tiles)
        audit = {"source_sample_id": str(sample_id), "source_revision": source.content_revision,
                 "request_sha256": fingerprint, "actor": actor, "created_at": _utc_now().isoformat(),
                 "operation": "merge_annotations", "tiles": audit_tiles}
        (building_root / "annotation_revision.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        if final_root.exists():
            raise TrainingUIAPIError("Каталог редакции уже существует без записи; нужна проверка хранилища.")
        building_root.replace(final_root)
        published = True
        row = _new_test_sample_row(new_id, request.name, generated, quality_metric=source.quality_metric)
        row.source_training_result_id = source.source_training_result_id
        row.source_pseudo_result_id = source.source_pseudo_result_id
        enabled = {tile.tile_index for tile in source.tiles if tile.enabled}
        for tile in row.tiles:
            tile.enabled = tile.tile_index in enabled
        session.add(row)
        session.flush()
        # Ни старые объектовые метрики, ни поснимочные кэши не переносятся.
        queue_test_sample_evaluation(session, row, config, source=JobSource.MANUAL)
        session.flush()
        yield _detail(session, row, config)
    except BaseException:
        shutil.rmtree(building_root, ignore_errors=True)
        if published:
            shutil.rmtree(final_root, ignore_errors=True)
        raise

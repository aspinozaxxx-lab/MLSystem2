"""Источники тестовых вырезок: версии класса, личные черновики и публикация."""

from copy import deepcopy
from datetime import timezone
from pathlib import Path
import uuid

import rasterio
from shapely.geometry import mapping, shape
from shapely.errors import GEOSException
from sqlalchemy import select

from ._annotation_versions import payload_hash
from ._polygon_history import identified_payload
from ._dataset_catalog import dataset_class_row
from ._dataset_editor import (
    DatasetEditorConflict, _dataset_images_root, _draft_info, _draft_summary,
    _reproject_editor_geojson, _safe_raster_path,
    list_editor_datasets, list_editor_scenes,
)
from ._models import (
    AnnotationImageRow, AnnotationVersionRow, DatasetAnnotationRow, DatasetRow, DatasetClassRow,
    ManagedDatasetSourceRow, TestSampleSourceRow, TestSampleTileRow,
)
from ._raster_valid_data import editor_clipping_footprint
from ._test_region_crop import crop_test_region
from .contracts import (
    AnnotationVersionCatalog, AnnotationVersionDetail, AnnotationVersionInfo, AnnotationVersionUsage,
    DatasetEditorSceneDetail, DatasetEditorSceneInfo, TrainingUIAPIError,
)


def is_source(name):
    if not name.startswith("source_") or not name.endswith(".geojson"):
        return False
    try:
        return name == source_name(uuid.UUID(name[7:-8]))
    except ValueError:
        return False


def source_name(image_id):
    return f"source_{image_id}.geojson"


def source_row(sample, image_id):
    return next((row for row in sample.sources if row.image_id == image_id), None)


def source_image(session, sample, name):
    try:
        image_id = uuid.UUID(name.removeprefix("source_").removesuffix(".geojson"))
    except ValueError as exc:
        raise TrainingUIAPIError("Некорректное имя исходного снимка.") from exc
    if name != source_name(image_id):
        raise TrainingUIAPIError("Некорректное имя исходного снимка.")
    image = session.get(AnnotationImageRow, image_id)
    class_row = dataset_class_row(session, sample.class_key)
    permitted = class_row and session.scalar(select(AnnotationVersionRow.id).where(
        AnnotationVersionRow.image_id == image_id, AnnotationVersionRow.class_id == class_row.id,
        AnnotationVersionRow.deleted_at.is_(None)).limit(1))
    if image is None or (source_row(sample, image_id) is None and not permitted):
        raise TrainingUIAPIError("Снимок не имеет разметки в текущем классе.")
    return image


def source_path(dataset, image):
    if image.imagery_type != dataset.imagery_type.value:
        raise TrainingUIAPIError("Тип исходного снимка не соответствует тестовому набору.")
    return _safe_raster_path(_dataset_images_root(dataset), image.relative_path)


def raster_revision(path):
    """Идентичность файла и маски без чтения многогигабайтного TIFF при автосохранении."""
    files = [path, Path(str(path) + ".msk")]
    return payload_hash([(p.name, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
                         for p in files if p.exists()])


def version_catalog(session, sample, image, selected=None):
    class_row = dataset_class_row(session, sample.class_key)
    rows = list(session.execute(select(AnnotationVersionRow.id, AnnotationVersionRow.name, AnnotationVersionRow.revision,
        AnnotationVersionRow.created_at, AnnotationVersionRow.updated_at).where(
        AnnotationVersionRow.image_id == image.id, AnnotationVersionRow.class_id == class_row.id,
        AnnotationVersionRow.deleted_at.is_(None)).order_by(AnnotationVersionRow.created_at, AnnotationVersionRow.id)))
    usages = {}
    bindings = {}
    for version_id, dataset_id, key, name in session.execute(select(DatasetAnnotationRow.version_id, DatasetRow.id, DatasetRow.key, DatasetRow.name)
        .join(DatasetRow, DatasetRow.id == DatasetAnnotationRow.dataset_id)
        .where(DatasetAnnotationRow.image_id == image.id, DatasetRow.class_id == class_row.id, DatasetRow.deleted_at.is_(None))):
        usages.setdefault(version_id, []).append(AnnotationVersionUsage(dataset_key=key, dataset_name=name))
        bindings[dataset_id] = version_id
    for source_id, key, name, class_name in session.execute(
        select(ManagedDatasetSourceRow.source_dataset_id, DatasetRow.key, DatasetRow.name, DatasetClassRow.name)
        .join(DatasetRow, DatasetRow.id == ManagedDatasetSourceRow.managed_dataset_id)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(ManagedDatasetSourceRow.source_dataset_id.in_(bindings), DatasetRow.deleted_at.is_(None))
    ):
        usages[bindings[source_id]].append(AnnotationVersionUsage(dataset_key=key, dataset_name=f"{class_name} / {name}", indirect=True))
    versions = [AnnotationVersionInfo(id=row.id, name=row.name, revision=row.revision,
        created_at=row.created_at.replace(tzinfo=timezone.utc) if row.created_at.tzinfo is None else row.created_at,
        updated_at=row.updated_at.replace(tzinfo=timezone.utc) if row.updated_at.tzinfo is None else row.updated_at,
        usages=sorted({u.dataset_key: u for u in usages.get(row.id, [])}.values(), key=lambda u: u.dataset_name.casefold())) for row in rows]
    saved = source_row(sample, image.id)
    if saved and not any(str(v.id) == saved.version_info["id"] for v in versions):
        versions.append(AnnotationVersionInfo.model_validate(saved.version_info))
    selected = selected or (uuid.UUID(saved.version_info["id"]) if saved else (rows[0].id if rows else None))
    return AnnotationVersionCatalog(image_id=image.id,
        revision=payload_hash([v.model_dump(mode="json") for v in versions]), selected_id=selected, versions=versions)


def _positive_payload(payload, path):
    result = identified_payload(_reproject_editor_geojson(deepcopy(payload), path))
    result["features"] = [f for f in result["features"] if (f.get("properties") or {}).get("_mlsystem2_role", "positive") == "positive"]
    for feature in result["features"]:
        feature.setdefault("properties", {})["_mlsystem2_role"] = "positive"
    return result


def source_version(session, config, sample, dataset, name, version_id):
    image = source_image(session, sample, name)
    # Редакция и её геометрия читаются вместе: параллельная публикация не должна
    # подменить объекты между чтением номера редакции и формированием снимка.
    version = session.scalar(select(AnnotationVersionRow).where(AnnotationVersionRow.id == version_id)
        .with_for_update(read=True).execution_options(populate_existing=True))
    catalog = version_catalog(session, sample, image)
    info = next((v for v in catalog.versions if v.id == version_id), None)
    if info is None:
        raise TrainingUIAPIError("Версия не принадлежит этому снимку и классу.")
    saved = source_row(sample, image.id)
    payload = version.geojson if version and version.deleted_at is None else saved.geojson
    return AnnotationVersionDetail(catalog=catalog, version_id=version_id,
        geojson=_positive_payload(payload, source_path(dataset, image)))


def source_scenes(session, config, sample, dataset, drafts, *, include_class_sources=False):
    if include_class_sources:
        # Общий редактор выполняет штатный однократный импорт старых файлов в каталог версий.
        for peer in list_editor_datasets(session, config).datasets:
            if peer.class_key == sample.class_key and not peer.managed:
                list_editor_scenes(session, config, peer.key, username="")
    ids = {row.image_id for row in sample.sources}
    if include_class_sources:
        class_row = dataset_class_row(session, sample.class_key)
        ids.update(session.scalars(select(AnnotationVersionRow.image_id).where(
            AnnotationVersionRow.class_id == class_row.id, AnnotationVersionRow.deleted_at.is_(None))))
    for name in drafts:
        if is_source(name):
            ids.add(source_image(session, sample, name).id)
    scenes = []
    for image_id in sorted(ids, key=str):
        name = source_name(image_id)
        image = source_image(session, sample, name)
        saved = source_row(sample, image_id)
        catalog = version_catalog(session, sample, image)
        info = saved.version_info if saved else next(v for v in catalog.versions if v.id == catalog.selected_id).model_dump(mode="json")
        payload = saved.geojson if saved else session.scalar(select(AnnotationVersionRow.geojson).where(AnnotationVersionRow.id == catalog.selected_id))
        if not saved:
            payload = {"features": [f for f in payload.get("features", [])
                if (f.get("properties") or {}).get("_mlsystem2_role", "positive") == "positive"]}
        features = payload.get("features", [])
        zones = sum((f.get("properties") or {}).get("_mlsystem2_role") == "annotation_zone" for f in features)
        revision = saved.revision if saved else "new"
        draft = drafts.pop(name, None)
        scenes.append(DatasetEditorSceneInfo(scene_id=name, annotation_name=name, image_name=image.relative_path,
            raster_url=f"/api/v1/test-samples/{sample.id}/editor/rasters/{name}", revision=revision,
            total_count=len(features) - zones, positive_count=len(features) - zones, hard_negative_count=0, annotation_zone_count=zones,
            draft=_draft_summary(draft, dataset, revision) if draft else None, test_source=True, image_id=image_id,
            annotation_version_id=uuid.UUID(info["id"]), annotation_version_name=info["name"],
            annotation_version_revision=info["revision"], annotation_version_count=len(catalog.versions)))
    return sorted(scenes, key=lambda scene: scene.image_name.casefold())


def source_detail(session, config, sample, dataset, scene, draft):
    image = source_image(session, sample, scene.annotation_name)
    path = source_path(dataset, image)
    saved = source_row(sample, image.id)
    catalog = version_catalog(session, sample, image)
    if saved:
        payload = deepcopy(saved.geojson)
    else:
        selected = source_version(session, config, sample, dataset, scene.annotation_name, catalog.selected_id)
        payload, catalog = selected.geojson, selected.catalog
        info = next(v for v in catalog.versions if v.id == selected.version_id)
        scene = scene.model_copy(update={"annotation_version_revision": info.revision, "annotation_version_name": info.name})
    with rasterio.open(path) as raster:
        grid = list(raster.transform)[:6]
    return DatasetEditorSceneDetail(scene=scene, geojson=payload,
        valid_data_footprint=dict(mapping(editor_clipping_footprint(path))),
        draft=_draft_info(draft, dataset, scene.revision) if draft else None, annotation_versions=catalog, raster_grid=grid)


def prepare_source_draft(session, config, sample, dataset, name, request, previous_draft):
    image = source_image(session, sample, name)
    saved = source_row(sample, image.id)
    if request.base_revision != (saved.revision if saved else "new"):
        raise DatasetEditorConflict("Тестовые области изменены другим пользователем. Обновите снимок.")
    state = request.annotation_versions
    if state is None or state.edits or state.replacements:
        raise TrainingUIAPIError("В тестовом наборе можно выбрать версию; менять исходную разметку следует в датасете.")
    path = source_path(dataset, image)
    catalog = version_catalog(session, sample, image)
    info = next((v.model_dump(mode="json") for v in catalog.versions if v.id == state.selected_id), None)
    # Используем только серверные снимки данных: клиент редактирует области, не эталонные объекты.
    frozen = []
    if previous_draft:
        meta = previous_draft.geojson.get("_mlsystem2_test_source", {})
        frozen.append((meta.get("version", {}), previous_draft.geojson))
    if saved:
        frozen.append((saved.version_info, saved.geojson))
    payload = None
    for previous_info, previous_payload in frozen:
        if previous_info.get("id") == str(state.selected_id) and previous_info.get("revision") == state.selected_revision:
            info, payload = previous_info, _positive_payload(previous_payload, path)
            break
    if payload is None:
        if info is None:
            raise DatasetEditorConflict("Исходная версия уже изменилась. Выберите её заново перед сохранением.")
        selected = source_version(session, config, sample, dataset, name, state.selected_id)
        catalog = selected.catalog
        info = next(v.model_dump(mode="json") for v in catalog.versions if v.id == selected.version_id)
        if state.selected_revision not in (None, info["revision"]):
            raise DatasetEditorConflict("Исходная версия уже изменилась. Выберите её заново перед сохранением.")
        payload = selected.geojson
    regions = []
    features = request.geojson.get("features", [])
    if not isinstance(features, list) or any(not isinstance(f, dict) or not isinstance(f.get("properties", {}), dict) for f in features):
        raise TrainingUIAPIError("Тестовые области должны быть списком GeoJSON-объектов со свойствами.")
    positive_ids = {str(f.get("id")) for f in payload["features"]}
    for feature in features:
        if (feature.get("properties") or {}).get("_mlsystem2_role") != "annotation_zone":
            continue
        try:
            geometry = shape(feature.get("geometry") or {})
        except (ValueError, TypeError, AttributeError, GEOSException) as exc:
            raise TrainingUIAPIError("Не удалось прочитать геометрию тестовой области.") from exc
        if geometry.geom_type not in {"Polygon", "MultiPolygon"} or geometry.is_empty or not geometry.is_valid:
            raise TrainingUIAPIError("Тестовая область должна быть корректным полигоном без самопересечений.")
        identifier = str(feature.get("id") or uuid.uuid4())
        if identifier in positive_ids or len(identifier) > 128 or any(item["id"] == identifier for item in regions):
            raise TrainingUIAPIError("Идентификаторы тестовых областей должны быть уникальны и не длиннее 128 символов.")
        regions.append({"type": "Feature", "id": identifier, "properties": {"_mlsystem2_role": "annotation_zone"}, "geometry": dict(mapping(geometry))})
    # Редактор работает в EPSG:3857; произвольный входной CRS обрабатываем тем же преобразованием.
    projected = _reproject_editor_geojson({**request.geojson, "features": regions}, path)
    payload["features"].extend(projected["features"])
    payload["_mlsystem2_test_source"] = {"version": info, "raster_revision": raster_revision(path)}
    return payload, state.model_copy(update={"catalog_revision": catalog.revision, "selected_revision": info["revision"]})


def publish_source(session, config, sample, dataset, stage, draft, next_index):
    from ._test_sample_editor import _tile_name, _write_geojson_atomic, _write_mask, _utc_now, _normalize

    image = source_image(session, sample, draft.annotation_name)
    path = source_path(dataset, image)
    saved = source_row(sample, image.id)
    if draft.base_revision != (saved.revision if saved else "new"):
        raise DatasetEditorConflict("Тестовые области изменены другим пользователем. Обновите снимок.")
    metadata = draft.geojson["_mlsystem2_test_source"]
    if not draft.deleted and metadata["raster_revision"] != raster_revision(path):
        raise DatasetEditorConflict("Исходный TIFF изменился после сохранения областей. Откройте снимок заново.")
    regions = [] if draft.deleted else [f for f in draft.geojson["features"] if f["properties"].get("_mlsystem2_role") == "annotation_zone"]
    old_tiles = {tile.source_region_id: tile for tile in sample.tiles if saved and tile.source_scene_id == saved.id}
    if regions and saved is None:
        saved = TestSampleSourceRow(id=uuid.uuid4(), image_id=image.id)
        sample.sources.append(saved)
    for region in regions:
        tile = old_tiles.pop(region["id"], None)
        if tile is None:
            tile = TestSampleTileRow(tile_index=next_index, source_name=image.relative_path, territory=Path(image.relative_path).parent.name,
                object_count=0, enabled=True, source_scene_id=saved.id, source_region_id=region["id"])
            next_index += 1
            sample.tiles.append(tile)
        raster_path = stage / Path(_tile_name(tile)).with_suffix(".tif")
        # Staging содержит hardlink: никогда не перезаписываем существующий inode.
        raster_path.unlink(missing_ok=True)
        payload = crop_test_region(path, raster_path, region["geometry"], draft.geojson, dataset.imagery_type)
        payload.pop("_mlsystem2_test_source", None)
        payload = _normalize(payload, {}, dataset, raster_path, clip=False)
        _write_geojson_atomic(stage / _tile_name(tile), payload)
        _write_mask(stage, tile, payload, sample.class_schema or [])
        tile.object_count = len(payload["features"])
        tile.class_object_counts = {item["slug"]: sum(f["properties"].get("_mlsystem2_class") == item["slug"] for f in payload["features"])
                                   for item in sample.class_schema or []}
        tile.updated_at = _utc_now()
        with rasterio.open(raster_path) as raster:
            sample.tile_width, sample.tile_height = raster.width, raster.height
    for tile in old_tiles.values():
        sample.tiles.remove(tile)
        for pattern in (f"tile_{tile.tile_index:03d}.*", f"tile_{tile.tile_index:03d}_*"):
            for path in stage.glob(pattern):
                path.unlink()
    if saved:
        if regions:
            saved.geojson, saved.version_info = deepcopy(draft.geojson), deepcopy(metadata["version"])
            saved.raster_revision = metadata["raster_revision"]
            saved.revision, saved.updated_at = uuid.uuid4().hex, _utc_now()
        else:
            sample.sources.remove(saved)
    return next_index

"""Управляемый каталог классов и их датасетов."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from pathlib import Path, PurePosixPath

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from mlsystem2.dataset_preparing.api import (
    is_per_image_footprint_name,
    load_dataset_manifest,
    resolve_scene_images,
)
from mlsystem2.dataset_preparing.contracts import (
    DatasetManifest,
    DatasetPreparationError,
    SceneImageResolutionRequest,
)

from ._config import TrainingUIAPIConfig
from ._datasets import (
    CUSTOM_KEY,
    CUSTOM_NAME,
    DEFAULT_DATASET_NAME,
    IMAGERY_CHANNELS,
    IMAGERY_FOLDERS,
    RASTER_SUFFIXES,
    _annotation_files,
    _dataset_image_count,
    _first_file,
    _image_index,
    _path_metadata,
    build_per_image_index,
    imagery_images_dir,
    resolve_scenes_file_images,
)
from ._models import (
    AutomationRuleRow,
    DatasetClassRow,
    DatasetEditorDraftRow,
    DatasetRow,
    ManagedDatasetSourceRow,
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TestSampleBatchItemRow,
    TestSampleRow,
    TrainingResultRow,
    TrainingResultTestMetricRow,
    TrainingTemplateRow,
)
from .contracts import (
    ClassInfo,
    DatasetCatalogInfo,
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetFormat,
    DatasetInfo,
    DatasetObjectType,
    DatasetPrimaryDatasetUpdate,
    DatasetSourceInfo,
    ImageryTypeInfo,
    ManagedDatasetCreate,
    ManagedDatasetCompositionCreate,
    ManagedDatasetCompositionSourceCreate,
    ManagedDatasetUpdate,
    TrainingUIAPIError,
)
from ._managed_datasets import (
    SOURCE_MANAGED,
    managed_dataset_version,
    managed_manifest,
    managed_source_infos,
    materialize_managed_dataset,
)


SOURCE_MLMARKUP = "mlmarkup"
DEFAULT_IMAGERY_TYPE = "kanopus"
IMAGERY_NAMES = {"kanopus": "Канопус", "ortho": "Ортофото"}
QUALITY_PIXEL = "pixel"
QUALITY_OBJECTS = "objects"
MANAGED_DATASET_PALETTE = (
    "#3B82F6",
    "#22C55E",
    "#F59E0B",
    "#8B5CF6",
    "#EF4444",
    "#06B6D4",
    "#EC4899",
    "#84CC16",
)
_SYNC_LOCK = threading.RLock()
_SYNC_TTL_SECONDS = 60.0
_LAST_SYNC_BY_ROOT: dict[Path, tuple[float, tuple[tuple[str, int], ...]]] = {}
_HISTORICAL_MODEL_NAME_STEMS = {
    ("Абразия", "main"): "abrasion",
    ("Ветровая эрозия", "main"): "wind_erosion",
    ("Водная эрозия", "main"): "water_erosion",
    ("Вырубки", "main"): "deforestation",
    ("Вырубки", "strict"): "deforestationStrict",
    ("Вырубки", "test"): "deforestation",
    ("Гари", "main"): "burnt_forests",
    ("Границы леса", "main"): "forest",
    ("ЗУ500", "main"): "zu500",
    ("Заболачивание", "main"): "swampings",
    ("Засоления", "main"): "salty",
    ("Захламнения", "main"): "landfills",
    ("Захламнения", "test"): "landfills",
    ("Карьеры", "main"): "careers",
    ("ОКС500", "main"): "oks500",
    ("Обвально-оползневые и осыпные", "main"): "landslides",
    ("Озера", "main"): "lakes",
    ("Опустынивание", "main"): "desertification",
    ("Опустынивание и ветровая эрозия", "main"): (
        "desertification_wind_erosion"
    ),
    ("Пашни", "main"): "areas_of_used_arable_land",
    ("Переувлажнения", "main"): "floodings",
    ("Переувлажнения", "test"): "floodings",
    ("Переувлажнения и заболачивания", "main"): "floodings_swampings",
    ("Переувлажнения и заболачивания", "test"): "floodings_swampings",
    ("Разрушки", "main"): "damaged_oks",
    ("Разрушки", "test"): "damaged_oks",
    ("Реки", "main"): "rivers",
    ("Реки", "test"): "rivers",
}
_TECHNICAL_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?$")


def synchronize_dataset_catalog(session: Session, config: TrainingUIAPIConfig) -> None:
    """Идемпотентно импортирует историю и новые папки MLMarkup."""

    with _SYNC_LOCK:
        initial_import = session.scalar(select(DatasetRow.id).limit(1)) is None
        _import_historical_dataset_keys(session, config)
        _import_mlmarkup_folders(session, config, preserve_legacy_keys=initial_import)
        _ensure_model_name_stems(session, config)
        session.flush()
        root = Path(config.mlmarkup_root).resolve()
        _LAST_SYNC_BY_ROOT[root] = (time.monotonic(), _catalog_tree_stamp(root))


def _synchronize_dataset_catalog_if_stale(
    session: Session,
    config: TrainingUIAPIConfig,
) -> None:
    root = Path(config.mlmarkup_root).resolve()
    tree_stamp = _catalog_tree_stamp(root)
    with _SYNC_LOCK:
        last_sync = _LAST_SYNC_BY_ROOT.get(root)
        if (
            last_sync is not None
            and time.monotonic() - last_sync[0] < _SYNC_TTL_SECONDS
            and last_sync[1] == tree_stamp
        ):
            return
        synchronize_dataset_catalog(session, config)


def _catalog_tree_stamp(root: Path) -> tuple[tuple[str, int], ...]:
    """Дешёвая ревизия папок class/dataset для мгновенного сброса TTL."""

    if not root.is_dir():
        return ()
    result: list[tuple[str, int]] = []
    try:
        class_dirs = [path for path in root.iterdir() if path.is_dir()]
        for class_dir in sorted(class_dirs, key=lambda item: item.name.casefold()):
            result.append((class_dir.name, class_dir.stat().st_mtime_ns))
            dataset_dirs = [path for path in class_dir.iterdir() if path.is_dir()]
            for dataset_dir in sorted(
                dataset_dirs,
                key=lambda item: item.name.casefold(),
            ):
                result.append(
                    (
                        f"{class_dir.name}/{dataset_dir.name}",
                        dataset_dir.stat().st_mtime_ns,
                    )
                )
    except OSError:
        return ()
    return tuple(result)


def _ensure_model_name_stems(
    session: Session,
    config: TrainingUIAPIConfig,
) -> None:
    rows = session.execute(
        select(DatasetRow, DatasetClassRow).join(
            DatasetClassRow,
            DatasetClassRow.id == DatasetRow.class_id,
        )
    ).all()
    for dataset, class_row in rows:
        _ensure_model_name_stem(session, dataset, class_row, config)


def _ensure_model_name_stem(
    session: Session,
    dataset: DatasetRow,
    class_row: DatasetClassRow,
    config: TrainingUIAPIConfig,
) -> None:
    if dataset.model_name_stem:
        return
    dataset_name = dataset.name.split(" [legacy ", maxsplit=1)[0]
    stem = _HISTORICAL_MODEL_NAME_STEMS.get((class_row.name, dataset_name))
    if stem is None:
        stem = _HISTORICAL_MODEL_NAME_STEMS.get((class_row.name, "main"))
    if stem is None and dataset.source_type == SOURCE_MLMARKUP:
        source_path = _resolved_source_path(config.mlmarkup_root, dataset.source_path)
        if source_path.is_dir() and _first_file(source_path, ".txt") is not None:
            annotation_file, _, _ = _annotation_files(source_path)
            if annotation_file is not None:
                stem = annotation_file.stem
    if stem is None:
        stem = session.scalar(
            select(DatasetRow.model_name_stem)
            .where(
                DatasetRow.class_id == dataset.class_id,
                DatasetRow.id != dataset.id,
                DatasetRow.model_name_stem.is_not(None),
            )
            .order_by(DatasetRow.deleted_at.asc().nulls_first(), DatasetRow.created_at)
            .limit(1)
        )
    if stem:
        dataset.model_name_stem = stem[:160]


def successful_training_results(
    session: Session,
    class_or_dataset_key: str,
    *,
    limit: int | None = None,
) -> list[TrainingResultRow]:
    """Вернуть успешные сети класса от новой к старой."""

    class_row = dataset_class_row(session, class_or_dataset_key)
    dataset_keys = (
        session.scalars(select(DatasetRow.key).where(DatasetRow.class_id == class_row.id)).all()
        if class_row is not None
        else [class_or_dataset_key]
    )
    statement = (
        select(TrainingResultRow)
        .where(
            (
                TrainingResultRow.dataset_key.in_(dataset_keys)
                | TrainingResultRow.class_key.in_(dataset_keys)
                | (TrainingResultRow.class_key == class_or_dataset_key)
            ),
            TrainingResultRow.status == "ok",
        )
        .order_by(
            TrainingResultRow.trained_at.desc().nullslast(),
            TrainingResultRow.created_at.desc(),
            TrainingResultRow.id.desc(),
        )
    )
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement).all())


def primary_training_result(
    session: Session,
    class_or_dataset_key: str,
) -> TrainingResultRow | None:
    """Вернуть эффективную сеть класса: явно выбранную либо последнюю успешную."""

    class_row = dataset_class_row(session, class_or_dataset_key)
    if class_row is not None and class_row.primary_training_result_id is not None:
        selected = session.get(TrainingResultRow, class_row.primary_training_result_id)
        if selected is not None and selected.status == "ok":
            return selected
    return next(
        iter(successful_training_results(session, class_or_dataset_key, limit=1)),
        None,
    )


def dataset_class_row(
    session: Session,
    class_or_dataset_key: str,
) -> DatasetClassRow | None:
    class_row = session.scalar(
        select(DatasetClassRow).where(DatasetClassRow.key == class_or_dataset_key)
    )
    if class_row is not None:
        return class_row
    return session.scalar(
        select(DatasetClassRow)
        .join(DatasetRow, DatasetRow.class_id == DatasetClassRow.id)
        .where(DatasetRow.key == class_or_dataset_key)
        .limit(1)
    )


def list_managed_datasets(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    include_custom: bool = True,
) -> list[DatasetInfo]:
    _synchronize_dataset_catalog_if_stale(session, config)
    rows = session.execute(
        select(DatasetRow, DatasetClassRow).join(
            DatasetClassRow,
            DatasetClassRow.id == DatasetRow.class_id,
        ).where(DatasetRow.deleted_at.is_(None))
    ).all()
    image_indexes: dict[Path, dict[str, list[Path]]] = {}
    per_image_indexes: dict[Path, dict[str, list[Path]]] = {}
    datasets = [
        _dataset_info(
            session,
            dataset,
            class_row,
            config,
            image_indexes=image_indexes,
            per_image_indexes=per_image_indexes,
        )
        for dataset, class_row in rows
    ]
    datasets.sort(
        key=lambda item: (
            (item.class_name or "").casefold(),
            not item.is_primary,
            (item.dataset_name or "").casefold(),
        )
    )
    if include_custom:
        datasets.append(_custom_dataset_info(config))
    return datasets


def list_managed_classes(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    include_custom: bool = True,
    managed_datasets: list[DatasetInfo] | None = None,
) -> list[ClassInfo]:
    datasets = (
        managed_datasets
        if managed_datasets is not None
        else list_managed_datasets(session, config, include_custom=False)
    )
    class_rows = session.scalars(select(DatasetClassRow)).all()
    by_class: dict[str, list[DatasetInfo]] = {}
    for dataset in datasets:
        if dataset.class_key is not None:
            by_class.setdefault(dataset.class_key, []).append(dataset)
    classes = [
        ClassInfo(
            key=row.key,
            name=row.name,
            technical_name=row.technical_name,
            updated_at=_latest_dataset_update(by_class.get(row.key, [])),
            datasets=by_class.get(row.key, []),
            quality_metric=row.quality_metric,
            imagery_type=row.imagery_type,
            primary_dataset_key=_dataset_key(session, row.primary_dataset_id),
        )
        for row in class_rows
    ]
    classes.sort(key=lambda item: item.name.casefold())
    if include_custom:
        custom_dataset = _custom_dataset_info(config)
        classes.append(
            ClassInfo(
                key=CUSTOM_KEY,
                name=CUSTOM_NAME,
                technical_name="custom",
                datasets=[custom_dataset],
                is_custom=True,
                imagery_type=DEFAULT_IMAGERY_TYPE,
                primary_dataset_key=CUSTOM_KEY,
            )
        )
    return classes


def find_managed_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetInfo | None:
    if dataset_key == CUSTOM_KEY:
        return _custom_dataset_info(config)
    _synchronize_dataset_catalog_if_stale(session, config)
    row = session.execute(
        select(DatasetRow, DatasetClassRow)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(
            DatasetRow.key == dataset_key,
            DatasetRow.deleted_at.is_(None),
        )
    ).one_or_none()
    if row is None:
        return None
    return _dataset_info(session, *row, config)


def find_managed_class(
    session: Session,
    config: TrainingUIAPIConfig,
    class_key: str,
) -> ClassInfo | None:
    return next(
        (item for item in list_managed_classes(session, config) if item.key == class_key),
        None,
    )


def managed_dataset_catalog(session: Session, config: TrainingUIAPIConfig) -> DatasetCatalogInfo:
    _synchronize_dataset_catalog_if_stale(session, config)
    return DatasetCatalogInfo(
        classes=list_managed_classes(session, config, include_custom=False),
        sources=_source_infos(session, config),
        imagery_types=list_imagery_types(config.images_root),
    )


def create_dataset_class(
    session: Session,
    request: DatasetClassCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    name = _clean_name(request.name, "Название класса")
    _ensure_class_name_available(session, name)
    technical_name = _class_technical_name(session, name, request.technical_name)
    session.add(
        DatasetClassRow(
            key=str(uuid.uuid4()),
            name=name,
            technical_name=technical_name,
            quality_metric=QUALITY_PIXEL,
            imagery_type=request.imagery_type.value,
        )
    )
    session.flush()
    return managed_dataset_catalog(session, config)


def update_dataset_class(
    session: Session,
    class_key: str,
    request: DatasetClassUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    row = _class_row(session, class_key)
    if request.name is not None:
        name = _clean_name(request.name, "Название класса")
        _ensure_class_name_available(session, name, exclude_id=row.id)
        row.name = name
    if request.technical_name is not None:
        technical_name = _clean_technical_name(request.technical_name)
        _ensure_class_technical_name_available(
            session,
            technical_name,
            exclude_id=row.id,
        )
        if technical_name != row.technical_name:
            _replace_source_class_technical_name(
                session,
                row,
                technical_name,
                config,
            )

    metric_changed = (
        request.quality_metric is not None
        and request.quality_metric.value != row.quality_metric
    )
    imagery_changed = (
        request.imagery_type is not None
        and request.imagery_type.value != row.imagery_type
    )
    if metric_changed:
        row.quality_metric = request.quality_metric.value
    if imagery_changed:
        row.imagery_type = request.imagery_type.value

    if metric_changed or imagery_changed:
        dataset_rows = session.scalars(
            select(DatasetRow).where(DatasetRow.class_id == row.id)
        ).all()
        for dataset in dataset_rows:
            dataset.config_revision += 1
            dataset.legacy_version = False
        if metric_changed:
            dataset_keys = [dataset.key for dataset in dataset_rows]
            if dataset_keys:
                for sample in session.scalars(
                    select(TestSampleRow).where(TestSampleRow.dataset_key.in_(dataset_keys))
                ).all():
                    sample.quality_metric = row.quality_metric
                for batch_item in session.scalars(
                    select(TestSampleBatchItemRow).where(
                        TestSampleBatchItemRow.dataset_key.in_(dataset_keys),
                        TestSampleBatchItemRow.status.in_(("queued", "running")),
                    )
                ).all():
                    batch_item.metric = row.quality_metric
    session.flush()
    return managed_dataset_catalog(session, config)


def set_primary_dataset(
    session: Session,
    class_key: str,
    request: DatasetPrimaryDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    class_row = _class_row(session, class_key)
    dataset = _dataset_row(session, request.dataset_key)
    if dataset.class_id != class_row.id:
        raise TrainingUIAPIError("Датасет не принадлежит выбранному классу")
    class_row.primary_dataset_id = dataset.id
    class_row.primary_dataset_locked = True
    session.flush()
    return managed_dataset_catalog(session, config)


def create_managed_dataset(
    session: Session,
    request: ManagedDatasetCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    class_row = _class_row(session, request.class_key)
    name = _clean_name(request.name, "Название датасета")
    source_path = _validate_source_path(
        config.mlmarkup_root,
        request.source_path,
        require_exists=True,
    )
    source_owner = session.scalar(
        select(DatasetRow).where(
            DatasetRow.source_type == SOURCE_MLMARKUP,
            DatasetRow.source_path == source_path,
        )
    )
    _ensure_dataset_name_available(
        session,
        class_row.id,
        name,
        exclude_id=source_owner.id if source_owner is not None else None,
    )
    if source_owner is None:
        dataset = DatasetRow(
            key=str(uuid.uuid4()),
            class_id=class_row.id,
            name=name,
            source_type=SOURCE_MLMARKUP,
            source_path=source_path,
            config_revision=1,
            legacy_version=False,
        )
        session.add(dataset)
        session.flush()
    else:
        previous_class = session.get(DatasetClassRow, source_owner.class_id)
        if previous_class is not None and previous_class.primary_dataset_id == source_owner.id:
            previous_class.primary_dataset_id = None
        source_owner.class_id = class_row.id
        source_owner.name = name
        source_owner.deleted_at = None
        source_owner.config_revision += 1
        source_owner.legacy_version = False
        dataset = source_owner
        session.flush()
    _assign_main_if_available(class_row, dataset)
    return managed_dataset_catalog(session, config)


def _managed_composition_source_specs(
    session: Session,
    config: TrainingUIAPIConfig,
    class_row: DatasetClassRow,
    requested_sources: list[ManagedDatasetCompositionSourceCreate],
    existing_relations: list[ManagedDatasetSourceRow] | None = None,
) -> list[
    tuple[DatasetRow, DatasetClassRow, ManagedDatasetCompositionSourceCreate, int, str]
]:
    requested_by_key = {item.dataset_key: item for item in requested_sources}
    source_rows = session.execute(
        select(DatasetRow, DatasetClassRow)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetRow.class_id)
        .where(
            DatasetRow.key.in_(requested_by_key),
            DatasetRow.deleted_at.is_(None),
        )
    ).all()
    if len(source_rows) != len(requested_by_key):
        found = {row.key for row, _class in source_rows}
        missing = sorted(set(requested_by_key) - found)
        raise TrainingUIAPIError("Не найдены исходные датасеты: " + ", ".join(missing))
    source_class_ids = {source_class.id for _dataset, source_class in source_rows}
    if len(source_class_ids) != len(source_rows):
        raise TrainingUIAPIError(
            "В одном управляемом датасете пока допускается один источник от каждого класса."
        )
    for source, source_class in source_rows:
        if source.source_type != SOURCE_MLMARKUP:
            raise TrainingUIAPIError("Управляемый датасет нельзя использовать как источник другого.")
        if source_class.imagery_type != class_row.imagery_type:
            raise TrainingUIAPIError(
                f"Тип снимков источника «{source_class.name}» не совпадает с целевым классом."
            )
        source_path = _resolved_source_path(config.mlmarkup_root, source.source_path)
        if _first_file(source_path, ".txt") is not None:
            raise TrainingUIAPIError(
                f"Источник «{source_class.name}\\{source.name}» должен быть per-image."
            )
        if load_dataset_manifest(source_path) is not None:
            raise TrainingUIAPIError(
                f"Источник «{source_class.name}\\{source.name}» должен быть binary."
            )

    existing_by_source = {
        item.source_dataset_id: item for item in (existing_relations or [])
    }
    ordered_sources = sorted(
        source_rows,
        key=lambda item: (
            0 if item[0].id in existing_by_source else 1,
            (
                existing_by_source[item[0].id].object_type_id
                if item[0].id in existing_by_source
                else -requested_by_key[item[0].key].priority
            ),
            item[1].name.casefold(),
            item[0].name.casefold(),
        ),
    )
    return [
        (
            source,
            source_class,
            requested_by_key[source.key],
            index,
            (
                requested_by_key[source.key].color
                or (
                    existing_by_source[source.id].color
                    if source.id in existing_by_source
                    else None
                )
                or MANAGED_DATASET_PALETTE[(index - 1) % len(MANAGED_DATASET_PALETTE)]
            ).upper(),
        )
        for index, (source, source_class) in enumerate(ordered_sources, start=1)
    ]


def _replace_managed_composition_sources(
    session: Session,
    dataset: DatasetRow,
    specs: list[
        tuple[DatasetRow, DatasetClassRow, ManagedDatasetCompositionSourceCreate, int, str]
    ],
) -> None:
    session.execute(
        delete(ManagedDatasetSourceRow).where(
            ManagedDatasetSourceRow.managed_dataset_id == dataset.id
        )
    )
    session.flush()
    for source, source_class, requested, object_type_id, color in specs:
        session.add(
            ManagedDatasetSourceRow(
                managed_dataset_id=dataset.id,
                source_dataset_id=source.id,
                priority=requested.priority,
                object_type_id=object_type_id,
                object_type_slug=source_class.technical_name,
                object_type_name=source_class.name,
                color=color,
            )
        )


def _managed_composition_signature(
    specs: list[
        tuple[DatasetRow, DatasetClassRow, ManagedDatasetCompositionSourceCreate, int, str]
    ],
) -> list[tuple[uuid.UUID, int, int, str, str, str]]:
    return [
        (
            source.id,
            requested.priority,
            object_type_id,
            source_class.technical_name,
            source_class.name,
            color,
        )
        for source, source_class, requested, object_type_id, color in specs
    ]


def create_managed_dataset_composition(
    session: Session,
    request: ManagedDatasetCompositionCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    """Создать виртуальный multiclass-датасет из binary per-image источников."""

    class_row = _class_row(session, request.class_key)
    name = _clean_name(request.name, "Название датасета")
    _ensure_dataset_name_available(session, class_row.id, name)
    specs = _managed_composition_source_specs(session, config, class_row, request.sources)

    dataset = DatasetRow(
        key=str(uuid.uuid4()),
        class_id=class_row.id,
        name=name,
        source_type=SOURCE_MANAGED,
        source_path=f"managed/{uuid.uuid4()}",
        config_revision=1,
        legacy_version=False,
    )
    session.add(dataset)
    session.flush()
    _replace_managed_composition_sources(session, dataset, specs)
    _assign_main_if_available(class_row, dataset)
    session.flush()
    return managed_dataset_catalog(session, config)


def update_managed_dataset(
    session: Session,
    dataset_key: str,
    request: ManagedDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    row = _dataset_row(session, dataset_key)
    if row.source_type == SOURCE_MANAGED and request.source_path is not None:
        raise TrainingUIAPIError(
            "Источник виртуального управляемого датасета меняется через его состав."
        )
    if row.source_type != SOURCE_MANAGED and request.sources is not None:
        raise TrainingUIAPIError("Состав задаётся только для виртуального управляемого датасета.")
    desired_name = (
        _clean_name(request.name, "Название датасета")
        if request.name is not None
        else row.name
    )
    desired_source = (
        _validate_source_path(config.mlmarkup_root, request.source_path, require_exists=True)
        if request.source_path is not None
        else row.source_path
    )
    _ensure_dataset_name_available(
        session,
        row.class_id,
        desired_name,
        exclude_id=row.id,
    )

    composition_specs = None
    composition_changed = False
    if request.sources is not None:
        class_row = session.get(DatasetClassRow, row.class_id)
        if class_row is None:
            raise TrainingUIAPIError(f"Класс датасета не найден: {dataset_key}")
        current_relations = session.scalars(
            select(ManagedDatasetSourceRow)
            .where(ManagedDatasetSourceRow.managed_dataset_id == row.id)
            .order_by(ManagedDatasetSourceRow.object_type_id)
        ).all()
        composition_specs = _managed_composition_source_specs(
            session,
            config,
            class_row,
            request.sources,
            current_relations,
        )
        current_signature = [
            (
                relation.source_dataset_id,
                relation.priority,
                relation.object_type_id,
                relation.object_type_slug,
                relation.object_type_name,
                relation.color.upper(),
            )
            for relation in current_relations
        ]
        composition_changed = current_signature != _managed_composition_signature(
            composition_specs
        )
        if composition_changed and session.scalar(
            select(DatasetEditorDraftRow.id)
            .where(DatasetEditorDraftRow.dataset_key == row.key)
            .limit(1)
        ) is not None:
            raise TrainingUIAPIError(
                "Сначала опубликуйте или отмените все черновики управляемого датасета."
            )

    changed = (
        desired_name != row.name
        or desired_source != row.source_path
        or composition_changed
    )
    source_owner: DatasetRow | None = None
    if desired_source != row.source_path:
        source_owner = session.scalar(
            select(DatasetRow).where(
                DatasetRow.source_type == row.source_type,
                DatasetRow.source_path == desired_source,
                DatasetRow.id != row.id,
            )
        )
        previous_source_path = row.source_path
        if source_owner is not None:
            source_owner.source_path = f".mlsystem2-source-swap/{uuid.uuid4()}"
            session.flush()
        row.source_path = desired_source
        session.flush()
        if source_owner is not None:
            source_owner.source_path = previous_source_path
            source_owner.config_revision += 1
            source_owner.legacy_version = False
    if composition_changed and composition_specs is not None:
        _replace_managed_composition_sources(session, row, composition_specs)
    row.name = desired_name
    if changed:
        row.config_revision += 1
        row.legacy_version = False
    session.flush()
    return managed_dataset_catalog(session, config)


def list_imagery_types(images_root: Path) -> list[ImageryTypeInfo]:
    root = Path(images_root).resolve()
    return [
        ImageryTypeInfo(
            key=imagery_type,
            name=IMAGERY_NAMES[imagery_type],
            folder=folder,
            path=str(imagery_images_dir(root, imagery_type)),
            input_channels=IMAGERY_CHANNELS[imagery_type],
            image_count=_recursive_raster_count(imagery_images_dir(root, imagery_type)),
        )
        for imagery_type, folder in IMAGERY_FOLDERS.items()
    ]


def _import_historical_dataset_keys(session: Session, config: TrainingUIAPIConfig) -> None:
    existing = set(session.scalars(select(DatasetRow.key)).all())
    for dataset_key in sorted(_historical_dataset_keys(session), key=str.casefold):
        if dataset_key in existing:
            continue
        class_name, dataset_name = _split_legacy_dataset_key(dataset_key)
        source_path = _legacy_source_path(config.mlmarkup_root, class_name, dataset_name)
        source_owner = session.scalar(
            select(DatasetRow).where(
                DatasetRow.source_type == SOURCE_MLMARKUP,
                DatasetRow.source_path == source_path,
            )
        )
        if source_owner is not None:
            continue
        class_row = _ensure_class(
            session,
            class_name,
            preserve_legacy_key=True,
            imagery_type=_infer_imagery_type(config, source_path),
        )
        normalized_name = _unique_dataset_name(session, class_row.id, dataset_name)
        dataset = DatasetRow(
            key=dataset_key,
            class_id=class_row.id,
            name=normalized_name,
            source_type=SOURCE_MLMARKUP,
            source_path=source_path,
            config_revision=1,
            legacy_version=True,
        )
        session.add(dataset)
        session.flush()
        _assign_main_if_available(class_row, dataset)
        existing.add(dataset_key)


def _import_mlmarkup_folders(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    preserve_legacy_keys: bool,
) -> None:
    assigned_rows = session.scalars(
        select(DatasetRow).where(DatasetRow.source_type == SOURCE_MLMARKUP)
    ).all()
    assigned_sources = {row.source_path for row in assigned_rows}
    source_root_classes: dict[str, set[uuid.UUID]] = {}
    for row in assigned_rows:
        source_parts = PurePosixPath(row.source_path).parts
        if source_parts:
            source_root_classes.setdefault(source_parts[0], set()).add(row.class_id)

    for class_name, dataset_name, source_path in _discover_mlmarkup_sources(config.mlmarkup_root):
        if source_path is not None and source_path in assigned_sources:
            continue
        mapped_classes = source_root_classes.get(class_name, set())
        class_row = (
            session.get(DatasetClassRow, next(iter(mapped_classes)))
            if len(mapped_classes) == 1
            else None
        )
        class_row = class_row or _ensure_class(
            session,
            class_name,
            preserve_legacy_key=preserve_legacy_keys,
            imagery_type=(
                _infer_imagery_type(config, source_path)
                if source_path is not None
                else DEFAULT_IMAGERY_TYPE
            ),
        )
        if source_path is None:
            continue
        preferred_key = f"{class_name}\\{dataset_name}"
        normalized_name = _unique_dataset_name(session, class_row.id, dataset_name)
        dataset_key = (
            _unique_dataset_key(session, preferred_key)
            if preserve_legacy_keys
            else str(uuid.uuid4())
        )
        dataset = DatasetRow(
            key=dataset_key,
            class_id=class_row.id,
            name=normalized_name,
            source_type=SOURCE_MLMARKUP,
            source_path=source_path,
            config_revision=1,
            legacy_version=True,
        )
        session.add(dataset)
        session.flush()
        _assign_main_if_available(class_row, dataset)
        assigned_sources.add(source_path)
        source_root_classes.setdefault(class_name, set()).add(class_row.id)


def _discover_mlmarkup_sources(root: Path) -> list[tuple[str, str, str | None]]:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    resolved_root = root.resolve()
    discovered: list[tuple[str, str, str | None]] = []
    for class_dir in sorted(
        (item for item in root.iterdir() if item.is_dir() and not item.name.startswith(".")),
        key=lambda item: item.name.casefold(),
    ):
        if not _is_within_root(class_dir.resolve(), resolved_root):
            discovered.append(
                (class_dir.name, DEFAULT_DATASET_NAME, class_dir.relative_to(root).as_posix())
            )
            continue
        dataset_dirs = sorted(
            (
                item
                for item in class_dir.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ),
            key=lambda item: (item.name != DEFAULT_DATASET_NAME, item.name.casefold()),
        )
        if dataset_dirs:
            for dataset_dir in dataset_dirs:
                discovered.append(
                    (
                        class_dir.name,
                        dataset_dir.name,
                        dataset_dir.relative_to(root).as_posix(),
                    )
                )
        elif _first_file(class_dir, ".txt") is not None or _first_file(class_dir, ".geojson") is not None:
            discovered.append(
                (class_dir.name, DEFAULT_DATASET_NAME, class_dir.relative_to(root).as_posix())
            )
        else:
            discovered.append((class_dir.name, "", None))
    return discovered


def _historical_dataset_keys(session: Session) -> set[str]:
    values: set[str] = set()
    columns = (
        TrainingTemplateRow.dataset_key,
        InferenceTemplateRow.dataset_key,
        AutomationRuleRow.dataset_key,
        JobRow.dataset_key,
        TrainingResultRow.dataset_key,
        TrainingResultRow.class_key,
        PseudoMarkupResultRow.dataset_key,
        PseudoMarkupResultRow.class_key,
        TestSampleRow.dataset_key,
        TestSampleBatchItemRow.dataset_key,
    )
    for column in columns:
        for value in session.scalars(select(column).where(column.is_not(None))).all():
            normalized = str(value).strip()
            if normalized and normalized.casefold() not in {CUSTOM_KEY, CUSTOM_NAME.casefold()}:
                values.add(normalized)
    return values


def _dataset_info(
    session: Session,
    dataset: DatasetRow,
    class_row: DatasetClassRow,
    config: TrainingUIAPIConfig,
    *,
    image_indexes: dict[Path, dict[str, list[Path]]] | None = None,
    per_image_indexes: dict[Path, dict[str, list[Path]]] | None = None,
) -> DatasetInfo:
    _ensure_model_name_stem(session, dataset, class_row, config)
    if dataset.source_type == SOURCE_MANAGED:
        return _managed_dataset_info(
            session,
            dataset,
            class_row,
            config,
            per_image_indexes=per_image_indexes,
        )
    source_path = _resolved_source_path(config.mlmarkup_root, dataset.source_path)
    images_dir = imagery_images_dir(config.images_root, class_row.imagery_type)
    source_inside_root = _is_within_root(source_path, Path(config.mlmarkup_root).resolve())
    images_inside_root = _is_within_root(images_dir, Path(config.images_root).resolve())
    diagnostics: list[str] = []
    scenes_file: Path | None = None
    annotation_file: Path | None = None
    hard_negative_file: Path | None = None
    dataset_format: DatasetFormat | None = None
    annotations_dir: Path | None = None
    updated_at = None
    source_version = None
    source_available = source_inside_root and source_path.is_dir()
    manifest: DatasetManifest | None = None
    source_status = "unknown"
    source_changes: list[str] = []
    class_counts: dict[str, int] = {}
    hard_negative_count = 0
    if not source_inside_root:
        diagnostics.append(
            f"Источник MLMarkup выходит за пределы разрешённого каталога: {dataset.source_path}"
        )
    elif not source_available:
        diagnostics.append(f"Источник MLMarkup недоступен: {dataset.source_path}")
    else:
        scenes_file = _first_file(source_path, ".txt")
        if scenes_file is not None:
            dataset_format = DatasetFormat.LEGACY
            annotation_file, hard_negative_file, source_diagnostics = _annotation_files(
                source_path
            )
            diagnostics.extend(source_diagnostics)
            if annotation_file is None and not source_diagnostics:
                diagnostics.append("В legacy-датасете не найден positive GeoJSON.")
        else:
            try:
                manifest = load_dataset_manifest(source_path)
            except DatasetPreparationError as exc:
                diagnostics.append(str(exc))
            dataset_format = (
                DatasetFormat.PER_IMAGE_MULTICLASS
                if manifest is not None
                else DatasetFormat.PER_IMAGE
            )
            annotations_dir = source_path
            if manifest is not None:
                source_status, source_changes = _manifest_source_status(
                    manifest,
                    Path(config.mlmarkup_root),
                )
                class_counts, hard_negative_count = _per_image_object_counts(
                    source_path,
                    manifest,
                    diagnostics,
                )
        updated_at, source_version = _path_metadata(source_path, config.mlmarkup_root)
    if not images_inside_root:
        diagnostics.append("Каталог снимков выходит за пределы MLSYSTEM2_IMAGES_ROOT.")
    elif not images_dir.is_dir():
        diagnostics.append(
            f"Каталог снимков типа «{IMAGERY_NAMES[class_row.imagery_type]}» недоступен."
        )
    version = source_version
    if not dataset.legacy_version:
        version = f"managed:{dataset.config_revision}:{source_version or 'missing'}"
    image_count: int | None = None
    index = None
    if scenes_file is not None and images_inside_root and images_dir.is_dir():
        if image_indexes is None:
            index = _image_index(images_dir)
        else:
            index = image_indexes.get(images_dir)
            if index is None:
                index = _image_index(images_dir)
                image_indexes[images_dir] = index
        image_count = _dataset_image_count(scenes_file, index)
    elif annotations_dir is not None and images_inside_root and images_dir.is_dir():
        if per_image_indexes is None:
            per_image_index = build_per_image_index(images_dir)
        else:
            per_image_index = per_image_indexes.get(images_dir)
            if per_image_index is None:
                per_image_index = build_per_image_index(images_dir)
                per_image_indexes[images_dir] = per_image_index
        image_count = _per_image_catalog_count(
            annotations_dir,
            per_image_index,
            diagnostics,
        )
    display_name = f"{class_row.name}\\{dataset.name}"
    return DatasetInfo(
        key=dataset.key,
        name=display_name,
        dataset_name=dataset.name,
        class_key=class_row.key,
        class_name=class_row.name,
        class_technical_name=class_row.technical_name,
        path=str(source_path),
        scenes_file=str(scenes_file) if scenes_file is not None else None,
        annotation_file=str(annotation_file) if annotation_file is not None else None,
        hard_negative_annotation_file=(
            str(hard_negative_file) if hard_negative_file is not None else None
        ),
        format=dataset_format,
        annotations_dir=str(annotations_dir) if annotations_dir is not None else None,
        image_count=image_count,
        version=version,
        updated_at=updated_at,
        quality_metric=class_row.quality_metric,
        imagery_type=class_row.imagery_type,
        input_channels=IMAGERY_CHANNELS[class_row.imagery_type],
        images_dir=str(images_dir) if images_inside_root else None,
        source_type=dataset.source_type,
        source_path=dataset.source_path,
        model_name_stem=dataset.model_name_stem,
        source_available=source_available,
        is_primary=class_row.primary_dataset_id == dataset.id,
        diagnostics=diagnostics,
        task="multiclass" if manifest is not None else "binary",
        object_types=(
            [
                DatasetObjectType(
                    id=item.id,
                    slug=item.slug,
                    name=item.name,
                    color=item.color,
                    priority=item.priority,
                )
                for item in manifest.classes
            ]
            if manifest is not None
            else []
        ),
        combined=bool(manifest and manifest.combined),
        source_status=("unavailable" if not source_available else source_status),
        source_changes=source_changes,
        class_counts=class_counts,
        hard_negative_count=hard_negative_count,
        manifest_path=(
            str(source_path / ".mlsystem2-dataset.json")
            if manifest is not None
            else None
        ),
    )


def _managed_dataset_info(
    session: Session,
    dataset: DatasetRow,
    class_row: DatasetClassRow,
    config: TrainingUIAPIConfig,
    *,
    per_image_indexes: dict[Path, dict[str, list[Path]]] | None,
) -> DatasetInfo:
    images_dir = imagery_images_dir(config.images_root, class_row.imagery_type)
    diagnostics: list[str] = []
    manifest = managed_manifest(session, dataset)
    managed_source_list = managed_source_infos(session, dataset.id)
    annotations_dir: Path | None = None
    version: str | None = None
    updated_at = None
    class_counts = {item.slug: 0 for item in manifest.classes}
    hard_negative_count = 0
    try:
        materialized = materialize_managed_dataset(
            session,
            config,
            dataset,
            source_root=config.mlmarkup_root,
            scope="live",
        )
        annotations_dir = materialized.path
        version = materialized.version
        updated_at = materialized.updated_at
        manifest = materialized.manifest
        class_counts = materialized.class_counts
        hard_negative_count = materialized.hard_negative_count
    except TrainingUIAPIError as exc:
        diagnostics.append(str(exc))
        try:
            version, updated_at = managed_dataset_version(
                session,
                dataset,
                config.mlmarkup_root,
            )
        except TrainingUIAPIError:
            version = f"managed:unavailable:{dataset.config_revision}"
    image_count = 0
    if annotations_dir is not None and images_dir.is_dir():
        if per_image_indexes is None:
            index = build_per_image_index(images_dir)
        else:
            index = per_image_indexes.get(images_dir)
            if index is None:
                index = build_per_image_index(images_dir)
                per_image_indexes[images_dir] = index
        image_count = _per_image_catalog_count(annotations_dir, index, diagnostics)
    display_name = f"{class_row.name}\\{dataset.name}"
    return DatasetInfo(
        key=dataset.key,
        name=display_name,
        dataset_name=dataset.name,
        class_key=class_row.key,
        class_name=class_row.name,
        class_technical_name=class_row.technical_name,
        path=str(annotations_dir) if annotations_dir is not None else None,
        format=DatasetFormat.PER_IMAGE_MULTICLASS,
        annotations_dir=str(annotations_dir) if annotations_dir is not None else None,
        image_count=image_count,
        version=version,
        updated_at=updated_at,
        quality_metric=class_row.quality_metric,
        imagery_type=class_row.imagery_type,
        input_channels=IMAGERY_CHANNELS[class_row.imagery_type],
        images_dir=str(images_dir),
        source_type=SOURCE_MANAGED,
        source_path=dataset.source_path,
        model_name_stem=dataset.model_name_stem,
        source_available=not diagnostics,
        is_primary=class_row.primary_dataset_id == dataset.id,
        diagnostics=diagnostics,
        task="multiclass",
        object_types=[
            DatasetObjectType(
                id=item.id,
                slug=item.slug,
                name=item.name,
                color=item.color,
                priority=item.priority,
            )
            for item in manifest.classes
        ],
        managed=True,
        managed_sources=managed_source_list,
        source_status="current" if not diagnostics else "unavailable",
        class_counts=class_counts,
        hard_negative_count=hard_negative_count,
        manifest_path=(
            str(annotations_dir / ".mlsystem2-dataset.json")
            if annotations_dir is not None
            else None
        ),
    )


def _manifest_source_status(
    manifest: DatasetManifest,
    mlmarkup_root: Path,
) -> tuple[str, list[str]]:
    if not manifest.sources:
        return "unknown", []
    changes: list[str] = []
    unavailable = False
    root = mlmarkup_root.resolve()
    for source in manifest.sources:
        raw_path = Path(source.path)
        source_path = raw_path if raw_path.is_absolute() else root / raw_path
        try:
            resolved = source_path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            unavailable = True
            changes.append(f"Исходная папка выходит за пределы MLMarkup: {source.path}")
            continue
        if not resolved.is_dir():
            unavailable = True
            changes.append(f"Исходная папка недоступна: {source.path}")
            continue
        current_hashes = _folder_file_hashes(resolved)
        if source.file_hashes:
            all_names = sorted(set(source.file_hashes) | set(current_hashes))
            for name in all_names:
                if name not in source.file_hashes:
                    changes.append(f"Добавлен исходный файл: {source.path}/{name}")
                elif name not in current_hashes:
                    changes.append(f"Удалён исходный файл: {source.path}/{name}")
                elif source.file_hashes[name] != current_hashes[name]:
                    changes.append(f"Изменён исходный файл: {source.path}/{name}")
        elif _tree_revision(current_hashes) != source.tree_revision:
            changes.append(f"Изменена исходная папка: {source.path}")
    if unavailable:
        return "unavailable", changes
    return ("stale", changes) if changes else ("current", [])


def _folder_file_hashes(folder: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(
        (item for item in folder.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(folder).as_posix().casefold(),
    ):
        relative = path.relative_to(folder).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[relative] = digest.hexdigest()
    return hashes


def _tree_revision(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _per_image_object_counts(
    folder: Path,
    manifest: DatasetManifest,
    diagnostics: list[str],
) -> tuple[dict[str, int], int]:
    counts = {item.slug: 0 for item in manifest.classes}
    hard_negative = 0
    try:
        for path in folder.glob("*.geojson"):
            if is_per_image_footprint_name(path.name):
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            for feature in payload.get("features", []):
                properties = feature.get("properties") or {}
                role = properties.get("_mlsystem2_role")
                if role == "hard_negative":
                    hard_negative += 1
                elif role == "positive":
                    slug = properties.get("_mlsystem2_class")
                    if slug in counts:
                        counts[str(slug)] += 1
    except (OSError, json.JSONDecodeError, AttributeError) as exc:
        diagnostics.append(f"Не удалось посчитать объекты multiclass-датасета: {exc}")
    return counts, hard_negative


def _custom_dataset_info(config: TrainingUIAPIConfig) -> DatasetInfo:
    return DatasetInfo(
        key=CUSTOM_KEY,
        name=CUSTOM_NAME,
        dataset_name=CUSTOM_NAME,
        class_key=CUSTOM_KEY,
        class_name=CUSTOM_NAME,
        class_technical_name="custom",
        is_custom=True,
        imagery_type=DEFAULT_IMAGERY_TYPE,
        input_channels=IMAGERY_CHANNELS[DEFAULT_IMAGERY_TYPE],
        images_dir=str(imagery_images_dir(config.images_root, DEFAULT_IMAGERY_TYPE)),
        source_available=True,
        is_primary=True,
    )


def _per_image_catalog_count(
    annotations_dir: Path,
    images_by_annotation: dict[str, list[Path]],
    diagnostics: list[str],
) -> int:
    annotation_files = sorted(
        (
            path
            for path in annotations_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".geojson"
            and not is_per_image_footprint_name(path.name)
        ),
        key=lambda item: item.name.casefold(),
    )
    missing = [
        path.name
        for path in annotation_files
        if not images_by_annotation.get(path.name.casefold())
    ]
    ambiguous = [
        path.name
        for path in annotation_files
        if len(images_by_annotation.get(path.name.casefold(), [])) > 1
    ]
    if missing:
        diagnostics.append(
            "Для GeoJSON не найдены TIFF: " + ", ".join(missing)
        )
    if ambiguous:
        diagnostics.append(
            "Имена GeoJSON неоднозначно сопоставлены с TIFF: "
            + ", ".join(ambiguous)
        )
    if not annotation_files:
        diagnostics.append(
            "Per-image датасет пуст: его можно редактировать, но нельзя использовать для обучения."
        )
    return len(annotation_files) - len(missing) - len(ambiguous)


def _source_infos(session: Session, config: TrainingUIAPIConfig) -> list[DatasetSourceInfo]:
    assigned = {
        row.source_path: row.key
        for row in session.scalars(
            select(DatasetRow).where(DatasetRow.source_type == SOURCE_MLMARKUP)
        ).all()
    }
    result: list[DatasetSourceInfo] = []
    for class_name, dataset_name, source_path in _discover_mlmarkup_sources(config.mlmarkup_root):
        if source_path is None:
            continue
        absolute = _resolved_source_path(config.mlmarkup_root, source_path)
        if not _is_within_root(absolute, Path(config.mlmarkup_root).resolve()):
            diagnostics = ["Источник выходит за пределы разрешённого каталога MLMarkup."]
        else:
            scenes_file = _first_file(absolute, ".txt")
            if scenes_file is None:
                diagnostics = []
                if _first_file(absolute, ".geojson") is None:
                    diagnostics.append(
                        "Per-image датасет пуст: его можно наполнить только через редактор."
                    )
            else:
                annotation, _hard_negative, diagnostics = _annotation_files(absolute)
                if annotation is None and not any(
                    "positive GeoJSON" in item for item in diagnostics
                ):
                    diagnostics.append("Не найден positive GeoJSON.")
        result.append(
            DatasetSourceInfo(
                key=source_path,
                name=f"{class_name}\\{dataset_name}",
                path=str(absolute),
                assigned_dataset_key=assigned.get(source_path),
                diagnostics=diagnostics,
            )
        )
    result.sort(key=lambda item: item.name.casefold())
    return result


def _ensure_class(
    session: Session,
    name: str,
    *,
    preserve_legacy_key: bool = False,
    imagery_type: str = DEFAULT_IMAGERY_TYPE,
) -> DatasetClassRow:
    row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == name))
    if row is not None:
        return row
    key = (
        name
        if preserve_legacy_key
        and session.scalar(select(DatasetClassRow.id).where(DatasetClassRow.key == name)) is None
        else str(uuid.uuid4())
    )
    row = DatasetClassRow(
        key=key,
        name=name,
        technical_name=_class_technical_name(session, name, None),
        quality_metric=QUALITY_PIXEL,
        imagery_type=imagery_type,
    )
    session.add(row)
    session.flush()
    return row


def _assign_main_if_available(class_row: DatasetClassRow, dataset: DatasetRow) -> None:
    if (
        dataset.name.casefold() == DEFAULT_DATASET_NAME
        and class_row.primary_dataset_id is None
        and not class_row.primary_dataset_locked
    ):
        class_row.primary_dataset_id = dataset.id


def _split_legacy_dataset_key(value: str) -> tuple[str, str]:
    normalized = value.replace("/", "\\").strip("\\")
    if "\\" not in normalized:
        return normalized, DEFAULT_DATASET_NAME
    class_name, dataset_name = normalized.rsplit("\\", 1)
    return class_name or normalized, dataset_name or DEFAULT_DATASET_NAME


def _legacy_source_path(root: Path, class_name: str, dataset_name: str) -> str:
    nested = _safe_relative_candidate(class_name, dataset_name)
    nested_path = _resolved_source_path(root, nested)
    resolved_root = Path(root).resolve()
    if _is_within_root(nested_path, resolved_root) and nested_path.is_dir():
        return nested
    if dataset_name.casefold() == DEFAULT_DATASET_NAME:
        flat = _safe_relative_candidate(class_name)
        flat_path = _resolved_source_path(root, flat)
        if _is_within_root(flat_path, resolved_root) and flat_path.is_dir():
            return flat
    return nested


def _infer_imagery_type(config: TrainingUIAPIConfig, source_path: str) -> str:
    source = _resolved_source_path(config.mlmarkup_root, source_path)
    if not _is_within_root(source, Path(config.mlmarkup_root).resolve()):
        return DEFAULT_IMAGERY_TYPE
    scenes = _first_file(source, ".txt") if source.is_dir() else None
    if scenes is not None:
        matched = [
            imagery_type
            for imagery_type in IMAGERY_FOLDERS
            if resolve_scenes_file_images(
                scenes,
                imagery_images_dir(config.images_root, imagery_type),
            )
        ]
        return matched[0] if len(matched) == 1 else DEFAULT_IMAGERY_TYPE
    matched = []
    for imagery_type in IMAGERY_FOLDERS:
        try:
            resolution = resolve_scene_images(
                SceneImageResolutionRequest(
                    images_dir=str(imagery_images_dir(config.images_root, imagery_type)),
                    annotations_dir=str(source),
                )
            )
        except (OSError, ValueError):
            continue
        if resolution.images and not resolution.missing_scenes and not resolution.ambiguous_scenes:
            matched.append(imagery_type)
    return matched[0] if len(matched) == 1 else DEFAULT_IMAGERY_TYPE


def _validate_source_path(root: Path, value: str, *, require_exists: bool) -> str:
    normalized = value.strip().replace("\\", "/").strip("/")
    candidate = PurePosixPath(normalized)
    if not normalized or candidate.is_absolute() or ".." in candidate.parts:
        raise TrainingUIAPIError("Путь источника должен находиться внутри MLMarkup")
    resolved = _resolved_source_path(root, candidate.as_posix())
    if not _is_within_root(resolved, Path(root).resolve()):
        raise TrainingUIAPIError("Путь источника выходит за пределы MLMarkup")
    if require_exists and not resolved.is_dir():
        raise TrainingUIAPIError(f"Папка источника не найдена: {normalized}")
    return candidate.as_posix()


def _resolved_source_path(root: Path, source_path: str) -> Path:
    parts = PurePosixPath(source_path).parts
    return Path(root).joinpath(*parts).resolve()


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_class_name_available(
    session: Session,
    name: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> None:
    statement = select(DatasetClassRow).where(DatasetClassRow.name == name)
    if exclude_id is not None:
        statement = statement.where(DatasetClassRow.id != exclude_id)
    if session.scalar(statement) is not None:
        raise TrainingUIAPIError(f"Класс с названием «{name}» уже существует")


def _ensure_class_technical_name_available(
    session: Session,
    technical_name: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> None:
    statement = select(DatasetClassRow).where(
        DatasetClassRow.technical_name == technical_name
    )
    if exclude_id is not None:
        statement = statement.where(DatasetClassRow.id != exclude_id)
    if session.scalar(statement) is not None:
        raise TrainingUIAPIError(
            f"Техническое имя «{technical_name}» уже используется другим классом"
        )


def _clean_technical_name(value: str) -> str:
    normalized = value.strip().lower()
    if not _TECHNICAL_NAME_RE.fullmatch(normalized):
        raise TrainingUIAPIError(
            "Техническое имя должно содержать только латинские строчные буквы, "
            "цифры, дефис и подчёркивание"
        )
    return normalized


def _class_technical_name(
    session: Session,
    class_name: str,
    requested: str | None,
) -> str:
    if requested is not None:
        value = _clean_technical_name(requested)
        _ensure_class_technical_name_available(session, value)
        return value
    historical = _HISTORICAL_MODEL_NAME_STEMS.get((class_name, "main"))
    base = _clean_technical_name(historical) if historical else "class"
    candidate = base
    suffix = 2
    while session.scalar(
        select(DatasetClassRow.id).where(
            DatasetClassRow.technical_name == candidate
        )
    ) is not None:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _replace_source_class_technical_name(
    session: Session,
    class_row: DatasetClassRow,
    technical_name: str,
    config: TrainingUIAPIConfig,
) -> None:
    old_slug = class_row.technical_name
    relations = session.execute(
        select(ManagedDatasetSourceRow, DatasetRow)
        .join(DatasetRow, DatasetRow.id == ManagedDatasetSourceRow.source_dataset_id)
        .where(DatasetRow.class_id == class_row.id)
    ).all()
    target_ids: set[uuid.UUID] = set()
    for relation, _source_dataset in relations:
        relation.object_type_slug = technical_name
        target_ids.add(relation.managed_dataset_id)
    class_row.technical_name = technical_name
    for target_id in target_ids:
        target = session.get(DatasetRow, target_id)
        if target is None:
            continue
        target.config_revision += 1
        target.legacy_version = False
        _canonicalize_managed_dataset_history(
            session,
            target,
            {old_slug: technical_name},
            config,
        )


def _canonicalize_managed_dataset_history(
    session: Session,
    dataset: DatasetRow,
    identifier_mapping: dict[str, str],
    config: TrainingUIAPIConfig,
) -> None:
    canonical_schema = [
        item.model_dump(mode="json") for item in managed_manifest(session, dataset).classes
    ]
    results = session.scalars(
        select(TrainingResultRow).where(
            (TrainingResultRow.dataset_key == dataset.key)
            | (TrainingResultRow.class_key == dataset.key)
        )
    ).all()
    result_ids: list[uuid.UUID] = []
    mapping_by_result: dict[uuid.UUID, dict[str, str]] = {}
    for result in results:
        schema, schema_mapping = _canonical_class_schema(
            result.class_schema,
            canonical_schema,
        )
        mapping = {**schema_mapping, **identifier_mapping}
        result.class_schema = schema
        result.training_metrics = _remap_identifiers(
            dict(result.training_metrics or {}),
            mapping,
        )
        result_ids.append(result.id)
        mapping_by_result[result.id] = mapping

    for job in session.scalars(
        select(JobRow).where(JobRow.dataset_key == dataset.key)
    ).all():
        state = dict(job.config or {})
        schema_mapping = _nested_class_schema_mapping(state, canonical_schema)
        mapping = {**schema_mapping, **identifier_mapping}
        state = _remap_identifiers(state, mapping)
        job.config = _canonicalize_nested_class_schemas(state, canonical_schema)

    samples = session.scalars(
        select(TestSampleRow)
        .where(TestSampleRow.dataset_key == dataset.key)
        .options(selectinload(TestSampleRow.tiles))
    ).all()
    for sample in samples:
        schema, schema_mapping = _canonical_class_schema(
            sample.class_schema,
            canonical_schema,
        )
        mapping = {**schema_mapping, **identifier_mapping}
        sample.class_schema = schema
        sample.evaluation_metrics = _remap_identifiers(
            dict(sample.evaluation_metrics or {}),
            mapping,
        )
        for tile in sample.tiles:
            tile.class_object_counts = _remap_identifiers(
                dict(tile.class_object_counts or {}),
                mapping,
            )
            tile.evaluation_metrics = _remap_identifiers(
                dict(tile.evaluation_metrics or {}),
                mapping,
            )
        sample_root = Path(config.stored_files_root) / "test-samples" / str(sample.id)
        if sample_root.is_dir():
            for path in sample_root.glob("tile_*.geojson"):
                _rewrite_identifier_geojson(path, mapping)

    if result_ids:
        for metric in session.scalars(
            select(TrainingResultTestMetricRow).where(
                TrainingResultTestMetricRow.training_result_id.in_(result_ids)
            )
        ).all():
            metric.metrics = _remap_identifiers(
                dict(metric.metrics or {}),
                mapping_by_result.get(metric.training_result_id, identifier_mapping),
            )
        for pseudo in session.scalars(
            select(PseudoMarkupResultRow).where(
                PseudoMarkupResultRow.training_result_id.in_(result_ids),
                PseudoMarkupResultRow.geojson_file_id.is_not(None),
            )
        ).all():
            stored = session.get(StoredFileRow, pseudo.geojson_file_id)
            if stored is None:
                continue
            _rewrite_identifier_geojson(
                Path(stored.path),
                mapping_by_result.get(pseudo.training_result_id, identifier_mapping),
            )
            path = Path(stored.path)
            if path.is_file():
                stored.size_bytes = path.stat().st_size


def _canonical_class_schema(
    values: object,
    canonical_schema: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, str]]:
    old_values = values if isinstance(values, list) else []
    old_by_id = {
        int(item.get("id", item.get("class_id", index))): item
        for index, item in enumerate(old_values, start=1)
        if isinstance(item, dict)
    }
    mapping: dict[str, str] = {}
    result: list[dict[str, object]] = []
    for canonical in canonical_schema:
        class_id = int(canonical["id"])
        old = dict(old_by_id.get(class_id) or {})
        old_slug = str(old.get("slug") or "")
        new_slug = str(canonical["slug"])
        if old_slug and old_slug != new_slug:
            mapping[old_slug] = new_slug
        old.update(canonical)
        result.append(old)
    return result, mapping


_CLASS_SCHEMA_FIELDS = frozenset({"class_schema", "object_types"})


def _is_class_schema_field(key: object) -> bool:
    return str(key).rsplit(".", maxsplit=1)[-1] in _CLASS_SCHEMA_FIELDS


def _nested_class_schema_mapping(
    value: object,
    canonical_schema: list[dict[str, object]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if _is_class_schema_field(key) and isinstance(item, list):
                _schema, item_mapping = _canonical_class_schema(item, canonical_schema)
                mapping.update(item_mapping)
            else:
                mapping.update(_nested_class_schema_mapping(item, canonical_schema))
    elif isinstance(value, list):
        for item in value:
            mapping.update(_nested_class_schema_mapping(item, canonical_schema))
    return mapping


def _canonicalize_nested_class_schemas(
    value,
    canonical_schema: list[dict[str, object]],
):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if _is_class_schema_field(key) and isinstance(item, list):
                result[key], _mapping = _canonical_class_schema(item, canonical_schema)
            else:
                result[key] = _canonicalize_nested_class_schemas(item, canonical_schema)
        return result
    if isinstance(value, list):
        return [_canonicalize_nested_class_schemas(item, canonical_schema) for item in value]
    return value


def _remap_identifiers(value, mapping: dict[str, str]):
    if not mapping:
        return value
    if isinstance(value, dict):
        return {
            mapping.get(str(key), str(key)): _remap_identifiers(item, mapping)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap_identifiers(item, mapping) for item in value]
    if isinstance(value, str):
        return mapping.get(value, value)
    return value


def _rewrite_identifier_geojson(path: Path, mapping: dict[str, str]) -> None:
    if not mapping or not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingUIAPIError(
            f"Не удалось обновить технические идентификаторы в {path.name}: {exc}"
        ) from exc
    remapped = _remap_identifiers(payload, mapping)
    if remapped == payload:
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(remapped, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_dataset_name_available(
    session: Session,
    class_id: uuid.UUID,
    name: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> None:
    statement = select(DatasetRow).where(
        DatasetRow.class_id == class_id,
        DatasetRow.name == name,
    )
    if exclude_id is not None:
        statement = statement.where(DatasetRow.id != exclude_id)
    if session.scalar(statement) is not None:
        raise TrainingUIAPIError(f"Датасет с названием «{name}» уже существует в классе")


def _class_row(session: Session, key: str) -> DatasetClassRow:
    row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.key == key))
    if row is None:
        raise TrainingUIAPIError(f"Класс не найден: {key}")
    return row


def _dataset_row(session: Session, key: str) -> DatasetRow:
    row = session.scalar(
        select(DatasetRow).where(
            DatasetRow.key == key,
            DatasetRow.deleted_at.is_(None),
        )
    )
    if row is None:
        raise TrainingUIAPIError(f"Датасет не найден: {key}")
    return row


def _dataset_key(session: Session, dataset_id: uuid.UUID | None) -> str | None:
    if dataset_id is None:
        return None
    return session.scalar(select(DatasetRow.key).where(DatasetRow.id == dataset_id))


def _clean_name(value: str, field_name: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise TrainingUIAPIError(f"{field_name} не может быть пустым")
    return normalized


def _unique_dataset_key(session: Session, preferred: str) -> str:
    if session.scalar(select(DatasetRow.id).where(DatasetRow.key == preferred)) is None:
        return preferred
    return str(uuid.uuid4())


def _unique_dataset_name(session: Session, class_id: uuid.UUID, preferred: str) -> str:
    base = preferred or DEFAULT_DATASET_NAME
    name = base
    suffix = 2
    while session.scalar(
        select(DatasetRow).where(
            DatasetRow.class_id == class_id,
            DatasetRow.name == name,
        )
    ) is not None:
        name = f"{base} (MLMarkup)" if suffix == 2 else f"{base} (MLMarkup) {suffix}"
        suffix += 1
    return name


def _safe_relative_candidate(*parts: str) -> str:
    safe_parts = [part.strip().replace("\\", "/").strip("/") for part in parts]
    candidate = PurePosixPath(*safe_parts)
    if candidate.is_absolute() or ".." in candidate.parts:
        return "/".join(part.replace("..", "_") for part in safe_parts)
    return candidate.as_posix()


def _latest_dataset_update(datasets: list[DatasetInfo]):
    values = [item.updated_at for item in datasets if item.updated_at is not None]
    return max(values) if values else None


def _recursive_raster_count(path: Path) -> int:
    if not Path(path).is_dir():
        return 0
    return sum(
        1
        for item in Path(path).rglob("*")
        if item.is_file() and item.suffix.casefold() in RASTER_SUFFIXES
    )


__all__ = [
    "create_dataset_class",
    "create_managed_dataset",
    "find_managed_class",
    "find_managed_dataset",
    "list_imagery_types",
    "list_managed_classes",
    "list_managed_datasets",
    "managed_dataset_catalog",
    "set_primary_dataset",
    "synchronize_dataset_catalog",
    "update_dataset_class",
    "update_managed_dataset",
]

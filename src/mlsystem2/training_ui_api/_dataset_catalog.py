"""Управляемый каталог классов, подклассов и датасетов."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._datasets import (
    CUSTOM_KEY,
    CUSTOM_NAME,
    DEFAULT_VARIANT,
    RASTER_SUFFIXES,
    _annotation_files,
    _dataset_image_count,
    _first_file,
    _image_index,
    _path_metadata,
    resolve_scenes_file_images,
)
from ._models import (
    AutomationRuleRow,
    DatasetClassRow,
    DatasetRow,
    DatasetSubclassRow,
    InferenceTemplateRow,
    JobRow,
    PseudoMarkupResultRow,
    TestSampleBatchItemRow,
    TestSampleRow,
    TrainingResultRow,
    TrainingTemplateRow,
)
from .contracts import (
    ClassInfo,
    DatasetCatalogInfo,
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetInfo,
    DatasetPrimarySubclassUpdate,
    DatasetSourceInfo,
    DatasetSubclassCreate,
    DatasetSubclassInfo,
    DatasetSubclassUpdate,
    ImageTypeInfo,
    ManagedDatasetCreate,
    ManagedDatasetUpdate,
    TrainingUIAPIError,
)


SOURCE_MLMARKUP = "mlmarkup"
IMAGE_TYPE_ALL = "all"
QUALITY_PIXEL = "pixel"
QUALITY_OBJECTS = "objects"
_SYNC_LOCK = threading.RLock()


def synchronize_dataset_catalog(session: Session, config: TrainingUIAPIConfig) -> None:
    """Идемпотентно импортирует историю и новые папки MLMarkup."""

    with _SYNC_LOCK:
        initial_import = session.scalar(select(DatasetRow.id).limit(1)) is None
        _import_historical_dataset_keys(session, config)
        _import_mlmarkup_folders(session, config, preserve_legacy_keys=initial_import)
        session.flush()


def list_managed_datasets(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    include_custom: bool = True,
) -> list[DatasetInfo]:
    synchronize_dataset_catalog(session, config)
    rows = session.execute(
        select(DatasetRow, DatasetSubclassRow, DatasetClassRow)
        .join(DatasetSubclassRow, DatasetSubclassRow.id == DatasetRow.subclass_id)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetSubclassRow.class_id)
    ).all()
    image_indexes: dict[Path, dict[str, list[Path]]] = {}
    datasets = [
        _dataset_info(
            dataset_row,
            subclass_row,
            class_row,
            config,
            image_indexes=image_indexes,
        )
        for dataset_row, subclass_row, class_row in rows
    ]
    datasets.sort(
        key=lambda item: (
            (item.class_name or "").casefold(),
            not item.is_primary,
            (item.variant_name or "").casefold(),
        )
    )
    if include_custom:
        datasets.append(
            DatasetInfo(
                key=CUSTOM_KEY,
                name=CUSTOM_NAME,
                is_custom=True,
                source_available=True,
            )
        )
    return datasets


def list_managed_classes(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    include_custom: bool = True,
) -> list[ClassInfo]:
    datasets = list_managed_datasets(session, config, include_custom=False)
    class_rows = session.scalars(select(DatasetClassRow)).all()
    subclass_rows = session.scalars(select(DatasetSubclassRow)).all()
    by_class: dict[str, list[DatasetInfo]] = {}
    dataset_by_subclass = {
        item.subclass_key: item for item in datasets if item.subclass_key is not None
    }
    subclasses_by_class: dict[uuid.UUID, list[DatasetSubclassRow]] = {}
    for subclass in subclass_rows:
        subclasses_by_class.setdefault(subclass.class_id, []).append(subclass)
    for dataset in datasets:
        if dataset.class_key is not None:
            by_class.setdefault(dataset.class_key, []).append(dataset)
    classes = [
        ClassInfo(
            key=row.key,
            name=row.name,
            updated_at=_latest_dataset_update(by_class.get(row.key, [])),
            variants=by_class.get(row.key, []),
            subclasses=[
                DatasetSubclassInfo(
                    key=subclass.key,
                    name=subclass.name,
                    is_primary=row.primary_subclass_id == subclass.id,
                    dataset=dataset_by_subclass.get(subclass.key),
                )
                for subclass in sorted(
                    subclasses_by_class.get(row.id, []),
                    key=lambda item: (
                        row.primary_subclass_id != item.id,
                        item.name.casefold(),
                    ),
                )
            ],
            quality_metric=row.quality_metric,
            primary_subclass_key=_subclass_key(session, row.primary_subclass_id),
        )
        for row in class_rows
    ]
    classes.sort(key=lambda item: item.name.casefold())
    if include_custom:
        custom_dataset = DatasetInfo(
            key=CUSTOM_KEY,
            name=CUSTOM_NAME,
            is_custom=True,
            source_available=True,
        )
        classes.append(
            ClassInfo(
                key=CUSTOM_KEY,
                name=CUSTOM_NAME,
                variants=[custom_dataset],
                is_custom=True,
            )
        )
    return classes


def find_managed_dataset(
    session: Session,
    config: TrainingUIAPIConfig,
    dataset_key: str,
) -> DatasetInfo | None:
    if dataset_key == CUSTOM_KEY:
        return DatasetInfo(key=CUSTOM_KEY, name=CUSTOM_NAME, is_custom=True)
    synchronize_dataset_catalog(session, config)
    row = session.execute(
        select(DatasetRow, DatasetSubclassRow, DatasetClassRow)
        .join(DatasetSubclassRow, DatasetSubclassRow.id == DatasetRow.subclass_id)
        .join(DatasetClassRow, DatasetClassRow.id == DatasetSubclassRow.class_id)
        .where(DatasetRow.key == dataset_key)
    ).one_or_none()
    if row is None:
        return None
    return _dataset_info(*row, config)


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
    synchronize_dataset_catalog(session, config)
    return DatasetCatalogInfo(
        classes=list_managed_classes(session, config, include_custom=False),
        sources=_source_infos(session, config),
        image_types=list_image_types(config.images_root),
    )


def create_dataset_class(
    session: Session,
    request: DatasetClassCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    name = _clean_name(request.name, "Название класса")
    _ensure_class_name_available(session, name)
    session.add(
        DatasetClassRow(
            key=str(uuid.uuid4()),
            name=name,
            quality_metric=QUALITY_PIXEL,
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
    if request.quality_metric is not None and request.quality_metric.value != row.quality_metric:
        row.quality_metric = request.quality_metric.value
        dataset_rows = session.scalars(
            select(DatasetRow)
            .join(DatasetSubclassRow, DatasetSubclassRow.id == DatasetRow.subclass_id)
            .where(DatasetSubclassRow.class_id == row.id)
        ).all()
        for dataset in dataset_rows:
            dataset.config_revision += 1
            dataset.legacy_version = False
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


def set_primary_subclass(
    session: Session,
    class_key: str,
    request: DatasetPrimarySubclassUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    class_row = _class_row(session, class_key)
    subclass = _subclass_row(session, request.subclass_key)
    if subclass.class_id != class_row.id:
        raise TrainingUIAPIError("Подкласс не принадлежит выбранному классу")
    if session.scalar(select(DatasetRow).where(DatasetRow.subclass_id == subclass.id)) is None:
        raise TrainingUIAPIError("Основным можно назначить только подкласс с датасетом")
    class_row.primary_subclass_id = subclass.id
    class_row.primary_subclass_locked = True
    session.flush()
    return managed_dataset_catalog(session, config)


def create_dataset_subclass(
    session: Session,
    request: DatasetSubclassCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    class_row = _class_row(session, request.class_key)
    name = _clean_name(request.name, "Название подкласса")
    _ensure_subclass_name_available(session, class_row.id, name)
    session.add(
        DatasetSubclassRow(
            key=str(uuid.uuid4()),
            class_id=class_row.id,
            name=name,
        )
    )
    session.flush()
    return managed_dataset_catalog(session, config)


def update_dataset_subclass(
    session: Session,
    subclass_key: str,
    request: DatasetSubclassUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    row = _subclass_row(session, subclass_key)
    name = _clean_name(request.name, "Название подкласса")
    _ensure_subclass_name_available(session, row.class_id, name, exclude_id=row.id)
    row.name = name
    session.flush()
    return managed_dataset_catalog(session, config)


def create_managed_dataset(
    session: Session,
    request: ManagedDatasetCreate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    subclass = _subclass_row(session, request.subclass_key)
    if session.scalar(select(DatasetRow).where(DatasetRow.subclass_id == subclass.id)) is not None:
        raise TrainingUIAPIError("У подкласса уже есть датасет")
    source_path = _validate_source_path(config.mlmarkup_root, request.source_path, require_exists=True)
    image_type = _validate_image_type(config.images_root, request.image_type)
    source_owner = session.scalar(
        select(DatasetRow).where(
            DatasetRow.source_type == SOURCE_MLMARKUP,
            DatasetRow.source_path == source_path,
        )
    )
    if source_owner is None:
        session.add(
            DatasetRow(
                key=str(uuid.uuid4()),
                subclass_id=subclass.id,
                source_type=SOURCE_MLMARKUP,
                source_path=source_path,
                image_type=image_type,
                config_revision=1,
                legacy_version=False,
            )
        )
    else:
        previous_subclass = session.get(DatasetSubclassRow, source_owner.subclass_id)
        previous_class = (
            session.get(DatasetClassRow, previous_subclass.class_id)
            if previous_subclass is not None
            else None
        )
        if previous_class is not None and previous_class.primary_subclass_id == source_owner.subclass_id:
            previous_class.primary_subclass_id = None
        source_owner.subclass_id = subclass.id
        source_owner.image_type = image_type
        source_owner.config_revision += 1
        source_owner.legacy_version = False
    session.flush()
    target_class = session.get(DatasetClassRow, subclass.class_id)
    if target_class is not None:
        _assign_main_if_available(target_class, subclass)
    return managed_dataset_catalog(session, config)


def update_managed_dataset(
    session: Session,
    dataset_key: str,
    request: ManagedDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    row = session.scalar(select(DatasetRow).where(DatasetRow.key == dataset_key))
    if row is None:
        raise TrainingUIAPIError(f"Датасет не найден: {dataset_key}")
    source_path = _validate_source_path(config.mlmarkup_root, request.source_path, require_exists=True)
    image_type = _validate_image_type(config.images_root, request.image_type)
    if source_path != row.source_path:
        source_owner = session.scalar(
            select(DatasetRow).where(
                DatasetRow.source_type == row.source_type,
                DatasetRow.source_path == source_path,
                DatasetRow.id != row.id,
            )
        )
        previous_source_path = row.source_path
        if source_owner is not None:
            source_owner.source_path = f".mlsystem2-source-swap/{uuid.uuid4()}"
            session.flush()
        row.source_path = source_path
        row.image_type = image_type
        row.config_revision += 1
        row.legacy_version = False
        session.flush()
        if source_owner is not None:
            source_owner.source_path = previous_source_path
            source_owner.config_revision += 1
            source_owner.legacy_version = False
    elif image_type != row.image_type:
        row.image_type = image_type
        row.config_revision += 1
        row.legacy_version = False
    session.flush()
    return managed_dataset_catalog(session, config)


def list_image_types(images_root: Path) -> list[ImageTypeInfo]:
    root = Path(images_root).resolve()
    if not root.exists() or not root.is_dir():
        return [ImageTypeInfo(key=IMAGE_TYPE_ALL, name="Все снимки", path=str(root), image_count=0)]
    result = [
        ImageTypeInfo(
            key=IMAGE_TYPE_ALL,
            name="Все снимки",
            path=str(root),
            image_count=_recursive_raster_count(root),
        )
    ]
    for directory in sorted(
        (
            item
            for item in root.iterdir()
            if item.is_dir()
            and not item.name.startswith(".")
            and _is_within_root(item.resolve(), root)
        ),
        key=lambda item: item.name.casefold(),
    ):
        result.append(
            ImageTypeInfo(
                key=directory.name,
                name=directory.name,
                path=str(directory),
                image_count=_recursive_raster_count(directory),
            )
        )
    return result


def _import_historical_dataset_keys(session: Session, config: TrainingUIAPIConfig) -> None:
    existing = set(session.scalars(select(DatasetRow.key)).all())
    for dataset_key in sorted(_historical_dataset_keys(session), key=str.casefold):
        if dataset_key in existing:
            continue
        class_name, subclass_name = _split_legacy_dataset_key(dataset_key)
        class_row = _ensure_class(session, class_name, preserve_legacy_key=True)
        source_path = _legacy_source_path(config.mlmarkup_root, class_name, subclass_name)
        source_owner = session.scalar(
            select(DatasetRow).where(
                DatasetRow.source_type == SOURCE_MLMARKUP,
                DatasetRow.source_path == source_path,
            )
        )
        if source_owner is not None:
            continue
        subclass = _ensure_subclass_for_dataset(session, class_row, subclass_name, dataset_key)
        dataset = DatasetRow(
            key=dataset_key,
            subclass_id=subclass.id,
            source_type=SOURCE_MLMARKUP,
            source_path=source_path,
            image_type=_infer_image_type(config, source_path),
            config_revision=1,
            legacy_version=True,
        )
        session.add(dataset)
        session.flush()
        _assign_main_if_available(class_row, subclass)
        existing.add(dataset_key)


def _import_mlmarkup_folders(
    session: Session,
    config: TrainingUIAPIConfig,
    *,
    preserve_legacy_keys: bool,
) -> None:
    assigned_rows = session.execute(
        select(DatasetRow, DatasetSubclassRow)
        .join(DatasetSubclassRow, DatasetSubclassRow.id == DatasetRow.subclass_id)
        .where(DatasetRow.source_type == SOURCE_MLMARKUP)
    ).all()
    assigned_sources = {row.source_path for row, _subclass in assigned_rows}
    source_root_classes: dict[str, set[uuid.UUID]] = {}
    for row, subclass in assigned_rows:
        source_parts = PurePosixPath(row.source_path).parts
        if source_parts:
            source_root_classes.setdefault(source_parts[0], set()).add(subclass.class_id)
    for class_name, subclass_name, source_path in _discover_mlmarkup_sources(config.mlmarkup_root):
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
        )
        if source_path is None:
            continue
        preferred_key = f"{class_name}\\{subclass_name}"
        subclass = _ensure_subclass_for_dataset(
            session,
            class_row,
            subclass_name,
            preferred_key,
        )
        dataset_key = (
            _unique_dataset_key(session, preferred_key)
            if preserve_legacy_keys
            else str(uuid.uuid4())
        )
        session.add(
            DatasetRow(
                key=dataset_key,
                subclass_id=subclass.id,
                source_type=SOURCE_MLMARKUP,
                source_path=source_path,
                image_type=_infer_image_type(config, source_path),
                config_revision=1,
                legacy_version=True,
            )
        )
        session.flush()
        _assign_main_if_available(class_row, subclass)
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
                (class_dir.name, DEFAULT_VARIANT, class_dir.relative_to(root).as_posix())
            )
            continue
        variant_dirs = sorted(
            (
                item
                for item in class_dir.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            ),
            key=lambda item: (item.name != DEFAULT_VARIANT, item.name.casefold()),
        )
        if variant_dirs:
            for variant_dir in variant_dirs:
                discovered.append(
                    (
                        class_dir.name,
                        variant_dir.name,
                        variant_dir.relative_to(root).as_posix(),
                    )
                )
        elif _first_file(class_dir, ".txt") is not None or _first_file(class_dir, ".geojson") is not None:
            discovered.append(
                (class_dir.name, DEFAULT_VARIANT, class_dir.relative_to(root).as_posix())
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
    dataset: DatasetRow,
    subclass: DatasetSubclassRow,
    class_row: DatasetClassRow,
    config: TrainingUIAPIConfig,
    *,
    image_indexes: dict[Path, dict[str, list[Path]]] | None = None,
) -> DatasetInfo:
    source_path = _resolved_source_path(config.mlmarkup_root, dataset.source_path)
    images_dir = _images_dir(config.images_root, dataset.image_type)
    source_inside_root = _is_within_root(source_path, Path(config.mlmarkup_root).resolve())
    images_inside_root = _is_within_root(images_dir, Path(config.images_root).resolve())
    diagnostics: list[str] = []
    scenes_file: Path | None = None
    annotation_file: Path | None = None
    hard_negative_file: Path | None = None
    updated_at = None
    source_version = None
    source_available = source_inside_root and source_path.is_dir()
    if not source_inside_root:
        diagnostics.append(
            f"Источник MLMarkup выходит за пределы разрешённого каталога: {dataset.source_path}"
        )
    elif not source_available:
        diagnostics.append(f"Источник MLMarkup недоступен: {dataset.source_path}")
    else:
        scenes_file = _first_file(source_path, ".txt")
        annotation_file, hard_negative_file, source_diagnostics = _annotation_files(source_path)
        diagnostics.extend(source_diagnostics)
        if scenes_file is None:
            diagnostics.append("В источнике датасета не найден TXT со списком сцен.")
        if annotation_file is None and not source_diagnostics:
            diagnostics.append("В источнике датасета не найден positive GeoJSON.")
        updated_at, source_version = _path_metadata(source_path, config.mlmarkup_root)
    if not images_inside_root:
        diagnostics.append(f"Тип снимков выходит за пределы MLSYSTEM2_IMAGES_ROOT: {dataset.image_type}")
    elif not images_dir.is_dir():
        diagnostics.append(f"Тип снимков недоступен: {dataset.image_type}")
    version = source_version
    if not dataset.legacy_version:
        version = f"managed:{dataset.config_revision}:{source_version or 'missing'}"
    index = None
    if scenes_file is not None and images_inside_root and images_dir.is_dir():
        if image_indexes is None:
            index = _image_index(images_dir)
        else:
            index = image_indexes.get(images_dir)
            if index is None:
                index = _image_index(images_dir)
                image_indexes[images_dir] = index
    display_name = f"{class_row.name}\\{subclass.name}"
    return DatasetInfo(
        key=dataset.key,
        name=display_name,
        class_key=class_row.key,
        class_name=class_row.name,
        subclass_key=subclass.key,
        variant_key=subclass.name,
        variant_name=subclass.name,
        path=str(source_path),
        scenes_file=str(scenes_file) if scenes_file is not None else None,
        annotation_file=str(annotation_file) if annotation_file is not None else None,
        hard_negative_annotation_file=str(hard_negative_file) if hard_negative_file is not None else None,
        image_count=_dataset_image_count(scenes_file, index),
        version=version,
        updated_at=updated_at,
        quality_metric=class_row.quality_metric,
        image_type=dataset.image_type,
        images_dir=str(images_dir) if images_inside_root else None,
        source_type=dataset.source_type,
        source_path=dataset.source_path,
        source_available=source_available,
        is_primary=class_row.primary_subclass_id == subclass.id,
        diagnostics=diagnostics,
    )


def _source_infos(session: Session, config: TrainingUIAPIConfig) -> list[DatasetSourceInfo]:
    assigned = {
        row.source_path: row.key
        for row in session.scalars(
            select(DatasetRow).where(DatasetRow.source_type == SOURCE_MLMARKUP)
        ).all()
    }
    result: list[DatasetSourceInfo] = []
    for class_name, subclass_name, source_path in _discover_mlmarkup_sources(config.mlmarkup_root):
        if source_path is None:
            continue
        absolute = _resolved_source_path(config.mlmarkup_root, source_path)
        if not _is_within_root(absolute, Path(config.mlmarkup_root).resolve()):
            diagnostics = ["Источник выходит за пределы разрешённого каталога MLMarkup."]
        else:
            annotation, _hard_negative, diagnostics = _annotation_files(absolute)
            if _first_file(absolute, ".txt") is None:
                diagnostics.append("Не найден TXT со списком сцен.")
            if annotation is None and not any("positive GeoJSON" in item for item in diagnostics):
                diagnostics.append("Не найден positive GeoJSON.")
        result.append(
            DatasetSourceInfo(
                key=source_path,
                name=f"{class_name}\\{subclass_name}",
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
    row = DatasetClassRow(key=key, name=name, quality_metric=QUALITY_PIXEL)
    session.add(row)
    session.flush()
    return row


def _ensure_subclass_for_dataset(
    session: Session,
    class_row: DatasetClassRow,
    name: str,
    dataset_key: str,
) -> DatasetSubclassRow:
    normalized_name = name or DEFAULT_VARIANT
    candidates = session.scalars(
        select(DatasetSubclassRow).where(
            DatasetSubclassRow.class_id == class_row.id,
            DatasetSubclassRow.name == normalized_name,
        )
    ).all()
    for candidate in candidates:
        owner = session.scalar(select(DatasetRow).where(DatasetRow.subclass_id == candidate.id))
        if owner is None or owner.key == dataset_key:
            return candidate
    if candidates:
        normalized_name = _unique_subclass_name(session, class_row.id, f"{normalized_name} (MLMarkup)")
    row = DatasetSubclassRow(
        key=str(uuid.uuid4()),
        class_id=class_row.id,
        name=normalized_name,
    )
    session.add(row)
    session.flush()
    return row


def _assign_main_if_available(class_row: DatasetClassRow, subclass: DatasetSubclassRow) -> None:
    if (
        subclass.name.casefold() == DEFAULT_VARIANT
        and class_row.primary_subclass_id is None
        and not class_row.primary_subclass_locked
    ):
        class_row.primary_subclass_id = subclass.id


def _split_legacy_dataset_key(value: str) -> tuple[str, str]:
    normalized = value.replace("/", "\\").strip("\\")
    if "\\" not in normalized:
        return normalized, DEFAULT_VARIANT
    class_name, subclass_name = normalized.rsplit("\\", 1)
    return class_name or normalized, subclass_name or DEFAULT_VARIANT


def _legacy_source_path(root: Path, class_name: str, subclass_name: str) -> str:
    nested = _safe_relative_candidate(class_name, subclass_name)
    nested_path = _resolved_source_path(root, nested)
    resolved_root = Path(root).resolve()
    if _is_within_root(nested_path, resolved_root) and nested_path.is_dir():
        return nested
    if subclass_name.casefold() == DEFAULT_VARIANT:
        flat = _safe_relative_candidate(class_name)
        flat_path = _resolved_source_path(root, flat)
        if _is_within_root(flat_path, resolved_root) and flat_path.is_dir():
            return flat
    return nested


def _infer_image_type(config: TrainingUIAPIConfig, source_path: str) -> str:
    source = _resolved_source_path(config.mlmarkup_root, source_path)
    if not _is_within_root(source, Path(config.mlmarkup_root).resolve()):
        return IMAGE_TYPE_ALL
    scenes = _first_file(source, ".txt") if source.is_dir() else None
    if scenes is None:
        return IMAGE_TYPE_ALL
    images = resolve_scenes_file_images(scenes, config.images_root)
    top_levels: set[str] = set()
    root = config.images_root.resolve()
    for image in images:
        try:
            relative = image.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if len(relative.parts) > 1:
            top_levels.add(relative.parts[0])
        else:
            return IMAGE_TYPE_ALL
    return next(iter(top_levels)) if len(top_levels) == 1 else IMAGE_TYPE_ALL


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


def _validate_image_type(root: Path, value: str) -> str:
    normalized = value.strip()
    allowed = {item.key for item in list_image_types(root)}
    if normalized not in allowed:
        raise TrainingUIAPIError(f"Тип снимков не найден: {normalized}")
    return normalized


def _images_dir(root: Path, image_type: str) -> Path:
    return Path(root).resolve() if image_type == IMAGE_TYPE_ALL else (Path(root) / image_type).resolve()


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


def _ensure_subclass_name_available(
    session: Session,
    class_id: uuid.UUID,
    name: str,
    *,
    exclude_id: uuid.UUID | None = None,
) -> None:
    statement = select(DatasetSubclassRow).where(
        DatasetSubclassRow.class_id == class_id,
        DatasetSubclassRow.name == name,
    )
    if exclude_id is not None:
        statement = statement.where(DatasetSubclassRow.id != exclude_id)
    if session.scalar(statement) is not None:
        raise TrainingUIAPIError(f"Подкласс с названием «{name}» уже существует")


def _class_row(session: Session, key: str) -> DatasetClassRow:
    row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.key == key))
    if row is None:
        raise TrainingUIAPIError(f"Класс не найден: {key}")
    return row


def _subclass_row(session: Session, key: str) -> DatasetSubclassRow:
    row = session.scalar(select(DatasetSubclassRow).where(DatasetSubclassRow.key == key))
    if row is None:
        raise TrainingUIAPIError(f"Подкласс не найден: {key}")
    return row


def _subclass_key(session: Session, subclass_id: uuid.UUID | None) -> str | None:
    if subclass_id is None:
        return None
    return session.scalar(select(DatasetSubclassRow.key).where(DatasetSubclassRow.id == subclass_id))


def _clean_name(value: str, field_name: str) -> str:
    normalized = " ".join(value.split())
    if not normalized:
        raise TrainingUIAPIError(f"{field_name} не может быть пустым")
    return normalized


def _unique_dataset_key(session: Session, preferred: str) -> str:
    if session.scalar(select(DatasetRow.id).where(DatasetRow.key == preferred)) is None:
        return preferred
    return str(uuid.uuid4())


def _unique_subclass_name(session: Session, class_id: uuid.UUID, preferred: str) -> str:
    name = preferred
    suffix = 2
    while session.scalar(
        select(DatasetSubclassRow).where(
            DatasetSubclassRow.class_id == class_id,
            DatasetSubclassRow.name == name,
        )
    ) is not None:
        name = f"{preferred} {suffix}"
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
    return sum(
        1
        for item in Path(path).rglob("*")
        if item.is_file() and item.suffix.casefold() in RASTER_SUFFIXES
    )


__all__ = [
    "create_dataset_class",
    "create_dataset_subclass",
    "create_managed_dataset",
    "find_managed_class",
    "find_managed_dataset",
    "list_image_types",
    "list_managed_classes",
    "list_managed_datasets",
    "managed_dataset_catalog",
    "set_primary_subclass",
    "synchronize_dataset_catalog",
    "update_dataset_class",
    "update_dataset_subclass",
    "update_managed_dataset",
]

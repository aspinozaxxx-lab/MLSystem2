"""Управляемый каталог классов и их датасетов."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import load_dataset_manifest, resolve_scene_images
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
    imagery_images_dir,
    resolve_scenes_file_images,
)
from ._models import (
    AutomationRuleRow,
    DatasetClassRow,
    DatasetRow,
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
    DatasetFormat,
    DatasetInfo,
    DatasetObjectType,
    DatasetPrimaryDatasetUpdate,
    DatasetSourceInfo,
    ImageryTypeInfo,
    ManagedDatasetCreate,
    ManagedDatasetUpdate,
    TrainingUIAPIError,
)


SOURCE_MLMARKUP = "mlmarkup"
DEFAULT_IMAGERY_TYPE = "kanopus"
IMAGERY_NAMES = {"kanopus": "Канопус", "ortho": "Ортофото"}
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


def primary_training_result(
    session: Session,
    class_or_dataset_key: str,
) -> TrainingResultRow | None:
    """Вернуть явно выбранную основную сеть или совместимый последний результат."""

    class_row = dataset_class_row(session, class_or_dataset_key)
    if class_row is not None and class_row.primary_training_result_id is not None:
        return session.get(TrainingResultRow, class_row.primary_training_result_id)
    dataset_keys = (
        session.scalars(select(DatasetRow.key).where(DatasetRow.class_id == class_row.id)).all()
        if class_row is not None
        else [class_or_dataset_key]
    )
    return session.scalar(
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
        .limit(1)
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
    synchronize_dataset_catalog(session, config)
    rows = session.execute(
        select(DatasetRow, DatasetClassRow).join(
            DatasetClassRow,
            DatasetClassRow.id == DatasetRow.class_id,
        ).where(DatasetRow.deleted_at.is_(None))
    ).all()
    image_indexes: dict[Path, dict[str, list[Path]]] = {}
    datasets = [
        _dataset_info(dataset, class_row, config, image_indexes=image_indexes)
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
) -> list[ClassInfo]:
    datasets = list_managed_datasets(session, config, include_custom=False)
    class_rows = session.scalars(select(DatasetClassRow)).all()
    by_class: dict[str, list[DatasetInfo]] = {}
    for dataset in datasets:
        if dataset.class_key is not None:
            by_class.setdefault(dataset.class_key, []).append(dataset)
    classes = [
        ClassInfo(
            key=row.key,
            name=row.name,
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
    synchronize_dataset_catalog(session, config)
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
        imagery_types=list_imagery_types(config.images_root),
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


def update_managed_dataset(
    session: Session,
    dataset_key: str,
    request: ManagedDatasetUpdate,
    config: TrainingUIAPIConfig,
) -> DatasetCatalogInfo:
    row = _dataset_row(session, dataset_key)
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

    changed = desired_name != row.name or desired_source != row.source_path
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
    dataset: DatasetRow,
    class_row: DatasetClassRow,
    config: TrainingUIAPIConfig,
    *,
    image_indexes: dict[Path, dict[str, list[Path]]] | None = None,
) -> DatasetInfo:
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
        image_count = _per_image_catalog_count(
            annotations_dir,
            images_dir,
            diagnostics,
        )
    display_name = f"{class_row.name}\\{dataset.name}"
    return DatasetInfo(
        key=dataset.key,
        name=display_name,
        dataset_name=dataset.name,
        class_key=class_row.key,
        class_name=class_row.name,
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
        is_custom=True,
        imagery_type=DEFAULT_IMAGERY_TYPE,
        input_channels=IMAGERY_CHANNELS[DEFAULT_IMAGERY_TYPE],
        images_dir=str(imagery_images_dir(config.images_root, DEFAULT_IMAGERY_TYPE)),
        source_available=True,
        is_primary=True,
    )


def _per_image_catalog_count(
    annotations_dir: Path,
    images_dir: Path,
    diagnostics: list[str],
) -> int:
    try:
        resolution = resolve_scene_images(
            SceneImageResolutionRequest(
                images_dir=str(images_dir),
                annotations_dir=str(annotations_dir),
            )
        )
    except (OSError, ValueError) as exc:
        diagnostics.append(f"Не удалось сопоставить per-image разметку: {exc}")
        return 0
    if resolution.missing_scenes:
        diagnostics.append(
            "Для GeoJSON не найдены TIFF: " + ", ".join(resolution.missing_scenes)
        )
    if resolution.ambiguous_scenes:
        diagnostics.append(
            "Имена GeoJSON неоднозначно сопоставлены с TIFF: "
            + ", ".join(sorted(resolution.ambiguous_scenes))
        )
    if resolution.input_scene_count == 0:
        diagnostics.append(
            "Per-image датасет пуст: его можно редактировать, но нельзя использовать для обучения."
        )
    return len(resolution.images)


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

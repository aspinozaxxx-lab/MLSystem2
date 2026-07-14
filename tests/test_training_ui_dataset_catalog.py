from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest
from sqlalchemy import func, select

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import (
    create_dataset_class,
    create_dataset_subclass,
    create_managed_dataset,
    list_managed_classes,
    list_managed_datasets,
    set_primary_subclass,
    synchronize_dataset_catalog,
    update_dataset_class,
    update_managed_dataset,
)
from mlsystem2.training_ui_api._models import (
    DatasetClassRow,
    DatasetRow,
    DatasetSubclassRow,
    TrainingResultRow,
)
from mlsystem2.training_ui_api.contracts import (
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetPrimarySubclassUpdate,
    DatasetSubclassCreate,
    ManagedDatasetCreate,
    ManagedDatasetUpdate,
    TrainingUIAPIError,
)


def test_initial_import_preserves_legacy_keys_and_missing_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Вырубки" / "main", "scene-a")
    _write_image(config.images_root / "kanopus" / "region" / "scene-a.tif")

    with session_factory() as session:
        session.add(
            TrainingResultRow(
                source="manual",
                dataset_key="Реки\\main",
                class_key="Реки\\main",
                class_display_name="Реки\\main",
                architecture="segformer_b2",
                model_name="segformer b2",
                status="ok",
            )
        )
        session.flush()
        synchronize_dataset_catalog(session, config)

        datasets = {item.key: item for item in list_managed_datasets(session, config, include_custom=False)}
        assert set(datasets) == {"Вырубки\\main", "Реки\\main"}
        assert datasets["Вырубки\\main"].image_type == "kanopus"
        assert datasets["Вырубки\\main"].is_primary is True
        assert datasets["Вырубки\\main"].version is not None
        assert datasets["Вырубки\\main"].version.startswith(("fs:", "git:"))
        assert datasets["Реки\\main"].source_available is False
        assert datasets["Реки\\main"].diagnostics
        rivers = next(item for item in list_managed_classes(session, config, include_custom=False) if item.name == "Реки")
        assert rivers.primary_subclass_key == rivers.subclasses[0].key


def test_sync_adds_new_sources_idempotently_and_keeps_missing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Лес" / "main", "scene-a")

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        _write_source(config.mlmarkup_root / "Пожары" / "summer", "scene-b")
        synchronize_dataset_catalog(session, config)
        synchronize_dataset_catalog(session, config)

        new_class = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Пожары"))
        assert new_class is not None
        assert new_class.key != "Пожары"
        uuid.UUID(new_class.key)
        new_dataset = session.scalar(
            select(DatasetRow)
            .join(DatasetSubclassRow, DatasetSubclassRow.id == DatasetRow.subclass_id)
            .where(DatasetSubclassRow.class_id == new_class.id)
        )
        assert new_dataset is not None
        assert new_dataset.key != "Пожары\\summer"
        uuid.UUID(new_dataset.key)
        assert session.scalar(select(func.count()).select_from(DatasetRow)) == 2

        old_source = config.mlmarkup_root / "Пожары" / "summer"
        renamed_source = config.mlmarkup_root / "Пожары" / "autumn"
        old_source.rename(renamed_source)
        synchronize_dataset_catalog(session, config)
        datasets = list_managed_datasets(session, config, include_custom=False)
        old_dataset = next(item for item in datasets if item.key == new_dataset.key)
        assert old_dataset.source_available is False
        assert any(item.source_path == "Пожары/autumn" for item in datasets)
        assert session.scalar(select(func.count()).select_from(DatasetRow)) == 3

        shutil.rmtree(config.mlmarkup_root / "Пожары")
        synchronize_dataset_catalog(session, config)
        assert session.scalar(select(func.count()).select_from(DatasetRow)) == 3


def test_manual_settings_survive_sync_and_metric_bumps_dataset_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Лес" / "main", "scene-a")
    _write_source(config.mlmarkup_root / "Лес" / "rare", "scene-b")
    _write_image(config.images_root / "kanopus" / "scene-a.tif")
    _write_image(config.images_root / "sentinel" / "scene-b.tif")

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        class_row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес"))
        assert class_row is not None
        subclasses = session.scalars(
            select(DatasetSubclassRow).where(DatasetSubclassRow.class_id == class_row.id)
        ).all()
        main = next(item for item in subclasses if item.name == "main")
        rare = next(item for item in subclasses if item.name == "rare")
        main_dataset = session.scalar(select(DatasetRow).where(DatasetRow.subclass_id == main.id))
        rare_dataset = session.scalar(select(DatasetRow).where(DatasetRow.subclass_id == rare.id))
        assert main_dataset is not None
        assert rare_dataset is not None
        version_before = next(
            item.version
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.key == main_dataset.key
        )

        update_dataset_class(
            session,
            class_row.key,
            DatasetClassUpdate(name="Лесной покров", quality_metric="objects"),
            config,
        )
        set_primary_subclass(
            session,
            class_row.key,
            DatasetPrimarySubclassUpdate(subclass_key=rare.key),
            config,
        )
        update_managed_dataset(
            session,
            main_dataset.key,
            ManagedDatasetUpdate(source_path="Лес/rare", image_type="all"),
            config,
        )
        _write_source(config.mlmarkup_root / "Лес" / "new", "scene-c")
        synchronize_dataset_catalog(session, config)

        session.refresh(class_row)
        session.refresh(main_dataset)
        session.refresh(rare_dataset)
        assert class_row.name == "Лесной покров"
        assert class_row.quality_metric == "objects"
        assert class_row.primary_subclass_id == rare.id
        assert main_dataset.source_path == "Лес/rare"
        assert rare_dataset.source_path == "Лес/main"
        assert main_dataset.image_type == "all"
        assert main_dataset.legacy_version is False
        assert rare_dataset.legacy_version is False
        managed = next(
            item
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.key == main_dataset.key
        )
        assert managed.version != version_before
        assert managed.version.startswith("managed:")
        assert session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес")) is None
        new_subclass = session.scalar(
            select(DatasetSubclassRow).where(
                DatasetSubclassRow.class_id == class_row.id,
                DatasetSubclassRow.name == "new",
            )
        )
        assert new_subclass is not None


def test_editor_creation_rejects_unsafe_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Свободный" / "main", "scene-a")

    with session_factory() as session:
        catalog = create_dataset_class(session, DatasetClassCreate(name="Ручной класс"), config)
        class_info = next(item for item in catalog.classes if item.name == "Ручной класс")
        catalog = create_dataset_subclass(
            session,
            DatasetSubclassCreate(class_key=class_info.key, name="вариант"),
            config,
        )
        subclass = next(
            item
            for item in next(item for item in catalog.classes if item.key == class_info.key).subclasses
            if item.name == "вариант"
        )
        with pytest.raises(TrainingUIAPIError, match="внутри MLMarkup"):
            create_managed_dataset(
                session,
                ManagedDatasetCreate(
                    subclass_key=subclass.key,
                    source_path="../outside",
                    image_type="all",
                ),
                config,
            )
        catalog = create_managed_dataset(
            session,
            ManagedDatasetCreate(
                subclass_key=subclass.key,
                source_path="Свободный/main",
                image_type="all",
            ),
            config,
        )
        assigned = next(
            item
            for item in next(item for item in catalog.classes if item.key == class_info.key).subclasses
            if item.key == subclass.key
        )
        assert assigned.dataset is not None
        assert assigned.dataset.key == "Свободный\\main"
        original_class = next(item for item in catalog.classes if item.name == "Свободный")
        assert original_class.subclasses[0].dataset is None
        assert session.scalar(select(func.count()).select_from(DatasetRow)) == 1


def test_sync_does_not_read_source_symlink_outside_mlmarkup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    outside_source = tmp_path / "outside" / "main"
    _write_source(outside_source, "scene-a")
    class_path = config.mlmarkup_root / "Внешний"
    class_path.mkdir(parents=True)
    try:
        (class_path / "main").symlink_to(outside_source, target_is_directory=True)
    except OSError:
        pytest.skip("Создание символических ссылок недоступно в текущей ОС")

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        dataset = next(
            item
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.source_path == "Внешний/main"
        )
        assert dataset.source_available is False
        assert dataset.scenes_file is None
        assert any("пределы" in diagnostic for diagnostic in dataset.diagnostics)


def _catalog_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "stored"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    return config, session_factory


def _write_source(path: Path, scene: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "scenes.txt").write_text(f"{scene}\n", encoding="utf-8")
    (path / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()

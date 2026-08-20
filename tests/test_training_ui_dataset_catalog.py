from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape
from sqlalchemy import func, select

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import (
    create_dataset_class,
    create_managed_dataset,
    create_managed_dataset_composition,
    list_managed_classes,
    list_managed_datasets,
    set_primary_dataset,
    synchronize_dataset_catalog,
    update_dataset_class,
    update_managed_dataset,
)
from mlsystem2.training_ui_api._models import (
    DatasetClassRow,
    DatasetEditorDraftRow,
    DatasetRow,
    JobRow,
    PseudoMarkupResultRow,
    StoredFileRow,
    TrainingResultRow,
)
from mlsystem2.training_ui_api.contracts import (
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetPrimaryDatasetUpdate,
    ManagedDatasetCreate,
    ManagedDatasetCompositionCreate,
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

        datasets = {
            item.key: item
            for item in list_managed_datasets(session, config, include_custom=False)
        }
        assert set(datasets) == {"Вырубки\\main", "Реки\\main"}
        assert datasets["Вырубки\\main"].dataset_name == "main"
        assert datasets["Вырубки\\main"].imagery_type == "kanopus"
        assert datasets["Вырубки\\main"].input_channels == 4
        assert datasets["Вырубки\\main"].model_name_stem == "deforestation"
        assert datasets["Вырубки\\main"].is_primary is True
        assert datasets["Вырубки\\main"].version is not None
        assert datasets["Вырубки\\main"].version.startswith(("fs:", "git:"))
        assert datasets["Реки\\main"].source_available is False
        assert datasets["Реки\\main"].model_name_stem == "rivers"
        rivers = next(
            item
            for item in list_managed_classes(session, config, include_custom=False)
            if item.name == "Реки"
        )
        assert rivers.primary_dataset_key == rivers.datasets[0].key


def test_sync_adds_sources_idempotently_and_keeps_missing_source(
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
        uuid.UUID(new_class.key)
        new_dataset = session.scalar(
            select(DatasetRow).where(DatasetRow.class_id == new_class.id)
        )
        assert new_dataset is not None
        uuid.UUID(new_dataset.key)
        assert new_dataset.name == "summer"
        assert new_dataset.model_name_stem == "annotation"
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


def test_manual_settings_and_ortho_type_survive_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Лес" / "main", "scene-a")
    _write_source(config.mlmarkup_root / "Лес" / "rare", "scene-b")
    _write_image(config.images_root / "kanopus" / "scene-a.tif")
    _write_image(config.images_root / "kanopus" / "scene-b.tif")
    (config.images_root / "orto").mkdir(parents=True)

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        class_row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес"))
        assert class_row is not None
        rows = session.scalars(select(DatasetRow).where(DatasetRow.class_id == class_row.id)).all()
        main = next(item for item in rows if item.name == "main")
        rare = next(item for item in rows if item.name == "rare")
        revisions_before = {item.id: item.config_revision for item in rows}

        update_dataset_class(
            session,
            class_row.key,
            DatasetClassUpdate(
                name="Лесной покров",
                technical_name="forest_cover",
                quality_metric="objects",
                imagery_type="ortho",
            ),
            config,
        )
        set_primary_dataset(
            session,
            class_row.key,
            DatasetPrimaryDatasetUpdate(dataset_key=rare.key),
            config,
        )
        update_managed_dataset(
            session,
            main.key,
            ManagedDatasetUpdate(name="основной", source_path="Лес/rare"),
            config,
        )
        _write_source(config.mlmarkup_root / "Лес" / "new", "scene-c")
        synchronize_dataset_catalog(session, config)

        session.refresh(class_row)
        session.refresh(main)
        session.refresh(rare)
        assert class_row.name == "Лесной покров"
        assert class_row.technical_name == "forest_cover"
        assert class_row.quality_metric == "objects"
        assert class_row.imagery_type == "ortho"
        assert class_row.primary_dataset_id == rare.id
        assert main.source_path == "Лес/rare"
        assert rare.source_path == "Лес/main"
        assert main.config_revision == revisions_before[main.id] + 2
        assert rare.config_revision == revisions_before[rare.id] + 2
        managed = next(
            item
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.key == main.key
        )
        assert managed.images_dir == str(config.images_root / "orto")
        assert managed.input_channels == 3
        assert managed.imagery_type == "ortho"
        assert session.scalar(select(DatasetClassRow).where(DatasetClassRow.name == "Лес")) is None
        assert session.scalar(
            select(DatasetRow).where(
                DatasetRow.class_id == class_row.id,
                DatasetRow.name == "new",
            )
        ) is not None


def test_editor_creates_dataset_directly_and_preserves_source_owner_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Свободный" / "main", "scene-a")

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        original = session.scalar(select(DatasetRow))
        assert original is not None
        original_key = original.key
        catalog = create_dataset_class(
            session,
            DatasetClassCreate(
                name="Ручной класс",
                technical_name="manual_class",
                imagery_type="ortho",
            ),
            config,
        )
        class_info = next(item for item in catalog.classes if item.name == "Ручной класс")
        assert class_info.technical_name == "manual_class"
        with pytest.raises(TrainingUIAPIError, match="уже используется"):
            create_dataset_class(
                session,
                DatasetClassCreate(
                    name="Ещё один класс",
                    technical_name="manual_class",
                    imagery_type="kanopus",
                ),
                config,
            )
        with pytest.raises(TrainingUIAPIError, match="внутри MLMarkup"):
            create_managed_dataset(
                session,
                ManagedDatasetCreate(
                    class_key=class_info.key,
                    name="набор",
                    source_path="../outside",
                ),
                config,
            )
        catalog = create_managed_dataset(
            session,
            ManagedDatasetCreate(
                class_key=class_info.key,
                name="набор",
                source_path="Свободный/main",
            ),
            config,
        )
        target = next(item for item in catalog.classes if item.key == class_info.key)
        assert len(target.datasets) == 1
        assert target.datasets[0].key == original_key
        assert target.datasets[0].imagery_type == "ortho"
        assert session.scalar(select(func.count()).select_from(DatasetRow)) == 1


def test_primary_dataset_must_belong_to_class(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Первый" / "main", "scene-a")
    _write_source(config.mlmarkup_root / "Второй" / "main", "scene-b")
    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        classes = list_managed_classes(session, config, include_custom=False)
        first, second = classes
        with pytest.raises(TrainingUIAPIError, match="не принадлежит"):
            set_primary_dataset(
                session,
                first.key,
                DatasetPrimaryDatasetUpdate(dataset_key=second.datasets[0].key),
                config,
            )


def test_managed_composition_is_virtual_and_follows_source_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    image = config.images_root / "kanopus" / "images" / "scene.tif"
    image.parent.mkdir(parents=True)
    with rasterio.open(
        image,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 10, 1, 1),
        nodata=0,
    ) as target:
        target.write(np.ones((1, 10, 10), dtype=np.uint8))
    first_path = config.mlmarkup_root / "Первый" / "main"
    second_path = config.mlmarkup_root / "Второй" / "main"
    annotation_name = "images_scene.geojson"
    _write_per_image_source(first_path, annotation_name, box(1, 1, 7, 7), "first")
    _write_per_image_source(second_path, annotation_name, box(5, 5, 9, 9), "second")

    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        source_classes = {
            item.name: item
            for item in list_managed_classes(session, config, include_custom=False)
        }
        target_catalog = create_dataset_class(
            session,
            DatasetClassCreate(name="Составной", imagery_type="kanopus"),
            config,
        )
        target_class = next(item for item in target_catalog.classes if item.name == "Составной")
        catalog = create_managed_dataset_composition(
            session,
            ManagedDatasetCompositionCreate(
                class_key=target_class.key,
                name="main",
                sources=[
                    {
                        "dataset_key": source_classes["Первый"].datasets[0].key,
                        "priority": 100,
                        "color": "#112233",
                    },
                    {
                        "dataset_key": source_classes["Второй"].datasets[0].key,
                        "priority": 0,
                        "color": "#445566",
                    },
                ],
            ),
            config,
        )
        managed = next(item for item in catalog.classes if item.key == target_class.key).datasets[0]
        assert managed.managed is True
        assert managed.source_type == "managed"
        assert managed.annotations_dir is not None
        assert managed.image_count == 1
        assert managed.class_counts == {
            managed.object_types[0].slug: 1,
            managed.object_types[1].slug: 1,
        }
        assert managed.version is not None and managed.version.startswith("managed:")
        assert not (config.mlmarkup_root / "Составной" / "main").exists()
        payload = json.loads(
            (Path(managed.annotations_dir) / annotation_name).read_text(encoding="utf-8")
        )
        by_class = {
            feature["properties"]["_mlsystem2_class"]: feature
            for feature in payload["features"]
        }
        assert all(
            feature["properties"].get("_mlsystem2_source_dataset_key")
            for feature in payload["features"]
        )
        first_slug = managed.object_types[0].slug
        second_slug = managed.object_types[1].slug
        assert first_slug == source_classes["Первый"].technical_name
        assert second_slug == source_classes["Второй"].technical_name
        assert shape(by_class[first_slug]["geometry"]).intersection(
            shape(by_class[second_slug]["geometry"])
        ).area == 0

        pseudo_path = tmp_path / "managed-pseudo.geojson"
        pseudo_path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "_mlsystem2_class": first_slug,
                                "object_type": first_slug,
                            },
                            "geometry": mapping(box(1, 1, 2, 2)),
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        stored = StoredFileRow(
            kind="pseudo_markup_geojson",
            original_name=pseudo_path.name,
            content_type="application/geo+json",
            path=str(pseudo_path),
            size_bytes=pseudo_path.stat().st_size,
        )
        training = TrainingResultRow(
            source="manual",
            dataset_key=managed.key,
            class_key=managed.key,
            class_display_name=managed.name,
            architecture="segformer_b2",
            model_name="combined",
            task="multiclass",
            class_schema=[item.model_dump(mode="json") for item in managed.object_types],
            training_metrics={"val_per_class_metrics": [{"slug": first_slug, "f1": 0.7}]},
            status="ok",
        )
        session.add_all([stored, training])
        session.flush()
        historical_job = JobRow(
            type="training",
            source="manual",
            status="completed",
            queue_position=1,
            dataset_key=managed.key,
            dataset_name=managed.name,
            model_name="combined",
            architecture="segformer_b2",
            config={
                "dataset.object_types": [
                    item.model_dump(mode="json") for item in managed.object_types
                ],
                "dataset": {
                    "object_types": [
                        item.model_dump(mode="json") for item in managed.object_types
                    ]
                },
                "editor_pseudo": {
                    "object_types": [
                        item.model_dump(mode="json") for item in managed.object_types
                    ]
                },
            },
        )
        session.add(historical_job)
        session.add(
            PseudoMarkupResultRow(
                dataset_key=managed.key,
                training_result_id=training.id,
                class_key=managed.key,
                source_dataset_name=managed.name,
                geojson_file_id=stored.id,
                status="ok",
            )
        )
        session.flush()

        update_dataset_class(
            session,
            source_classes["Первый"].key,
            DatasetClassUpdate(technical_name="first_objects"),
            config,
        )
        refreshed_managed = next(
            item
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.key == managed.key
        )
        assert refreshed_managed.object_types[0].slug == "first_objects"
        assert training.class_schema[0]["slug"] == "first_objects"
        assert training.training_metrics["val_per_class_metrics"][0]["slug"] == "first_objects"
        assert historical_job.config["dataset"]["object_types"][0]["slug"] == "first_objects"
        assert historical_job.config["dataset.object_types"][0]["slug"] == "first_objects"
        assert (
            historical_job.config["editor_pseudo"]["object_types"][0]["slug"]
            == "first_objects"
        )
        migrated_pseudo = json.loads(pseudo_path.read_text(encoding="utf-8"))
        assert migrated_pseudo["features"][0]["properties"] == {
            "_mlsystem2_class": "first_objects",
            "object_type": "first_objects",
        }

        original_version = managed.version
        updated_catalog = update_managed_dataset(
            session,
            managed.key,
            ManagedDatasetUpdate(
                name="обновлённый",
                sources=[
                    {
                        "dataset_key": source_classes["Первый"].datasets[0].key,
                        "priority": -10,
                        "color": "#AABBCC",
                    },
                    {
                        "dataset_key": source_classes["Второй"].datasets[0].key,
                        "priority": 200,
                        "color": "#DDEEFF",
                    },
                ],
            ),
            config,
        )
        updated = next(
            item
            for item in updated_catalog.classes
            if item.key == target_class.key
        ).datasets[0]
        assert updated.key == managed.key
        assert updated.dataset_name == "обновлённый"
        assert [item.class_name for item in updated.managed_sources] == ["Первый", "Второй"]
        assert [item.priority for item in updated.managed_sources] == [-10, 200]
        assert [item.color for item in updated.managed_sources] == ["#AABBCC", "#DDEEFF"]
        assert updated.version != original_version

        draft = DatasetEditorDraftRow(
            dataset_key=managed.key,
            annotation_name=annotation_name,
            username="картограф",
            base_revision="revision",
            geojson={"type": "FeatureCollection", "features": []},
        )
        session.add(draft)
        session.flush()
        with pytest.raises(TrainingUIAPIError, match="черновики"):
            update_managed_dataset(
                session,
                managed.key,
                ManagedDatasetUpdate(
                    sources=[
                        {
                            "dataset_key": source_classes["Первый"].datasets[0].key,
                            "priority": -10,
                            "color": "#AABBCC",
                        },
                        {
                            "dataset_key": source_classes["Второй"].datasets[0].key,
                            "priority": 200,
                            "color": "#FFFFFF",
                        },
                    ]
                ),
                config,
            )
        session.delete(draft)
        session.flush()

        old_version = updated.version
        _write_per_image_source(first_path, annotation_name, box(1, 1, 8, 8), "first")
        refreshed = next(
            item
            for item in list_managed_datasets(session, config, include_custom=False)
            if item.key == managed.key
        )
        assert refreshed.version != old_version


def _write_per_image_source(
    path: Path,
    annotation_name: str,
    geometry,
    feature_id: str,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / annotation_name).write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": [
                    {
                        "type": "Feature",
                        "id": feature_id,
                        "properties": {"_mlsystem2_role": "positive"},
                        "geometry": mapping(geometry),
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_new_ortho_source_is_detected_from_scene_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, session_factory = _catalog_environment(tmp_path, monkeypatch)
    _write_source(config.mlmarkup_root / "Крыши" / "main", "ryazan")
    _write_image(config.images_root / "orto" / "ryazan" / "ortho.tif")
    with session_factory() as session:
        synchronize_dataset_catalog(session, config)
        class_info = next(
            item
            for item in list_managed_classes(session, config, include_custom=False)
            if item.name == "Крыши"
        )
        assert class_info.imagery_type == "ortho"
        assert class_info.datasets[0].input_channels == 3
        assert class_info.datasets[0].image_count == 1


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

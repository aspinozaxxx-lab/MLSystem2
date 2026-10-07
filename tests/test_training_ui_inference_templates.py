"""Общие шаблоны, привязки классов и перенос исторических настроек."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import select

from mlsystem2.training_ui_api import _service, _test_samples
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import update_dataset_class
from mlsystem2.training_ui_api._inference_templates import migrate_inference_templates
from mlsystem2.training_ui_api._models import (
    DatasetClassRow, DatasetRow, InferenceTemplateRow, JobRow, TrainingResultRow,
)
from mlsystem2.training_ui_api._template_selection import effective_inference_template_row
from mlsystem2.training_ui_api._templates import INFERENCE_CONFIG_SCHEMA, INFERENCE_BASE_DEFAULT_CONFIG
from mlsystem2.training_ui_api.contracts import (
    DatasetClassUpdate, InferenceTemplateCreate, InferenceTemplateClassUpdate, InferenceTemplateUpdate, TrainingUIAPIError,
)


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "MLMarkup"))
    configure_schema(None)
    config = get_config()
    factory = create_session_factory(config)
    Base.metadata.create_all(factory.kw["bind"])
    with factory() as session:
        yield session, config
    factory.kw["bind"].dispose()


def _class(session, name="Реки", imagery_type="kanopus"):
    row = DatasetClassRow(key=str(uuid4()), name=name, technical_name=str(uuid4()), imagery_type=imagery_type)
    session.add(row)
    session.flush()
    return row


def _dataset(session, class_row, name="main"):
    row = DatasetRow(key=str(uuid4()), class_id=class_row.id, name=name,
                     source_type="mlmarkup", source_path=f"{class_row.name}/{name}")
    session.add(row)
    session.flush()
    return row


def _legacy(session, architecture, *, dataset=None, class_row=None, values=None, updated_at=None):
    config = {**INFERENCE_BASE_DEFAULT_CONFIG, **(values or {})}
    row = InferenceTemplateRow(
        architecture=architecture,
        dataset_key=dataset.key if dataset is not None else None,
        dataset_name=f"{class_row.name}\\{dataset.name}" if dataset is not None else None,
        display_name="Исторический шаблон",
        config_schema=deepcopy(INFERENCE_CONFIG_SCHEMA), default_config=config,
        baseline_default_config=deepcopy(config), source="manual", baseline_source="manual",
        archived_at=datetime.now(timezone.utc), is_active=True, version=7,
        updated_at=updated_at or datetime.now(timezone.utc),
    )
    session.add(row)
    session.flush()
    return row


def _template(session, config, name="Общий шаблон"):
    return _service.create_inference_template(session, InferenceTemplateCreate(display_name=name), config)


def _assign(session, config, class_row, template):
    return _service.assign_inference_template(session, class_row.key,
        InferenceTemplateClassUpdate(template_id=template.id if template else None), config)


def test_shared_config_and_class_moves_across_imagery_types(database):
    session, config = database
    first = _class(session)
    second = _class(session, "Здания", "ortho")
    one, two = _dataset(session, first), _dataset(session, second)
    shared = _template(session, config)
    other = _template(session, config, "Подробные контуры")
    _assign(session, config, first, shared)
    _assign(session, config, second, shared)
    listed = _service.inference_templates(session).templates
    assert next(item for item in listed if item.id == shared.id).class_keys == [second.key, first.key]
    _service.update_inference_template_by_id(session, shared.id,
        InferenceTemplateUpdate(default_config={"postprocess.min_area_m2": 100}), config)
    assert effective_inference_template_row(session, one.key).default_config["postprocess.min_area_m2"] == 100
    assert effective_inference_template_row(session, two.key).id == shared.id
    before = _test_samples._effective_inference_template(session, one.key, "none")[2]
    _assign(session, config, first, other)
    assert effective_inference_template_row(session, one.key).id == other.id
    assert effective_inference_template_row(session, two.key).id == shared.id
    assert _test_samples._effective_inference_template(session, one.key, "none")[2] != before
    _assign(session, config, first, None)
    assert effective_inference_template_row(session, one.key) is None


def test_metadata_validation_reset_and_delete_detaches_classes(database):
    session, config = database
    first = _class(session)
    template = _template(session, config)
    with pytest.raises(TrainingUIAPIError, match="название"):
        _service.create_inference_template(session, InferenceTemplateCreate(display_name="   "), config)
    with pytest.raises(TrainingUIAPIError, match="уже существует"):
        _template(session, config, "общий шаблон")
    _assign(session, config, first, template)
    updated = _service.update_inference_template_by_id(session, template.id,
        InferenceTemplateUpdate(display_name="Контуры", description="Для тонких объектов",
            default_config={"postprocess.simplify_m": 5}), config)
    assert updated.description == "Для тонких объектов"
    reset = _service.update_inference_template_by_id(session, template.id,
        InferenceTemplateUpdate(reset_to_baseline=True, description=None), config)
    assert reset.default_config == template.default_config
    assert reset.display_name == "Контуры" and reset.description is None
    deleted = _service.delete_inference_template(session, template.id, config)
    assert deleted.class_keys == [first.key]
    assert first.inference_template_id is None
    assert session.get(InferenceTemplateRow, template.id).archived_at is not None
    assert not _service.inference_templates(session).templates
    with pytest.raises(TrainingUIAPIError, match="недоступен"):
        _assign(session, config, first, template)
    recreated = _template(session, config, "Контуры")
    assert recreated.id != template.id


def test_new_class_without_template_and_type_change_keeps_assignment(database):
    session, config = database
    first = _class(session)
    assert effective_inference_template_row(session, first.key) is None
    template = _template(session, config)
    _assign(session, config, first, template)
    update_dataset_class(session, first.key, DatasetClassUpdate(imagery_type="ortho"), config)
    assert effective_inference_template_row(session, first.key).id == template.id
    assert effective_inference_template_row(session, first.key).default_config == template.default_config
    assert effective_inference_template_row(session, "custom") is None


def test_migration_uses_primary_network_and_preserves_job_snapshot(database):
    session, config = database
    class_row = _class(session)
    first, second = _dataset(session, class_row), _dataset(session, class_row, "test")
    _legacy(session, "smp_segformer_b0", values={"postprocess.simplify_m": 10})
    chosen = _legacy(session, "smp_segformer_b2", dataset=second, class_row=class_row,
                     values={"postprocess.min_area_m2": 123, "postprocess.simplify_m": 0.5})
    results = [TrainingResultRow(
        source="manual", class_key=ds.key, dataset_key=ds.key,
        class_display_name=ds.name, architecture=arch, model_name="Сеть", status="ok",
    ) for ds, arch in ((first, "smp_segformer_b0"), (second, "smp_segformer_b2"))]
    session.add_all(results)
    session.flush()
    class_row.primary_training_result_id = results[1].id
    snapshot = {"inference_template_id": str(chosen.id), "inference_template_version": 7,
                "inference_template_config": deepcopy(chosen.default_config)}
    job = JobRow(type="inference", source="manual", status="queued", architecture="smp_segformer_b2",
                 config=snapshot, dataset_key=second.key, queue_position=1,
                 dataset_name="Реки / Тест", model_name="Сеть", tile_size=512)
    session.add(job)
    session.flush()
    migrate_inference_templates(session)
    template = effective_inference_template_row(session, first.key)
    assert template.id != chosen.id
    assert template.default_config == chosen.default_config
    assert template.baseline_default_config == chosen.default_config
    assert chosen.archived_at is not None
    assert job.config == snapshot
    assert all(item.class_keys for item in _service.inference_templates(session).templates)
    migrate_inference_templates(session)
    assert effective_inference_template_row(session, second.key).id == template.id


def test_migration_selects_common_normalized_config_with_stable_tie(database):
    session, _ = database
    class_row = _class(session)
    dataset = _dataset(session, class_row)
    older = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for arch in ("smp_segformer_b0", "smp_segformer_b1"):
        _legacy(session, arch, dataset=dataset, class_row=class_row,
                values={"postprocess.min_area_m2": 100}, updated_at=older)
    _legacy(session, "smp_segformer_b2", dataset=dataset, class_row=class_row,
            values={"postprocess.min_area_m2": 200}, updated_at=older + timedelta(days=1))
    other_class = _class(session, "Озёра")
    other_dataset = _dataset(session, other_class)
    _legacy(session, "smp_segformer_b0", dataset=other_dataset, class_row=other_class,
            values={"postprocess.min_area_m2": 100}, updated_at=older)
    latest = _legacy(session, "smp_segformer_b1", dataset=other_dataset, class_row=other_class,
                     values={"postprocess.min_area_m2": 200}, updated_at=older + timedelta(days=1))
    latest.default_config = {"postprocess.min_area_m2": 200}
    _class(session, "Без настроек", "ortho")
    migrate_inference_templates(session)
    assert effective_inference_template_row(session, dataset.key).default_config["postprocess.min_area_m2"] == 100
    assert effective_inference_template_row(session, other_dataset.key).default_config["postprocess.min_area_m2"] == 200
    assert len(effective_inference_template_row(session, other_dataset.key).default_config) == len(INFERENCE_BASE_DEFAULT_CONFIG)
    assert len(session.scalars(select(InferenceTemplateRow).where(InferenceTemplateRow.archived_at.is_(None))).all()) == 3


def test_migration_matches_orphan_template_by_dataset_name_and_latest_success(database):
    session, _ = database
    class_row = _class(session)
    dataset = _dataset(session, class_row)
    old = _legacy(session, "smp_segformer_b2", dataset=dataset, class_row=class_row,
                  values={"postprocess.simplify_m": 12})
    old.dataset_key = "исторический-ключ"
    session.add(TrainingResultRow(source="manual", class_key=dataset.key, dataset_key=dataset.key,
        class_display_name="Реки", architecture="smp_segformer_b2", model_name="Сеть", status="ok"))
    session.flush()
    migrate_inference_templates(session)
    assert old.dataset_key == "исторический-ключ"
    assert effective_inference_template_row(session, dataset.key).default_config["postprocess.simplify_m"] == 12


def test_f1_uses_class_settings_for_all_model_architectures(database):
    session, config = database
    class_row = _class(session)
    dataset = _dataset(session, class_row)
    template = _template(session, config)
    _assign(session, config, class_row, template)
    resolved, values, initial_hash = _test_samples._effective_inference_template(session, dataset.key, "none")
    assert resolved.id == template.id
    _service.update_inference_template_by_id(session, template.id,
        InferenceTemplateUpdate(default_config={"postprocess.min_area_m2": 77}), config)
    resolved, values, updated_hash = _test_samples._effective_inference_template(session, dataset.key, "none")
    assert values["postprocess.min_area_m2"] == 77
    assert updated_hash != initial_hash

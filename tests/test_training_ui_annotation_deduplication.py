"""Разовая очистка без изменения геометрии, черновиков и истории обучения."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import uuid

import pytest
from sqlalchemy import select

from mlsystem2.training_ui_api import _annotation_deduplication as cleanup
from mlsystem2.training_ui_api import _automation, _dataset_editor as editor
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import (
    AnnotationPublicationRow, AnnotationVersionRow, AutomationControlRow, AutomationRuleRow,
    DatasetAnnotationRow, DatasetEditorDraftRow, DatasetRow,
    ManagedDatasetSourceRow,
)
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError
from test_training_ui_annotation_versions import (
    editor_environment as _versions_environment, _copy, _scene, _save, _state,
)
from test_training_ui_dataset_editor import _git


@pytest.fixture
def editor_environment(tmp_path, monkeypatch):
    yield from _versions_environment.__wrapped__(tmp_path, monkeypatch)


def _config(env):
    return replace(get_config(), mlmarkup_root=env.editor_root)


def _setup(env):
    second = _copy(env)
    config = _config(env)
    factory = create_session_factory(config)
    with factory() as session:
        row = session.scalar(select(DatasetRow).where(DatasetRow.key == env.dataset_key))
        row.name = "main"
        session.commit()
    return config, factory, second


def _payload():
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "properties": {"name": "Первый"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                    [[1, 1], [1, 2], [2, 2], [2, 1], [1, 1]],
                ]}}]}


def test_signature_ignores_metadata_ring_order_and_collinear_points():
    first, second = _payload(), _payload()
    second["features"][0]["properties"] = {"name": "Другое имя", "id": 12}
    second["_mlsystem2_managed_dataset_key"] = "старый-источник"
    second["features"][0]["geometry"]["coordinates"][0] = [[4, 0], [2, 0], [0, 0], [0, 4], [4, 4], [4, 0]]
    second["features"][0]["geometry"]["coordinates"][1].reverse()
    assert cleanup.geometry_signature(first) == cleanup.geometry_signature(second)


@pytest.mark.parametrize("change", ["hole", "role", "class", "crs", "count", "small_change"])
def test_signature_preserves_training_semantics_and_exact_geometry(change):
    first, second = _payload(), _payload()
    feature = second["features"][0]
    if change == "hole":
        feature["geometry"]["coordinates"].pop()
    elif change == "role":
        feature["properties"]["_mlsystem2_role"] = "hard_negative"
    elif change == "class":
        feature["properties"]["_mlsystem2_class"] = "реки"
    elif change == "crs":
        second["crs"]["properties"]["name"] = "EPSG:4326"
    elif change == "count":
        second["features"].append(deepcopy(feature))
    else:
        feature["geometry"]["coordinates"][0][1][0] += 1e-12
    assert cleanup.geometry_signature(first) != cleanup.geometry_signature(second)


def test_inspection_is_read_only_and_apply_keeps_main_and_dates(editor_environment):
    env = editor_environment
    config, factory, second = _setup(env)
    first = _scene(env)
    keep = first["annotation_versions"]["selected_id"]
    old_date = datetime(2001, 2, 3, tzinfo=timezone.utc)
    with factory() as session:
        session.get(AnnotationVersionRow, uuid.UUID(keep)).updated_at = old_date
        session.commit()
    original = (env.editor_dataset / first["scene"]["annotation_name"]).read_bytes()
    head = _git(env.editor_root, "rev-parse", "HEAD").stdout
    with factory() as session:
        before = len(list(session.scalars(select(AnnotationVersionRow))))
        report = cleanup.inspect_duplicates(session, config)
        assert len(list(session.scalars(select(AnnotationVersionRow)))) == before
        assert len(report["groups"]) == 1
        assert report["groups"][0]["keep"] == keep
        assert _git(env.editor_root, "rev-parse", "HEAD").stdout == head
        result = cleanup.apply_duplicates(session, config, report)
        assert result["merged_versions"] == 1
        rows = list(session.scalars(select(DatasetAnnotationRow)))
        assert {str(row.version_id) for row in rows} == {keep}
        assert session.get(AnnotationVersionRow, uuid.UUID(keep)).updated_at.replace(tzinfo=timezone.utc) == old_date
        assert len(list(session.scalars(select(AnnotationVersionRow).where(AnnotationVersionRow.deleted_at.is_(None))))) == 1
        assert (env.editor_dataset / first["scene"]["annotation_name"]).read_bytes() == original
        pending, exclusions = cleanup.automation_exclusions(session)
        assert not pending
        assert second in {key for key, version in exclusions}
        assert env.dataset_key not in {key for key, version in exclusions}
        after = cleanup.inspect_duplicates(session, config)
        assert not after["groups"]
        assert cleanup.apply_duplicates(session, config, after)["merged_versions"] == 0
        assert cleanup.apply_duplicates(session, config, report)["already_applied"]
        assert _git(env.editor_root, "rev-parse", "HEAD").stdout.strip() == result["commit"]


def test_lazy_files_are_imported_only_on_apply(editor_environment):
    env = editor_environment
    config, factory, _ = _setup(env)
    with factory() as session:
        report = cleanup.inspect_duplicates(session, config)
        assert len(report["groups"]) == 1
        assert any(item["id"].startswith("file:") for item in report["groups"][0]["members"])
        assert cleanup.apply_duplicates(session, config, report)["merged_versions"] == 1
        assert not cleanup.inspect_duplicates(session, config)["groups"]


def test_draft_protects_whole_image_catalog(editor_environment):
    env = editor_environment
    config, factory, _ = _setup(env)
    detail = _scene(env)
    assert _save(env, detail, _state(detail)).status_code == 200
    with factory() as session:
        draft = session.scalar(select(DatasetEditorDraftRow))
        saved = (deepcopy(draft.geojson), deepcopy(draft.annotation_versions), draft.updated_at)
        report = cleanup.inspect_duplicates(session, config)
        assert report["groups"][0]["skip"]
        assert cleanup.apply_duplicates(session, config, report)["merged_versions"] == 0
        session.refresh(draft)
        assert saved == (draft.geojson, draft.annotation_versions, draft.updated_at)


def test_stale_plan_is_rejected_before_mutation(editor_environment):
    env = editor_environment
    config, factory, _ = _setup(env)
    detail = _scene(env)
    with factory() as session:
        report = cleanup.inspect_duplicates(session, config)
    assert _save(env, detail, _state(detail)).status_code == 200
    with factory() as session, pytest.raises(TrainingUIAPIError, match="изменились"):
        cleanup.apply_duplicates(session, config, report)


def test_unknown_push_outcome_recovers_without_losing_exclusions(editor_environment, monkeypatch):
    env = editor_environment
    config, factory, second = _setup(env)
    original = editor._push_with_retry

    def push_then_disconnect(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("Соединение оборвалось после принятия коммита")

    monkeypatch.setattr(editor, "_push_with_retry", push_then_disconnect)
    with factory() as session:
        report = cleanup.inspect_duplicates(session, config)
        with pytest.raises(RuntimeError, match="оборвалось"):
            cleanup.apply_duplicates(session, config, report)
        assert second in cleanup.automation_exclusions(session)[0]
    monkeypatch.setattr(editor, "_push_with_retry", original)
    with factory() as session:
        editor.recover_annotation_publications(session, config)
        assert not cleanup.inspect_duplicates(session, config)["groups"]
        pending, exclusions = cleanup.automation_exclusions(session)
        assert not pending and second in {key for key, _ in exclusions}
        editor.recover_annotation_publications(session, config)
        assert len(list(session.scalars(select(AnnotationPublicationRow)))) == 1


def test_automation_skips_only_technical_revision_and_preserves_rules(editor_environment, monkeypatch):
    env = editor_environment
    config, factory, second = _setup(env)
    with factory() as session:
        result = cleanup.apply_duplicates(session, config, cleanup.inspect_duplicates(session, config))
        current = next(item for item in _automation._automation_datasets(session, config) if item.key == second)
        session.add(AutomationRuleRow(dataset_key=second, architecture="smp_segformer_b0", training_enabled=True, pseudo_markup_enabled=True))
        control = session.scalar(select(AutomationControlRow))
        if control is None:
            session.add(AutomationControlRow(key="automation", enabled=True))
        else:
            control.enabled = True
        session.commit()
        calls = []
        monkeypatch.setattr(_automation, "_ensure_training_for_rule", lambda *args: calls.append("training"))
        monkeypatch.setattr(_automation, "_ensure_pseudo_markup_for_rule", lambda *args: calls.append("pseudo"))
        monkeypatch.setattr(_automation, "_cancel_stale_automation_jobs", lambda *args: calls.append("cancel"))
        monkeypatch.setattr(_automation, "_automation_datasets", lambda *args: [current])
        _automation.sync_automation_once(session, config)
        assert not calls
        monkeypatch.setattr(_automation, "_automation_datasets", lambda *args: [current.model_copy(update={"version": "следующая-ревизия"})])
        _automation.sync_automation_once(session, config)
        assert calls == ["cancel", "training", "pseudo"]
        assert result["commit"]
        assert session.scalar(select(AutomationControlRow)).enabled
        assert session.scalar(select(AutomationRuleRow).where(AutomationRuleRow.dataset_key == second)).training_enabled


def test_same_geometry_with_different_properties_preserves_main_payload(editor_environment):
    env = editor_environment
    config, factory, second = _setup(env)
    with factory() as session:
        row = session.scalar(select(DatasetRow).where(DatasetRow.key == second))
        path = config.mlmarkup_editor_root / row.source_path / "Olskij_SCN01.part.geojson"
        candidate = json.loads(path.read_text(encoding="utf-8"))
        candidate["features"][0]["properties"]["описание"] = "Дополнительные сведения копии"
        editor._write_geojson_atomic(path, candidate)
        _git(env.editor_root, "add", "--", path.relative_to(env.editor_root).as_posix())
        _git(env.editor_root, "commit", "-m", "Уточнить метаданные копии")
        _git(env.editor_root, "push", "origin", "HEAD:main")
        result = cleanup.apply_duplicates(session, config, cleanup.inspect_duplicates(session, config))
        assert result["merged_versions"] == 1
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
            (env.editor_dataset / path.name).read_text(encoding="utf-8"))


def _managed(session, source_key):
    source = session.scalar(select(DatasetRow).where(DatasetRow.key == source_key))
    target = DatasetRow(id=uuid.uuid4(), key="сводный-набор", class_id=source.class_id,
        name="Сводный", source_type="managed", source_path="managed/сводный-набор")
    session.add(target)
    session.flush()
    session.add(ManagedDatasetSourceRow(managed_dataset_id=target.id, source_dataset_id=source.id,
        object_type_id=1, object_type_slug="rivers", object_type_name="Реки", color="#336699", priority=0))
    second_source = session.scalar(select(DatasetRow).where(DatasetRow.name == "main"))
    session.add(ManagedDatasetSourceRow(managed_dataset_id=target.id, source_dataset_id=second_source.id,
        object_type_id=2, object_type_slug="other", object_type_name="Другая разметка", color="#669933", priority=1))
    session.commit()
    return target


def test_managed_draft_protects_source_image(editor_environment):
    env = editor_environment
    config, factory, second = _setup(env)
    with factory() as session:
        target = _managed(session, second)
        session.add(DatasetEditorDraftRow(dataset_key=target.key, annotation_name="Olskij_SCN01.part.geojson",
            username="Редактор", base_revision="revision", geojson=_payload()))
        session.commit()
        report = cleanup.inspect_duplicates(session, config)
        assert report["groups"][0]["skip"]
        assert cleanup.apply_duplicates(session, config, report)["merged_versions"] == 0


def test_dependent_managed_revision_is_also_excluded(editor_environment):
    env = editor_environment
    config, factory, second = _setup(env)
    with factory() as session:
        target = _managed(session, second)
        cleanup.apply_duplicates(session, config, cleanup.inspect_duplicates(session, config))
        pending, exclusions = cleanup.automation_exclusions(session)
        assert not pending
        assert {second, target.key} <= {key for key, _ in exclusions}


def test_failed_write_rolls_back_files_and_leaves_versions_active(editor_environment, monkeypatch):
    env = editor_environment
    config, factory, _ = _setup(env)
    original = editor._write_geojson_atomic

    def write_then_fail(*args):
        original(*args)
        raise RuntimeError("Сбой записи")

    monkeypatch.setattr(editor, "_write_geojson_atomic", write_then_fail)
    with factory() as session:
        before = _git(env.editor_root, "rev-parse", "HEAD").stdout
        with pytest.raises(RuntimeError, match="Сбой записи"):
            cleanup.apply_duplicates(session, config, cleanup.inspect_duplicates(session, config))
        assert _git(env.editor_root, "rev-parse", "HEAD").stdout == before
        assert not _git(env.editor_root, "status", "--porcelain").stdout
        assert not list(session.scalars(select(AnnotationVersionRow).where(AnnotationVersionRow.deleted_at.is_not(None))))
        assert not list(session.scalars(select(AnnotationPublicationRow)))


def test_recovery_does_not_suppress_later_content_revision(editor_environment, monkeypatch):
    env = editor_environment
    config, factory, second = _setup(env)
    original = editor._complete_annotation_publication

    def fail(*args, **kwargs):
        raise RuntimeError("Сбой после push")

    monkeypatch.setattr(editor, "_complete_annotation_publication", fail)
    with factory() as session:
        with pytest.raises(RuntimeError):
            cleanup.apply_duplicates(session, config, cleanup.inspect_duplicates(session, config))
        row = session.scalar(select(DatasetRow).where(DatasetRow.key == second))
        path = config.mlmarkup_editor_root / row.source_path / "Olskij_SCN01.part.geojson"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["features"] = data["features"][:1]
        editor._write_geojson_atomic(path, data)
        _git(env.editor_root, "add", "--", path.relative_to(env.editor_root).as_posix())
        _git(env.editor_root, "commit", "-m", "Новая содержательная правка")
        _git(env.editor_root, "push", "origin", "HEAD:main")
    monkeypatch.setattr(editor, "_complete_annotation_publication", original)
    with factory() as session:
        editor.recover_annotation_publications(session, config)
        assert not cleanup.automation_exclusions(session)[0]
        assert second not in {key for key, _ in cleanup.automation_exclusions(session)[1]}

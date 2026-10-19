"""История относится к опубликованной версии, а не к автосохранениям пользователя."""

from copy import deepcopy
import json
import uuid

from sqlalchemy import select
from shapely.geometry import box, mapping, shape

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import AnnotationFeatureEventRow, AnnotationPublicationRow
from test_training_ui_annotation_versions import editor_environment, _scene, _state, _save, _publish, _base, _copy, _git


def _history(env, detail, feature_id, key=None, version=None):
    return env.client.get(f"{_base(key or env.dataset_key)}/scenes/{detail['scene']['annotation_name']}/polygon-history",
        params={"feature_id": feature_id, "version_id": version or detail["annotation_versions"]["selected_id"]})


def test_drafts_and_cancellation_never_enter_polygon_history(editor_environment):
    env = editor_environment
    detail = _scene(env)
    key = detail["geojson"]["features"][0]["id"]
    old = _history(env, detail, key).json()
    assert [entry["action"] for entry in old["entries"]] == ["baseline"]
    assert old["entries"][0]["published_at"] is None
    assert old["entries"][0]["username"] is None
    changed = deepcopy(detail["geojson"])
    changed["features"][0]["geometry"] = mapping(box(1, 1, 2, 2))
    for _ in range(2):
        saved = _save(env, detail, _state(detail), changed)
        assert saved.status_code == 200, saved.text
        assert _history(env, detail, key).json() == old
    assert env.client.delete(f"{_base(env.dataset_key)}/drafts").status_code == 200
    assert _history(env, detail, key).json() == old


def test_published_edit_records_author_before_and_after_once(editor_environment):
    env = editor_environment
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    feature = candidate["features"][0]
    feature["geometry"] = mapping(box(1, 1, 2, 2))
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    _publish(env)
    current = _scene(env)
    response = _history(env, current, feature["id"])
    assert response.status_code == 200, response.text
    events = response.json()["entries"]
    assert [entry["action"] for entry in events] == ["baseline", "updated"]
    assert events[0]["username"] is None and events[0]["published_at"] is None
    assert shape(events[0]["feature"]["geometry"]).equals(shape(detail["geojson"]["features"][0]["geometry"]))
    assert shape(events[1]["feature"]["geometry"]).equals(box(1, 1, 2, 2))
    assert events[1]["username"] and events[1]["published_at"] and events[1]["commit"]
    assert _save(env, current, _state(current)).status_code == 200
    _publish(env)
    assert _history(env, _scene(env), feature["id"]).json()["entries"] == events


def test_new_polygon_creation_deletion_and_restore_are_separate_events(editor_environment):
    env = editor_environment
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    feature = deepcopy(candidate["features"][0])
    feature["id"] = "новый-полигон"
    feature["geometry"] = mapping(box(2, 2, 2.5, 2.5))
    candidate["features"].append(feature)
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    _publish(env)
    current = _scene(env)
    assert [entry["action"] for entry in _history(env, current, feature["id"]).json()["entries"]] == ["created"]
    candidate = deepcopy(current["geojson"])
    candidate["features"] = [item for item in candidate["features"] if item["id"] != feature["id"]]
    assert _save(env, current, _state(current), candidate).status_code == 200
    _publish(env)
    current = _scene(env)
    assert [entry["action"] for entry in _history(env, current, feature["id"]).json()["entries"]] == ["created", "deleted"]
    with create_session_factory(get_config())() as session:
        assert len(session.scalars(select(AnnotationFeatureEventRow).where(AnnotationFeatureEventRow.feature_id == feature["id"])).all()) == 2


def test_history_is_isolated_between_independent_versions(editor_environment):
    env = editor_environment
    second = _copy(env)
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    candidate["features"][0]["geometry"] = mapping(box(1, 1, 2, 2))
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    _publish(env)
    other = _scene(env, second)
    assert [event["action"] for event in _history(env, other, candidate["features"][0]["id"], second).json()["entries"]] == ["baseline"]
    response = _history(env, detail, 1, version=str(uuid.uuid4()))
    assert response.status_code == 400


def test_editor_draft_clips_objects_and_zones_and_preserves_identity(editor_environment):
    env = editor_environment
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    for role in ["positive", "hard_negative", "annotation_zone"]:
        feature = deepcopy(candidate["features"][0])
        feature["id"] = f"краевой-{role}"
        feature["properties"]["_mlsystem2_role"] = role
        feature["geometry"] = mapping(box(-5, -5, 5, 5))
        candidate["features"].append(feature)
    saved = _save(env, detail, _state(detail), candidate)
    assert saved.status_code == 200, saved.text
    footprint = shape(detail["valid_data_footprint"])
    clipped = [feature for feature in saved.json()["geojson"]["features"] if str(feature["id"]).startswith("краевой-")]
    assert len(clipped) == 3
    assert all(footprint.covers(shape(item["geometry"])) for item in clipped)
    _publish(env)
    assert all(footprint.covers(shape(item["geometry"])) for item in _scene(env)["geojson"]["features"])


def test_external_zone_does_not_silently_enable_whole_image(editor_environment):
    env = editor_environment
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    zone = deepcopy(candidate["features"][0])
    zone["id"] = "внешняя-зона"
    zone["properties"]["_mlsystem2_role"] = "annotation_zone"
    zone["geometry"] = mapping(box(100, 100, 120, 120))
    candidate["features"].append(zone)
    response = _save(env, detail, _state(detail), candidate)
    assert response.status_code == 400
    assert "целиком вне снимка" in response.text


def test_new_scene_records_the_first_polygon_publication(editor_environment):
    env = editor_environment
    detail = _scene(env)
    feature = deepcopy(detail["geojson"]["features"][0])
    response = env.client.post(f"{_base(env.dataset_key)}/drafts/import", json={"scenes": [
        {"annotation_name": "batch_SCN02.geojson", "geojson": {**detail["geojson"], "features": [feature]}},
    ]})
    assert response.status_code == 200, response.text
    _publish(env)
    current = env.client.get(f"{_base(env.dataset_key)}/scenes/batch_SCN02.geojson").json()
    events = _history(env, current, feature["id"]).json()["entries"]
    assert [entry["action"] for entry in events] == ["created"]
    assert events[0]["username"] == "editor" and events[0]["published_at"]


def test_history_is_atomic_and_recovery_is_idempotent(editor_environment, monkeypatch):
    from mlsystem2.training_ui_api import _dataset_editor

    env = editor_environment
    detail = _scene(env)
    candidate = deepcopy(detail["geojson"])
    candidate["features"][0]["geometry"] = mapping(box(1, 1, 2, 2))
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    record = _dataset_editor.record_polygon_history
    def fail(*args, **kwargs):
        record(*args, **kwargs)
        raise RuntimeError("Имитация отказа транзакции истории")
    monkeypatch.setattr(_dataset_editor, "record_polygon_history", fail)
    import pytest
    with pytest.raises(RuntimeError, match="отказа транзакции"):
        env.client.post(f"{_base(env.dataset_key)}/drafts/publish")
    with create_session_factory(get_config())() as session:
        assert not session.scalars(select(AnnotationFeatureEventRow)).all()
        assert session.scalar(select(AnnotationPublicationRow).where(AnnotationPublicationRow.completed.is_(False))) is not None
    monkeypatch.setattr(_dataset_editor, "record_polygon_history", record)
    current = _scene(env)
    first = _history(env, current, 1).json()
    assert [entry["action"] for entry in first["entries"]] == ["baseline", "updated"]
    with create_session_factory(get_config())() as session:
        publication = session.scalars(select(AnnotationPublicationRow).where(AnnotationPublicationRow.completed.is_(True))).all()[-1]
        _dataset_editor._complete_annotation_publication(session, publication, publication.commit_sha)
        session.commit()
    assert _history(env, current, 1).json() == first


def test_old_file_without_ids_has_stable_history_without_read_side_effects(editor_environment):
    env = editor_environment
    path = env.editor_dataset / "Olskij_SCN01.part.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for feature in payload["features"]:
        feature.pop("id", None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _git(env.editor_root, "add", "--", path.relative_to(env.editor_root).as_posix())
    _git(env.editor_root, "commit", "-m", "Импорт прежнего файла без ID")
    _git(env.editor_root, "push", "origin", "HEAD:main")
    before = path.read_bytes()
    detail = _scene(env)
    key = detail["geojson"]["features"][0]["id"]
    assert key == _scene(env)["geojson"]["features"][0]["id"]
    assert path.read_bytes() == before
    assert _history(env, detail, key).json()["entries"][0]["action"] == "baseline"
    candidate = deepcopy(detail["geojson"])
    candidate["features"][0]["geometry"] = mapping(box(1, 1, 2, 2))
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    _publish(env)
    assert [event["action"] for event in _history(env, _scene(env), key).json()["entries"]] == ["baseline", "updated"]


def test_reading_old_external_zone_cannot_silently_enable_whole_image(editor_environment):
    env = editor_environment
    path = env.editor_dataset / "Olskij_SCN01.part.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    zone = {"type": "Feature", "id": "прежняя-внешняя-зона", "properties": {"_mlsystem2_role": "annotation_zone"}, "geometry": mapping(box(100, 100, 120, 120))}
    payload["features"].append(zone)
    path.write_text(json.dumps(payload), encoding="utf-8")
    _git(env.editor_root, "add", "--", path.relative_to(env.editor_root).as_posix())
    _git(env.editor_root, "commit", "-m", "Прежняя зона за снимком")
    _git(env.editor_root, "push", "origin", "HEAD:main")
    detail = _scene(env)
    assert any(feature["id"] == zone["id"] for feature in detail["geojson"]["features"])
    changed = deepcopy(detail["geojson"])
    changed["features"][0]["geometry"] = mapping(box(1, 1, 2, 2))
    rejected = _save(env, detail, _state(detail), changed)
    assert rejected.status_code == 400 and "целиком вне снимка" in rejected.text
    changed["features"] = [feature for feature in changed["features"] if feature["id"] != zone["id"]]
    assert _save(env, detail, _state(detail), changed).status_code == 200

"""Проверки сущностей версий через настоящий HTTP, БД и локальный Git remote."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import uuid

import pytest
from sqlalchemy import select

from mlsystem2.training_ui_api import _dataset_editor
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import AnnotationPublicationRow, AnnotationVersionRow, DatasetAnnotationRow, DatasetRow
from test_training_ui_dataset_editor import editor_environment as _editor_environment, _git


@pytest.fixture
def editor_environment(tmp_path, monkeypatch):
    yield from _editor_environment.__wrapped__(tmp_path, monkeypatch)


def _base(key):
    from urllib.parse import quote
    return f"/api/v1/dataset-editor/datasets/{quote(key, safe='')}"


def _scene(env, key=None):
    base = _base(key or env.dataset_key)
    scenes = env.client.get(f"{base}/scenes")
    assert scenes.status_code == 200, scenes.text
    name = scenes.json()["scenes"][0]["annotation_name"]
    response = env.client.get(f"{base}/scenes/{name}")
    assert response.status_code == 200, response.text
    return response.json()


def _state(detail, selected=None):
    catalog = detail["annotation_versions"]
    return {"catalog_revision": catalog["revision"], "selected_id": selected or catalog["selected_id"], "edits": [], "replacements": {}}


def _save(env, detail, state, geojson=None, key=None):
    return env.client.put(f"{_base(key or env.dataset_key)}/drafts/{detail['scene']['annotation_name']}", json={
        "base_revision": detail["scene"]["revision"], "geojson": geojson or detail["geojson"], "annotation_versions": state,
    })


def _publish(env, key=None):
    response = env.client.post(f"{_base(key or env.dataset_key)}/drafts/publish")
    assert response.status_code == 200, response.text
    return response


def _copy(env):
    response = env.client.post(f"{_base(env.dataset_key)}/copy", json={"name": "Второй набор"})
    assert response.status_code == 200, response.text
    return response.json()["dataset_key"]


def _link_copy(env):
    key = _copy(env)
    second = _scene(env, key)
    first = _scene(env)
    state = _state(second, first["annotation_versions"]["selected_id"])
    response = _save(env, second, state, first["geojson"], key)
    assert response.status_code == 200, response.text
    _publish(env, key)
    return key


def test_versions_import_existing_datasets_independently_without_changing_files(editor_environment):
    env = editor_environment
    key = _copy(env)
    head = _git(env.editor_root, "rev-parse", "HEAD").stdout
    original = (env.editor_dataset / "Olskij_SCN01.part.geojson").read_bytes()
    first, second = _scene(env), _scene(env, key)
    assert first["scene"]["image_id"] == second["scene"]["image_id"]
    assert first["scene"]["annotation_version_count"] == 2
    assert first["annotation_versions"]["selected_id"] != second["annotation_versions"]["selected_id"]
    assert _git(env.editor_root, "rev-parse", "HEAD").stdout == head
    assert (env.editor_dataset / "Olskij_SCN01.part.geojson").read_bytes() == original
    assert _scene(env)["scene"]["annotation_version_count"] == 2


def test_image_versions_preserve_actual_annotation_filename_in_each_dataset(editor_environment):
    env = editor_environment
    second_key = _copy(env)
    folder = "Реки/Второй набор"
    _git(env.editor_root, "mv", f"{folder}/Olskij_SCN01.part.geojson", f"{folder}/временное-имя.geojson")
    _git(env.editor_root, "mv", f"{folder}/временное-имя.geojson", f"{folder}/olskij_scn01.part.geojson")
    _git(env.editor_root, "commit", "-m", "Уточнить регистр имени файла")
    _git(env.editor_root, "push", "origin", "HEAD:main")
    first, second = _scene(env), _scene(env, second_key)
    assert first["scene"]["image_id"] == second["scene"]["image_id"]
    state = _state(second, first["annotation_versions"]["selected_id"])
    assert _save(env, second, state, first["geojson"], second_key).status_code == 200
    _publish(env, second_key)
    manifest = json.loads((env.editor_root / folder / ".mlsystem2-annotation-selections.json").read_text(encoding="utf-8"))
    assert list(manifest["annotations"]) == ["olskij_scn01.part.geojson"]
    assert _scene(env, second_key)["annotation_versions"]["selected_id"] == first["annotation_versions"]["selected_id"]


def test_switch_is_a_draft_and_shared_edit_updates_both_datasets(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    first, second = _scene(env), _scene(env, second_key)
    assert first["annotation_versions"]["selected_id"] == second["annotation_versions"]["selected_id"]
    selected = next(version for version in first["annotation_versions"]["versions"] if version["id"] == first["annotation_versions"]["selected_id"])
    assert {usage["dataset_key"] for usage in selected["usages"]} == {env.dataset_key, second_key}
    candidate = deepcopy(first["geojson"])
    candidate["features"] = candidate["features"][:1]
    saved = _save(env, first, _state(first), candidate)
    assert saved.status_code == 200, saved.text
    assert len(_scene(env, second_key)["geojson"]["features"]) == 2
    _publish(env)
    assert len(_scene(env, second_key)["geojson"]["features"]) == 1
    copies = [env.editor_dataset, env.editor_root / "Реки" / "Второй набор"]
    assert json.loads((copies[0] / first["scene"]["annotation_name"]).read_text(encoding="utf-8")) == json.loads((copies[1] / first["scene"]["annotation_name"]).read_text(encoding="utf-8"))
    assert all((folder / ".mlsystem2-annotation-selections.json").is_file() for folder in copies)


def test_selection_preserves_canonical_geometry_outside_display_footprint(editor_environment):
    env = editor_environment
    source_path = env.editor_dataset / "Olskij_SCN01.part.geojson"
    original = source_path.read_bytes()
    second_key = _link_copy(env)
    detail = _scene(env)
    selected_id = detail["annotation_versions"]["selected_id"]
    assert len(detail["geojson"]["features"]) == 2
    assert len(json.loads(original)["features"]) == 3
    assert source_path.read_bytes() == original
    selected = next(version for version in detail["annotation_versions"]["versions"] if version["id"] == selected_id)
    assert selected["revision"] == 1
    assert json.loads((env.editor_root / "Реки" / "Второй набор" / source_path.name).read_text(encoding="utf-8")) == json.loads(original)
    assert _scene(env, second_key)["annotation_versions"]["selected_id"] == selected_id


def test_new_independent_version_and_discard_do_not_change_shared_original(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    detail = _scene(env)
    original = detail["annotation_versions"]["selected_id"]
    new_id = str(uuid.uuid4())
    state = _state(detail, new_id)
    empty = {**detail["geojson"], "features": []}
    state["edits"] = [{"id": new_id, "name": "Пустая версия", "geojson": empty}]
    saved = _save(env, detail, state, empty)
    assert saved.status_code == 200, saved.text
    restored = _scene(env)
    assert restored["draft"]["annotation_versions"]["selected_id"] == new_id
    assert restored["annotation_versions"]["selected_id"] == original
    assert env.client.delete(f"{_base(env.dataset_key)}/drafts").status_code == 200
    assert _scene(env)["draft"] is None
    assert _save(env, detail, state, empty).status_code == 200
    _publish(env)
    assert _scene(env)["geojson"]["features"] == []
    assert len(_scene(env, second_key)["geojson"]["features"]) == 2
    assert _scene(env, second_key)["annotation_versions"]["selected_id"] == original


def test_delete_shared_version_replaces_every_usage_and_preserves_other_versions(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    detail = _scene(env)
    old_id = detail["annotation_versions"]["selected_id"]
    replacement = next(version["id"] for version in detail["annotation_versions"]["versions"] if version["id"] != old_id)
    target = env.client.get(f"{_base(env.dataset_key)}/scenes/{detail['scene']['annotation_name']}/versions/{replacement}").json()
    state = _state(detail, replacement)
    state["replacements"] = {old_id: replacement}
    saved = _save(env, detail, state, target["geojson"])
    assert saved.status_code == 200, saved.text
    assert len(_scene(env)["annotation_versions"]["versions"]) == 2
    _publish(env)
    for key in [env.dataset_key, second_key]:
        current = _scene(env, key)
        assert current["annotation_versions"]["selected_id"] == replacement
        assert [version["id"] for version in current["annotation_versions"]["versions"]] == [replacement]
    assert env.client.get(f"{_base(env.dataset_key)}/scenes/{detail['scene']['annotation_name']}/versions/{old_id}").status_code == 400


def test_shared_concurrent_draft_becomes_stale_and_cannot_overwrite_new_version(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    first, second = _scene(env), _scene(env, second_key)
    assert _save(env, second, _state(second), key=second_key).status_code == 200
    candidate = {**first["geojson"], "features": []}
    assert _save(env, first, _state(first), candidate).status_code == 200
    _publish(env)
    current = _scene(env, second_key)
    assert current["draft"]["stale"] is True
    conflict = env.client.post(f"{_base(second_key)}/drafts/publish")
    assert conflict.status_code == 409, conflict.text
    assert _scene(env)["geojson"]["features"] == []


def test_invalid_replacement_and_cross_scene_reference_are_rejected(editor_environment):
    env = editor_environment
    detail = _scene(env)
    original = detail["annotation_versions"]["selected_id"]
    for state in [
        {**_state(detail), "replacements": {original: original}},
        _state(detail, str(uuid.uuid4())),
        {**_state(detail), "replacements": {original: str(uuid.uuid4())}},
    ]:
        response = _save(env, detail, state)
        assert response.status_code == 400, response.text
    assert _scene(env)["draft"] is None


def test_published_git_recovers_database_after_process_failure(editor_environment, monkeypatch):
    env = editor_environment
    second_key = _link_copy(env)
    detail = _scene(env)
    candidate = {**detail["geojson"], "features": []}
    assert _save(env, detail, _state(detail), candidate).status_code == 200
    complete = _dataset_editor._complete_annotation_publication

    def failure(*args, **kwargs):
        raise RuntimeError("Имитация завершения процесса после push")

    monkeypatch.setattr(_dataset_editor, "_complete_annotation_publication", failure)
    with pytest.raises(RuntimeError, match="Имитация"):
        env.client.post(f"{_base(env.dataset_key)}/drafts/publish")
    with create_session_factory(get_config())() as session:
        pending = session.scalars(select(AnnotationPublicationRow).where(AnnotationPublicationRow.completed.is_(False))).all()
        assert len(pending) == 1
    monkeypatch.setattr(_dataset_editor, "_complete_annotation_publication", complete)
    assert _scene(env)["geojson"]["features"] == []
    assert _scene(env, second_key)["geojson"]["features"] == []
    assert _scene(env)["draft"] is None
    with create_session_factory(get_config())() as session:
        assert session.scalar(select(AnnotationPublicationRow).where(AnnotationPublicationRow.completed.is_(False))) is None


def test_failed_push_keeps_database_and_draft_unchanged(editor_environment, monkeypatch):
    env = editor_environment
    detail = _scene(env)
    state = _state(detail)
    candidate = {**detail["geojson"], "features": []}
    assert _save(env, detail, state, candidate).status_code == 200
    head = _git(env.editor_root, "rev-parse", "HEAD").stdout

    def failure(config, **kwargs):
        _dataset_editor._discard_local_commit(config)
        raise _dataset_editor.DatasetEditorGitError("Проверка отказа push")

    monkeypatch.setattr(_dataset_editor, "_push_with_retry", failure)
    response = env.client.post(f"{_base(env.dataset_key)}/drafts/publish")
    assert response.status_code == 503
    assert _git(env.editor_root, "rev-parse", "HEAD").stdout == head
    restored = _scene(env)
    assert len(restored["geojson"]["features"]) == 2
    assert restored["draft"]["geojson"]["features"] == []


def test_lost_push_response_recovers_accepted_remote_publication(editor_environment, monkeypatch):
    env = editor_environment
    second_key = _link_copy(env)
    detail = _scene(env)
    assert _save(env, detail, _state(detail), {**detail["geojson"], "features": []}).status_code == 200
    push = _dataset_editor._push_with_retry

    def lost_response(config, **kwargs):
        push(config, **kwargs)
        raise _dataset_editor.DatasetEditorGitError("Имитация потери ответа после принятия push")

    monkeypatch.setattr(_dataset_editor, "_push_with_retry", lost_response)
    assert env.client.post(f"{_base(env.dataset_key)}/drafts/publish").status_code == 503
    monkeypatch.setattr(_dataset_editor, "_push_with_retry", push)
    assert _scene(env)["geojson"]["features"] == []
    assert _scene(env, second_key)["geojson"]["features"] == []
    assert _scene(env)["draft"] is None


def test_version_entities_live_in_database_not_git_and_survive_unlink(editor_environment):
    env = editor_environment
    detail = _scene(env)
    version_id = uuid.UUID(detail["annotation_versions"]["selected_id"])
    response = env.client.request("DELETE", f"{_base(env.dataset_key)}/scenes/{detail['scene']['annotation_name']}", json={"revision": detail["scene"]["revision"]})
    assert response.status_code == 200, response.text
    _publish(env)
    with create_session_factory(get_config())() as session:
        assert session.get(AnnotationVersionRow, version_id).deleted_at is None
        assert session.scalar(select(DatasetAnnotationRow).where(DatasetAnnotationRow.version_id == version_id)) is None
    tracked = _git(env.editor_root, "ls-files").stdout
    assert ".mlsystem2-annotation-versions" not in tracked


def test_external_file_edit_imports_independent_version_without_overwriting_shared(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    original_id = _scene(env)["annotation_versions"]["selected_id"]
    path = env.editor_root / "Реки" / "Второй набор" / "Olskij_SCN01.part.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["features"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    _git(env.editor_root, "add", "--", path.relative_to(env.editor_root).as_posix())
    _git(env.editor_root, "commit", "-m", "Внешнее уточнение разметки")
    _git(env.editor_root, "push", "origin", "HEAD:main")
    assert _scene(env, second_key)["annotation_versions"]["selected_id"] != original_id
    assert _scene(env, second_key)["geojson"]["features"] == []
    first = _scene(env)
    assert first["annotation_versions"]["selected_id"] == original_id
    assert len(first["geojson"]["features"]) == 2


def test_returning_archived_dataset_does_not_select_deleted_version(editor_environment):
    env = editor_environment
    second_key = _link_copy(env)
    factory = create_session_factory(get_config())
    with factory() as session:
        row = session.scalar(select(DatasetRow).where(DatasetRow.key == second_key))
        row.deleted_at = datetime.now(timezone.utc)
        session.commit()
    detail = _scene(env)
    old_id = detail["annotation_versions"]["selected_id"]
    replacement = next(version["id"] for version in detail["annotation_versions"]["versions"] if version["id"] != old_id)
    state = _state(detail, replacement)
    state["replacements"] = {old_id: replacement}
    assert _save(env, detail, state).status_code == 200
    _publish(env)
    with factory() as session:
        row = session.scalar(select(DatasetRow).where(DatasetRow.key == second_key))
        row.deleted_at = None
        session.commit()
    restored = _scene(env, second_key)
    assert restored["annotation_versions"]["selected_id"] != old_id
    assert _save(env, restored, _state(restored), key=second_key).status_code == 200


def test_batch_publication_rebuilds_selection_manifest_and_rejects_other_image_version(editor_environment):
    env = editor_environment
    base = _base(env.dataset_key)
    assert env.client.post(f"{base}/scenes", json={"folder_path": "batch"}).status_code == 200
    first = env.client.get(f"{base}/scenes/Olskij_SCN01.part.geojson").json()
    second = env.client.get(f"{base}/scenes/batch_SCN02.geojson").json()
    foreign = second["annotation_versions"]["selected_id"]
    assert _save(env, first, _state(first, foreign)).status_code == 400
    forged = _state(first, foreign)
    forged["edits"] = [{"id": foreign, "name": "Чужая версия", "geojson": first["geojson"]}]
    assert _save(env, first, forged).status_code == 400
    assert env.client.request("DELETE", f"{base}/scenes/{first['scene']['annotation_name']}",
        json={"revision": first["scene"]["revision"]}).status_code == 200
    new_id = str(uuid.uuid4())
    state = _state(second, new_id)
    state["edits"] = [{"id": new_id, "name": "Независимая проверка", "geojson": second["geojson"]}]
    assert _save(env, second, state).status_code == 200
    _publish(env)
    manifest = json.loads((env.editor_dataset / ".mlsystem2-annotation-selections.json").read_text(encoding="utf-8"))
    assert set(manifest["annotations"]) == {"batch_SCN02.geojson", "batch_SCN03.geojson"}
    assert manifest["annotations"]["batch_SCN02.geojson"]["version_id"] == new_id

"""Личная заметка: доступ, пустой текст, конкуренция вкладок и жизненный цикл файлов."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import create_session_factory
from mlsystem2.training_ui_api._models import PersonalNoteAttachmentRow, PersonalNoteRow
from mlsystem2.training_ui_api.api import create_app


PREFIX = "/api/v1/personal-note"


@pytest.fixture
def note_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "stored"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "секрет-проверки")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SECURE_COOKIES", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USERS_JSON", json.dumps([
        {"username": "Aspinoza", "password": "test-password", "role": "admin", "aliases": ["mlsystem"]},
        {"username": "Alice", "password": "test-password", "role": "admin"},
    ]))
    client = TestClient(create_app())
    yield client
    client.close()


def login(client: TestClient, username: str = "Aspinoza") -> None:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": "test-password"})
    assert response.status_code == 200


def test_only_owner_can_access_every_operation(note_client):
    file_id = uuid4()
    operations = [
        ("GET", PREFIX, {}),
        ("PUT", PREFIX, {"json": {"content": "", "expected_revision": 0}}),
        ("POST", PREFIX + "/attachments", {"files": {"file": ("тест.txt", b"123")}}),
        ("GET", f"{PREFIX}/attachments/{file_id}/download", {}),
        ("DELETE", f"{PREFIX}/attachments/{file_id}", {}),
    ]
    for method, path, kwargs in operations:
        assert note_client.request(method, path, **kwargs).status_code == 401
    login(note_client, "Alice")
    for method, path, kwargs in operations:
        assert note_client.request(method, path, **kwargs).status_code == 403
    with create_session_factory(get_config())() as db:
        assert db.scalar(select(func.count()).select_from(PersonalNoteRow)) == 0
        assert db.scalar(select(func.count()).select_from(PersonalNoteAttachmentRow)) == 0


def test_exact_text_empty_note_conflict_and_no_history(note_client):
    login(note_client, "mlsystem")
    response = note_client.get(PREFIX)
    assert response.json() == {"content": "", "revision": 0, "attachments": []}
    assert response.headers["cache-control"] == "no-store"
    text = "  Заметка\n\tВторая строка\n<script>это текст</script>\n"
    first = note_client.put(PREFIX, json={"content": text, "expected_revision": 0})
    assert first.status_code == 200
    assert first.json()["content"] == text
    assert note_client.put(PREFIX, json={"content": "конфликт", "expected_revision": 0}).status_code == 409
    empty = note_client.put(PREFIX, json={"content": "", "expected_revision": 1})
    assert empty.status_code == 200
    assert empty.json() == {"content": "", "revision": 2, "attachments": []}
    assert note_client.put(PREFIX, json={"content": "конфликт", "expected_revision": 1}).status_code == 409
    assert note_client.get(PREFIX).json()["content"] == ""
    with create_session_factory(get_config())() as db:
        assert db.scalar(select(func.count()).select_from(PersonalNoteRow)) == 1
        assert db.get(PersonalNoteRow, 1).content == ""


def test_attachments_without_text_are_private_and_permanently_deleted(note_client):
    login(note_client)
    content = bytes(range(256)) * 10000
    uploaded = []
    for _ in range(2):
        response = note_client.post(PREFIX + "/attachments", files={
            "file": ("../путь\\данные.bin", content, "application/octet-stream"),
        })
        assert response.status_code == 201
        uploaded.append(response.json())
    first, second = uploaded
    assert first["original_name"] == "данные.bin"
    assert first["size_bytes"] == len(content)
    assert first["id"] != second["id"]
    assert note_client.get(PREFIX).json()["content"] == ""
    assert len(note_client.get(PREFIX).json()["attachments"]) == 2
    downloaded = note_client.get(first["download_url"])
    assert downloaded.content == content
    assert downloaded.headers["cache-control"] == "no-store"
    assert downloaded.headers["x-content-type-options"] == "nosniff"
    assert downloaded.headers["content-disposition"].startswith("attachment;")
    assert note_client.get(f"/api/v1/files/{first['id']}/download").status_code in {400, 404}
    login(note_client, "Alice")
    assert note_client.get(first["download_url"]).status_code == 403
    assert note_client.delete(f"{PREFIX}/attachments/{first['id']}").status_code == 403
    login(note_client)
    assert note_client.delete(f"{PREFIX}/attachments/{first['id']}").status_code == 204
    assert note_client.get(first["download_url"]).status_code == 404
    assert note_client.delete(f"{PREFIX}/attachments/{first['id']}").status_code == 404
    root = get_config().stored_files_root / "personal-note"
    assert sorted(path.name for path in root.iterdir()) == [second["id"]]
    with create_session_factory(get_config())() as db:
        assert db.scalar(select(func.count()).select_from(PersonalNoteAttachmentRow)) == 1


def test_upload_rollback_removes_unpublished_file(note_client, monkeypatch):
    from sqlalchemy.orm import Session

    login(note_client)

    def fail_commit(self):
        raise RuntimeError("Ошибка сохранения БД")

    monkeypatch.setattr(Session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="Ошибка сохранения БД"):
        note_client.post(PREFIX + "/attachments", files={"file": ("тест.txt", b"123")})
    root = get_config().stored_files_root / "personal-note"
    assert list(root.iterdir()) == []


def test_invalid_filename_does_not_write_file(note_client):
    login(note_client)
    response = note_client.post(PREFIX + "/attachments", files={"file": ("..", b"123")})
    assert response.status_code == 422
    assert not (get_config().stored_files_root / "personal-note").exists()

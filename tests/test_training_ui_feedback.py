"""Проверки авторства, подтверждения и восстановления обработки обращений."""

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from mlsystem2.training_ui_api.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "секрет-проверки")
    monkeypatch.setenv("MLSYSTEM2_FEEDBACK_API_TOKEN", "test-feedback-token")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USERS_JSON", json.dumps([
        {"username": "owner", "password": "test-password", "role": "admin"},
        {"username": "author", "password": "test-password", "role": "user", "aliases": ["alias"]},
    ]))
    return TestClient(create_app())


def login(client, user="alias"):
    assert client.post("/api/v1/auth/login", json={"username": user, "password": "test-password"}).status_code == 200


def submit(client):
    payload = {"submission_id": str(uuid4()), "kind": "improvement", "title": "Поиск снимков", "message": "Добавить поиск снимков по имени", "page_path": "#/dataset-editor/лес", "page_title": "Редактор датасетов"}
    response = client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200, response.text
    return response.json(), payload


def change(client, item, **values):
    return client.patch(f"/api/v1/feedback/{item['id']}", json={"expected_revision": item["revision"], **values})


def test_authorship_idempotence_and_shared_list(client):
    assert client.get("/api/v1/feedback").status_code == 401
    login(client)
    item, payload = submit(client)
    assert item["author"] == "author"
    assert item["status"] == "waiting"
    assert client.post("/api/v1/feedback", json=payload).json()["id"] == item["id"]
    assert client.post("/api/v1/feedback", json={**payload, "author": "владелец"}).status_code == 422
    assert change(client, item, status="preparing").status_code == 403
    login(client, "owner")
    assert client.get("/api/v1/feedback").json()["items"][0]["message"] == item["message"]


def test_implementation_requires_approved_unchanged_plan(client):
    login(client)
    item, _ = submit(client)
    login(client, "owner")
    assert change(client, item, status="implementing").status_code == 409
    preparing = change(client, item, status="preparing").json()
    assert change(client, item, progress="Старая версия").status_code == 409
    assert change(client, preparing, status="implementing", approval_note="Сделай и опубликуй").status_code == 409
    prepared = change(client, preparing, preparation="Поиск фильтрует список по имени без учёта регистра").json()
    assert change(client, prepared, status="implementing").status_code == 409
    assert change(client, prepared, status="implementing", preparation="Другое решение", approval_note="Принимаю решение").status_code == 409
    implementing = change(client, prepared, status="implementing", approval_note="Предложение принимаю, сделай и опубликуй").json()
    assert implementing["approved_by"] == "owner"
    assert implementing["approved_at"]
    assert change(client, implementing, preparation="Подменённое решение").status_code == 409
    assert change(client, implementing, status="implemented").status_code == 409
    completed = change(client, implementing, status="implemented", news_slug="search-images", commit_sha="a" * 40).json()
    assert completed["status"] == "implemented"
    assert change(client, completed, progress="Повторный запуск").status_code == 409
    assert client.get("/api/v1/feedback?active=true").json()["items"] == []


def test_service_token_and_pagination(client):
    login(client)
    first, _ = submit(client)
    second, _ = submit(client)
    client.cookies.clear()
    client.headers["Authorization"] = "Bearer invalid"
    assert client.get("/api/v1/feedback").status_code == 401
    client.headers["Authorization"] = "Bearer test-feedback-token"
    page = client.get("/api/v1/feedback?limit=1").json()
    assert page["items"][0]["id"] == second["id"]
    assert page["has_more"]
    assert client.get(f"/api/v1/feedback?before_id={second['id']}").json()["items"][0]["id"] == first["id"]
    assert change(client, first, status="preparing").status_code == 200
    assert client.get("/api/v1/feedback/999999").status_code == 404


def test_context_and_credit_validation(client):
    login(client)
    _, payload = submit(client)
    for path in ("https://example.org", "#//evil?token=secret", "#/bad\\path"):
        assert client.post("/api/v1/feedback", json={**payload, "submission_id": str(uuid4()), "page_path": path}).status_code == 422
    item = client.post("/api/v1/feedback", json={**payload, "submission_id": str(uuid4()), "credit_name": "Подпись автора"}).json()
    assert item["credit_name"] == "Подпись автора"

"""Проверка идентификации сессии и границ аналитики."""

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._usage import action_for_request, metrica_user_id
from mlsystem2.training_ui_api.api import create_app


@pytest.fixture
def usage_client(tmp_path, monkeypatch):
    for key, value in {
        "DATABASE_URL": f"sqlite:///{tmp_path / 'ui.db'}", "DATABASE_SCHEMA": "",
        "SCRATCH_ROOT": str(tmp_path / "scratch"), "SESSION_SECRET": "test-secret",
        "WORKER_ENABLED": "false", "COOKIE_SECURE": "false",
        "USERS_JSON": json.dumps([
            {"username": "owner", "password": "one", "role": "admin", "aliases": ["owner-alias"]},
            {"username": "member", "password": "two", "role": "user"},
        ]),
    }.items():
        monkeypatch.setenv(f"MLSYSTEM2_TRAINING_UI_{key}", value)
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "markup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_METRICA_COUNTER_ID", "113578839")
    with TestClient(create_app()) as client:
        yield client


def test_usage_identity_uses_canonical_session_not_browser_payload(usage_client):
    client = usage_client
    assert client.get("/api/v1/usage/config").status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "owner-alias", "password": "one"}).status_code == 200
    response = client.get("/api/v1/usage/config")
    assert response.headers["Cache-Control"] == "no-store"
    owner = response.json()
    assert owner == {"metrica_counter_id": 113578839, "metrica_user_id": metrica_user_id("owner", get_config())}
    client.post("/api/v1/auth/logout")
    client.post("/api/v1/auth/login", json={"username": "member", "password": "two"})
    member = client.get("/api/v1/usage/config").json()
    assert member["metrica_user_id"] != owner["metrica_user_id"]
    assert "username" not in member


def test_explicit_action_header_has_identity_and_polling_has_none(usage_client):
    client = usage_client
    client.post("/api/v1/auth/login", json={"username": "owner", "password": "one"})
    response = client.post("/api/v1/training-jobs", json={})
    assert response.status_code == 422
    assert response.headers["X-Grovika-Action"] == "training_start"
    assert response.headers["X-Grovika-User"] == metrica_user_id("owner", get_config())
    assert "X-Grovika-Action" not in client.get("/api/v1/queues/count").headers
    client.post("/api/v1/auth/logout")
    assert "X-Grovika-Action" not in client.post(f"/api/v1/jobs/{uuid4()}/move-up").headers


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "queues/count"), ("GET", "results/pseudo-markup/{result_id}/raster/{scene_id}"),
    ("POST", "results/pseudo-markup/compare/counts"),
    ("POST", "results/pseudo-markup/compare/{scene_id}/layers"),
    ("POST", "results/training/{result_id}/test-f1/view"),
    ("PUT", "dataset-editor/datasets/{dataset_key}/drafts/{annotation_name}"),
    ("PUT", "test-sample-batches/options/{dataset_key}/settings"),
    ("POST", "test-samples/reconcile"), ("PATCH", "feedback/{feedback_id}"),
])
def test_background_operations_are_not_feature_usage(method, path):
    assert action_for_request(method, f"/api/v1/{path}") is None


@pytest.mark.parametrize(("method", "path", "code"), [
    ("POST", "results/training/{result_id}/continue", "training_continue"),
    ("POST", "results/datasets/{dataset_key}/pseudo-markup", "pseudo_create"),
    ("POST", "results/datasets/{dataset_key}/test-f1", "test_f1_calculate"),
    ("POST", "dataset-editor/datasets/{dataset_key}/drafts/publish", "dataset_publish"),
    ("PUT", "inference-templates/by-id/{template_id}", "inference_template_save"),
    ("POST", "results/training/{result_id}/triton-zip", "model_export"),
    ("POST", "test-sample-batches", "test_create"),
    ("GET", "dataset-editor/datasets/{dataset_key}/download", "file_download"),
])
def test_user_operations_are_classified(method, path, code):
    assert action_for_request(method, f"/api/v1/{path}") == code

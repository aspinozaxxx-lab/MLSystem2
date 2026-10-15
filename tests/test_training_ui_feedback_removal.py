"""Удалённые обращения недоступны в API и больше не хранятся в БД."""

import io
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi.testclient import TestClient

from mlsystem2.training_ui_api._database import Base
from mlsystem2.training_ui_api.api import create_app


@pytest.mark.parametrize("authorization", ["anonymous", "session", "old_token"])
def test_removed_feedback_routes_are_unavailable(tmp_path, monkeypatch, authorization):
    for key, value in {
        "DATABASE_URL": f"sqlite:///{tmp_path / 'ui.db'}", "DATABASE_SCHEMA": "",
        "WORKER_ENABLED": "false", "COOKIE_SECURE": "false", "SESSION_SECRET": "учебный-секрет",
        "SCRATCH_ROOT": str(tmp_path / "scratch"), "STORED_FILES_ROOT": str(tmp_path / "files"),
        "FRONTEND_DIST": str(tmp_path / "frontend"),
        "USERS_JSON": json.dumps([{"username": "owner", "password": "test", "role": "admin"}]),
    }.items():
        monkeypatch.setenv(f"MLSYSTEM2_TRAINING_UI_{key}", value)
    monkeypatch.setenv("MLSYSTEM2_FEEDBACK_API_TOKEN", "old-feedback-token")
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "markup"))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    app = create_app()
    with TestClient(app) as client:
        if authorization == "session":
            assert client.post("/api/v1/auth/login", json={"username": "owner", "password": "test"}).status_code == 200
        elif authorization == "old_token":
            client.headers["Authorization"] = "Bearer old-feedback-token"
        for method, path in [("GET", "/feedback"), ("POST", "/feedback"),
                             ("GET", "/feedback/1"), ("PATCH", "/feedback/1")]:
            response = client.request(method, f"/api/v1{path}")
            assert response.status_code == 404
            assert "X-Grovika-Action" not in response.headers
    schema = app.openapi()
    assert not any("feedback" in path for path in schema["paths"])
    assert not any(name.startswith("Feedback") for name in schema["components"]["schemas"])
    assert not any(table.name == "feedback" for table in Base.metadata.tables.values())


def test_feedback_removal_migration_preserves_other_data_and_cannot_restore_messages(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    versions = Path(__file__).parents[1] / "migrations" / "versions"
    original = runpy.run_path(str(versions / "20261001_0028_feedback.py"))
    removal = runpy.run_path(str(versions / "20261015_0033_remove_feedback.py"))
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            original["upgrade"]()
            metadata = sa.MetaData()
            feedback = sa.Table("feedback", metadata, autoload_with=connection)
            datasets = sa.Table("datasets", metadata, sa.Column("id", sa.Integer, primary_key=True),
                                sa.Column("name", sa.String))
            datasets.create(connection)
            connection.execute(datasets.insert().values(id=1, name="Основной датасет"))
            connection.execute(feedback.insert().values(
                submission_id=uuid4().hex, kind="improvement", title="Предложение",
                message="Текст обращения", author="учебный-пользователь", page_path="#/",
                page_title="Главная", status="implemented", revision=4,
                preparation="Подготовленное решение", progress="Реализовано",
                approved_at=datetime.now(timezone.utc), approved_by="владелец",
            ))
            expected_data = connection.execute(sa.select(datasets)).all()
            removal["upgrade"]()
            assert "feedback" not in sa.inspect(connection).get_table_names()
            assert connection.execute(sa.select(datasets)).all() == expected_data
            removal["downgrade"]()
            assert sa.inspect(connection).has_table("feedback")
            assert connection.execute(sa.select(sa.func.count()).select_from(feedback)).scalar_one() == 0
            removal["upgrade"]()
            assert "feedback" not in sa.inspect(connection).get_table_names()
    engine.dispose()


@pytest.mark.parametrize("schema", ["", "training_ui"])
def test_feedback_removal_sql_targets_only_its_table(monkeypatch, schema):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", schema)
    migration = runpy.run_path(str(Path(__file__).parents[1] / "migrations" / "versions"
                                  / "20261015_0033_remove_feedback.py"))
    output = io.StringIO()
    context = MigrationContext.configure(dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output})
    with Operations.context(context):
        migration["upgrade"]()
    assert output.getvalue().strip() == f"DROP TABLE {schema + '.' if schema else ''}feedback;"

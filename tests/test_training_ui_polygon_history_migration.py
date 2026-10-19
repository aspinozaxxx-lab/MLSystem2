"""Индексированная история добавляется без изменения существующих версий и черновиков."""

import importlib.util
from io import StringIO
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa


def _migration():
    path = Path(__file__).parents[1] / "migrations/versions/20261019_0036_annotation_feature_history.py"
    spec = importlib.util.spec_from_file_location("polygon_history_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_migration_protects_existing_audit(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        for name in ("annotation_versions", "annotation_publications"):
            connection.execute(sa.text(f"CREATE TABLE {name} (id VARCHAR(32) PRIMARY KEY)"))
        with Operations.context(MigrationContext.configure(connection)):
            migration = _migration()
            migration.upgrade()
            connection.execute(sa.text("INSERT INTO annotation_feature_events (id, publication_id, version_id, feature_id, action, recorded_at) VALUES ('событие', 'публикация', 'версия', 'полигон', 'created', CURRENT_TIMESTAMP)"))
            with pytest.raises(RuntimeError, match="Откат удалит историю"):
                migration.downgrade()
            connection.execute(sa.text("DELETE FROM annotation_feature_events"))
            migration.downgrade()
            assert "annotation_feature_events" not in sa.inspect(connection).get_table_names()


def test_history_migration_renders_postgres_with_schema(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    output = StringIO()
    context = MigrationContext.configure(url="postgresql://", opts={"as_sql": True, "output_buffer": output})
    with Operations.context(context):
        _migration().upgrade()
    sql = output.getvalue()
    assert "CREATE TABLE training_ui.annotation_feature_events" in sql
    assert "REFERENCES training_ui.annotation_versions" in sql
    assert "ix_annotation_feature_history" in sql

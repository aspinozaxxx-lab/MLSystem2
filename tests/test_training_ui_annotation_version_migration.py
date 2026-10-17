"""Миграция хранит варианты в UI-БД и не меняет прежние черновики."""

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
import sqlalchemy as sa


def test_annotation_version_migration_preserves_drafts_and_protects_data(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    path = Path(__file__).parents[1] / "migrations/versions/20261017_0034_annotation_versions.py"
    spec = importlib.util.spec_from_file_location("annotation_version_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        for name in ("dataset_classes", "datasets", "dataset_editor_drafts"):
            connection.execute(sa.text(f"CREATE TABLE {name} (id VARCHAR(32) PRIMARY KEY)"))
        connection.execute(sa.text("INSERT INTO dataset_editor_drafts (id) VALUES ('сохранённый-черновик')"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
            assert connection.execute(sa.text("SELECT id, annotation_versions FROM dataset_editor_drafts")).all() == [("сохранённый-черновик", None)]
            assert {"annotation_images", "annotation_versions", "dataset_annotations", "annotation_publications"} <= set(sa.inspect(connection).get_table_names())
            connection.execute(sa.text("INSERT INTO annotation_images (id, imagery_type, relative_path, annotation_name) VALUES ('снимок', 'kanopus', 'папка/снимок.tif', 'папка_снимок.geojson')"))
            connection.execute(sa.text("INSERT INTO annotation_versions (id, image_id, class_id, name, geojson, revision) VALUES ('версия', 'снимок', 'класс', 'Основная', '{}', 1)"))
            with pytest.raises(RuntimeError, match="Откат удалит версии"):
                migration.downgrade()
            assert connection.execute(sa.text("SELECT name FROM annotation_versions")).scalar_one() == "Основная"
            connection.execute(sa.text("DELETE FROM annotation_versions"))
            migration.downgrade()
            assert connection.execute(sa.text("SELECT id FROM dataset_editor_drafts")).scalar_one() == "сохранённый-черновик"

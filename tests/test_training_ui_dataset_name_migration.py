"""Перенос уникальности активных наборов с сохранением архивных строк и ссылок."""

from pathlib import Path
import runpy

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.exc import IntegrityError


@pytest.fixture
def migration_environment(monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    namespace = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "migrations"
            / "versions"
            / "20261012_0032_active_dataset_uniqueness.py"
        )
    )
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    datasets = sa.Table(
        "datasets",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key", sa.String(), unique=True),
        sa.Column("class_id", sa.String()),
        sa.Column("name", sa.String()),
        sa.Column("source_type", sa.String()),
        sa.Column("source_path", sa.String()),
        sa.Column("deleted_at", sa.String()),
        sa.UniqueConstraint("source_type", "source_path", name="uq_datasets_source"),
        sa.UniqueConstraint("class_id", "name", name="uq_datasets_class_name"),
    )
    history = sa.Table(
        "history",
        metadata,
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("dataset_key", sa.String(), sa.ForeignKey("datasets.key")),
    )
    metadata.create_all(engine)
    with engine.connect() as connection:
        connection.execute(
            datasets.insert(),
            [
                {
                    "id": 1,
                    "key": "active",
                    "class_id": "rivers",
                    "name": "main",
                    "source_type": "mlmarkup",
                    "source_path": "Реки/main",
                    "deleted_at": None,
                },
                {
                    "id": 2,
                    "key": "old",
                    "class_id": "rivers",
                    "name": "test",
                    "source_type": "mlmarkup",
                    "source_path": "Реки/test",
                    "deleted_at": "2026-10-11",
                },
            ],
        )
        connection.execute(history.insert().values(id=1, dataset_key="old"))
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            yield connection, datasets, history, namespace
    engine.dispose()


def test_dataset_name_migration_preserves_history_and_allows_reuse(migration_environment):
    connection, datasets, history, migration = migration_environment
    before = connection.execute(sa.select(datasets).order_by(datasets.c.id)).all()
    migration["upgrade"]()
    assert connection.execute(sa.select(datasets).order_by(datasets.c.id)).all() == before
    connection.execute(
        datasets.insert().values(
            id=3,
            key="new",
            class_id="rivers",
            name="test",
            source_type="mlmarkup",
            source_path="Реки/test",
            deleted_at=None,
        )
    )
    assert connection.execute(sa.select(history.c.dataset_key)).scalar_one() == "old"
    for name, path in [("test", "Реки/another"), ("another", "Реки/test")]:
        with pytest.raises(IntegrityError), connection.begin_nested():
            connection.execute(
                datasets.insert().values(
                    id=4,
                    key="duplicate",
                    class_id="rivers",
                    name=name,
                    source_type="mlmarkup",
                    source_path=path,
                    deleted_at=None,
                )
            )
    with pytest.raises(RuntimeError, match="Откат невозможен"):
        migration["downgrade"]()
    assert {index["name"] for index in sa.inspect(connection).get_indexes("datasets")} == {
        "uq_datasets_active_source",
        "uq_datasets_active_class_name",
    }


def test_dataset_name_migration_can_downgrade_before_name_reuse(migration_environment):
    connection, datasets, _, migration = migration_environment
    before = connection.execute(sa.select(datasets).order_by(datasets.c.id)).all()
    migration["upgrade"]()
    migration["downgrade"]()
    assert connection.execute(sa.select(datasets).order_by(datasets.c.id)).all() == before
    assert {
        constraint["name"]
        for constraint in sa.inspect(connection).get_unique_constraints("datasets")
    } >= {
        "uq_datasets_source",
        "uq_datasets_class_name",
    }

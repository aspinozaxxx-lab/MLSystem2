"""Освободить названия и пути архивных датасетов без удаления их истории."""

import os

import sqlalchemy as sa
from alembic import op

revision = "20261012_0032"
down_revision = "20261007_0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    with op.batch_alter_table("datasets", schema=schema) as batch:
        batch.drop_constraint("uq_datasets_source", type_="unique")
        batch.drop_constraint("uq_datasets_class_name", type_="unique")
    for name, columns in [
        ("uq_datasets_active_source", ["source_type", "source_path"]),
        ("uq_datasets_active_class_name", ["class_id", "name"]),
    ]:
        op.create_index(
            name,
            "datasets",
            columns,
            schema=schema,
            unique=True,
            postgresql_where=sa.text("deleted_at IS NULL"),
            sqlite_where=sa.text("deleted_at IS NULL"),
        )


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        dataset = sa.table(
            "datasets",
            sa.column("class_id"),
            sa.column("name"),
            sa.column("source_type"),
            sa.column("source_path"),
            schema=schema,
        )
        for columns in [
            (dataset.c.class_id, dataset.c.name),
            (dataset.c.source_type, dataset.c.source_path),
        ]:
            duplicate = (
                op.get_bind()
                .execute(
                    sa.select(sa.func.count())
                    .select_from(dataset)
                    .group_by(*columns)
                    .having(sa.func.count() > 1)
                    .limit(1)
                )
                .first()
            )
            if duplicate is not None:
                raise RuntimeError(
                    "Откат невозможен: имена или пути уже использованы повторно. "
                    "История датасетов сохранена без изменений."
                )
    for name in ["uq_datasets_active_source", "uq_datasets_active_class_name"]:
        op.drop_index(name, table_name="datasets", schema=schema)
    with op.batch_alter_table("datasets", schema=schema) as batch:
        batch.create_unique_constraint("uq_datasets_source", ["source_type", "source_path"])
        batch.create_unique_constraint("uq_datasets_class_name", ["class_id", "name"])

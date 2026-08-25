"""Зафиксировать сеть и псевдоразметку источника тестовой разметки.

Revision ID: 20260825_0027
Revises: 20260820_0026
Create Date: 2026-08-25
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260825_0027"
down_revision = "20260820_0026"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "test_samples",
        sa.Column("source_training_result_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "test_samples",
        sa.Column("source_pseudo_result_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.create_foreign_key(
        "fk_test_samples_source_training_result_id_training_results",
        "test_samples",
        "training_results",
        ["source_training_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_test_samples_source_pseudo_result_id_pseudo_markup_results",
        "test_samples",
        "pseudo_markup_results",
        ["source_pseudo_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_test_samples_source_training_result_id",
        "test_samples",
        ["source_training_result_id"],
        schema=schema,
    )
    op.create_index(
        "ix_test_samples_source_pseudo_result_id",
        "test_samples",
        ["source_pseudo_result_id"],
        schema=schema,
    )

    op.add_column(
        "test_sample_batch_items",
        sa.Column("training_result_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "test_sample_batch_items",
        sa.Column("pseudo_markup_result_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.create_foreign_key(
        "fk_test_batch_items_training_result",
        "test_sample_batch_items",
        "training_results",
        ["training_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_test_batch_items_pseudo_result",
        "test_sample_batch_items",
        "pseudo_markup_results",
        ["pseudo_markup_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )

    test_samples = sa.table(
        "test_samples",
        sa.column("dataset_key", sa.String()),
        sa.column("evaluation_pseudo_result_id", sa.Uuid()),
        sa.column("source_training_result_id", sa.Uuid()),
        sa.column("source_pseudo_result_id", sa.Uuid()),
        schema=schema,
    )
    pseudo_results = sa.table(
        "pseudo_markup_results",
        sa.column("id", sa.Uuid()),
        sa.column("dataset_key", sa.String()),
        sa.column("training_result_id", sa.Uuid()),
        schema=schema,
    )
    connection = op.get_bind()
    matching_source = sa.exists(
        sa.select(pseudo_results.c.id).where(
            pseudo_results.c.id == test_samples.c.evaluation_pseudo_result_id,
            pseudo_results.c.dataset_key == test_samples.c.dataset_key,
        )
    )
    connection.execute(
        sa.update(test_samples)
        .where(
            test_samples.c.evaluation_pseudo_result_id.is_not(None),
            matching_source,
        )
        .values(source_pseudo_result_id=test_samples.c.evaluation_pseudo_result_id)
    )
    source_training_result = (
        sa.select(pseudo_results.c.training_result_id)
        .where(pseudo_results.c.id == test_samples.c.source_pseudo_result_id)
        .scalar_subquery()
    )
    connection.execute(
        sa.update(test_samples)
        .where(test_samples.c.source_pseudo_result_id.is_not(None))
        .values(source_training_result_id=source_training_result)
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "fk_test_batch_items_pseudo_result",
        "test_sample_batch_items",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_test_batch_items_training_result",
        "test_sample_batch_items",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_column("test_sample_batch_items", "pseudo_markup_result_id", schema=schema)
    op.drop_column("test_sample_batch_items", "training_result_id", schema=schema)

    op.drop_index(
        "ix_test_samples_source_pseudo_result_id",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_index(
        "ix_test_samples_source_training_result_id",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_constraint(
        "fk_test_samples_source_pseudo_result_id_pseudo_markup_results",
        "test_samples",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_test_samples_source_training_result_id_training_results",
        "test_samples",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_column("test_samples", "source_pseudo_result_id", schema=schema)
    op.drop_column("test_samples", "source_training_result_id", schema=schema)

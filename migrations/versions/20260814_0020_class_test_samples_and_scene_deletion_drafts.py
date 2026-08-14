"""Добавить черновое удаление снимка и основную тестовую разметку класса.

Revision ID: 20260814_0020
Revises: 20260813_0019
Create Date: 2026-08-14
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260814_0020"
down_revision = "20260813_0019"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "dataset_editor_drafts",
        sa.Column("deleted", sa.Boolean(), server_default=sa.false(), nullable=False),
        schema=schema,
    )
    prefix = f'"{schema}".' if schema else ""
    op.execute(
        sa.text(
            f"""
            UPDATE {prefix}test_samples AS sample
            SET class_key = class_row.key,
                class_name = class_row.name
            FROM {prefix}datasets AS dataset,
                 {prefix}dataset_classes AS class_row
            WHERE sample.dataset_key = dataset.key
              AND dataset.class_id = class_row.id
            """
        )
    )
    op.execute(
        sa.text(
            f"""
            WITH ranked AS (
                SELECT sample.id,
                       ROW_NUMBER() OVER (
                           PARTITION BY sample.class_key
                           ORDER BY
                               CASE WHEN sample.dataset_key = primary_dataset.key THEN 0 ELSE 1 END,
                               sample.updated_at DESC,
                               sample.created_at DESC,
                               sample.id
                       ) AS position
                FROM {prefix}test_samples AS sample
                LEFT JOIN {prefix}dataset_classes AS class_row
                  ON class_row.key = sample.class_key
                LEFT JOIN {prefix}datasets AS primary_dataset
                  ON primary_dataset.id = class_row.primary_dataset_id
                WHERE sample.is_primary
            )
            UPDATE {prefix}test_samples
            SET is_primary = false
            WHERE id IN (SELECT id FROM ranked WHERE position > 1)
            """
        )
    )
    op.drop_index(
        "uq_test_samples_primary_dataset_key",
        table_name="test_samples",
        schema=schema,
    )
    op.create_index(
        "uq_test_samples_primary_class_key",
        "test_samples",
        ["class_key"],
        unique=True,
        postgresql_where=sa.text("is_primary"),
        sqlite_where=sa.text("is_primary = 1"),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "uq_test_samples_primary_class_key",
        table_name="test_samples",
        schema=schema,
    )
    op.create_index(
        "uq_test_samples_primary_dataset_key",
        "test_samples",
        ["dataset_key"],
        unique=True,
        postgresql_where=sa.text("is_primary"),
        sqlite_where=sa.text("is_primary = 1"),
        schema=schema,
    )
    op.drop_column("dataset_editor_drafts", "deleted", schema=schema)

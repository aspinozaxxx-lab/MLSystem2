"""Добавить основную обученную сеть класса.

Revision ID: 20260804_0014
Revises: 20260804_0013
Create Date: 2026-08-04
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260804_0014"
down_revision = "20260804_0013"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "dataset_classes",
        sa.Column("primary_training_result_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.create_foreign_key(
        "fk_dataset_classes_primary_training_result",
        "dataset_classes",
        "training_results",
        ["primary_training_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    prefix = f'"{schema}".' if schema else ""
    op.execute(
        sa.text(
            f"""
            WITH ranked AS (
                SELECT results.id, datasets.class_id,
                       ROW_NUMBER() OVER (
                           PARTITION BY datasets.class_id
                           ORDER BY results.trained_at DESC NULLS LAST,
                                    results.created_at DESC,
                                    results.id DESC
                       ) AS position
                FROM {prefix}training_results AS results
                JOIN {prefix}datasets AS datasets
                  ON datasets.key = COALESCE(results.dataset_key, results.class_key)
                WHERE results.status = 'ok'
            )
            UPDATE {prefix}dataset_classes AS classes
            SET primary_training_result_id = ranked.id
            FROM ranked
            WHERE ranked.class_id = classes.id AND ranked.position = 1
            """
        )
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_constraint(
        "fk_dataset_classes_primary_training_result",
        "dataset_classes",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_column("dataset_classes", "primary_training_result_id", schema=schema)

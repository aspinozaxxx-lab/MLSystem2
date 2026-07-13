"""Добавить минимальное число тайлов групповой тестовой выборки.

Revision ID: 20260713_0010
Revises: 20260713_0009
Create Date: 2026-07-13
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260713_0010"
down_revision = "20260713_0009"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def _table(name: str) -> str:
    schema = _schema()
    return f"{schema}.{name}" if schema else name


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "test_sample_batches",
        sa.Column("min_image_count", sa.Integer(), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"UPDATE {_table('test_sample_batches')} "
            "SET min_image_count = image_count WHERE min_image_count IS NULL"
        )
    )
    op.alter_column(
        "test_sample_batches",
        "min_image_count",
        existing_type=sa.Integer(),
        nullable=False,
        schema=schema,
    )


def downgrade() -> None:
    op.drop_column(
        "test_sample_batches",
        "min_image_count",
        schema=_schema(),
    )

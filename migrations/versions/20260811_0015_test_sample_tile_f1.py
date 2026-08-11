"""Сохранять пиксельный и объектовый F1 каждого тестового тайла.

Revision ID: 20260811_0015
Revises: 20260804_0014
Create Date: 2026-08-11
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260811_0015"
down_revision = "20260804_0014"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "test_sample_tiles",
        sa.Column("pixel_f1", sa.Float(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "test_sample_tiles",
        sa.Column("object_f1", sa.Float(), nullable=True),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_column("test_sample_tiles", "object_f1", schema=schema)
    op.drop_column("test_sample_tiles", "pixel_f1", schema=schema)

"""Добавить мягкое удаление управляемых датасетов.

Revision ID: 20260812_0017
Revises: 20260812_0016
Create Date: 2026-08-12
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260812_0017"
down_revision = "20260812_0016"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "datasets",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        schema=schema,
    )
    op.create_index(
        "ix_datasets_deleted_at",
        "datasets",
        ["deleted_at"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "ix_datasets_deleted_at",
        table_name="datasets",
        schema=schema,
    )
    op.drop_column("datasets", "deleted_at", schema=schema)

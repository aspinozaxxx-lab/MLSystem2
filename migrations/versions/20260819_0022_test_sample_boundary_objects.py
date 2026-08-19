"""Добавить исключение граничных объектов тестовой разметки.

Revision ID: 20260819_0022
Revises: 20260815_0021
Create Date: 2026-08-19
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260819_0022"
down_revision = "20260815_0021"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "test_samples",
        sa.Column(
            "exclude_boundary_objects",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        schema=schema,
    )
    op.add_column(
        "test_sample_batch_items",
        sa.Column(
            "exclude_boundary_objects",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_column(
        "test_sample_batch_items",
        "exclude_boundary_objects",
        schema=schema,
    )
    op.drop_column(
        "test_samples",
        "exclude_boundary_objects",
        schema=schema,
    )

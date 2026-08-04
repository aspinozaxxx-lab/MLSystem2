"""Сохранять текст ошибки задания.

Revision ID: 20260804_0013
Revises: 20260720_0012
Create Date: 2026-08-04
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260804_0013"
down_revision = "20260720_0012"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    op.add_column(
        "jobs",
        sa.Column("error", sa.Text(), nullable=True),
        schema=_schema(),
    )


def downgrade() -> None:
    op.drop_column("jobs", "error", schema=_schema())

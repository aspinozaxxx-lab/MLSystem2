"""Store generated GeoJSON object counts.

Revision ID: 20260625_0007
Revises: 20260625_0006
Create Date: 2026-06-25
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260625_0007"
down_revision = "20260625_0006"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    op.add_column("stored_files", sa.Column("object_count", sa.Integer(), nullable=True), schema=_schema())


def downgrade() -> None:
    op.drop_column("stored_files", "object_count", schema=_schema())

"""Store pseudo-markup image counts.

Revision ID: 20260625_0006
Revises: 20260611_0005
Create Date: 2026-06-25
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260625_0006"
down_revision = "20260611_0005"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    op.add_column("pseudo_markup_results", sa.Column("image_count", sa.Integer(), nullable=True), schema=_schema())


def downgrade() -> None:
    op.drop_column("pseudo_markup_results", "image_count", schema=_schema())

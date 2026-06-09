"""stored file size bigint

Revision ID: 20260609_0004
Revises: 20260606_0003
Create Date: 2026-06-09
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260609_0004"
down_revision = "20260606_0003"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.alter_column(
        "stored_files",
        "size_bytes",
        existing_type=sa.Integer(),
        type_=sa.BigInteger(),
        existing_nullable=False,
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.alter_column(
        "stored_files",
        "size_bytes",
        existing_type=sa.BigInteger(),
        type_=sa.Integer(),
        existing_nullable=False,
        schema=schema,
    )

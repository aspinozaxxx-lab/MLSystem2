"""Добавить серверные черновики редактора и ключ идемпотентных заданий.

Revision ID: 20260813_0019
Revises: 20260812_0018
Create Date: 2026-08-13
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260813_0019"
down_revision = "20260812_0018"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.create_table(
        "dataset_editor_drafts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("dataset_key", sa.String(length=180), nullable=False),
        sa.Column("annotation_name", sa.String(length=512), nullable=False),
        sa.Column("username", sa.String(length=180), nullable=False),
        sa.Column("base_revision", sa.String(length=128), nullable=False),
        sa.Column("geojson", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_key",
            "annotation_name",
            "username",
            name="uq_dataset_editor_drafts_owner_scene",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_dataset_editor_drafts_dataset_owner",
        "dataset_editor_drafts",
        ["dataset_key", "username"],
        schema=schema,
    )
    op.add_column(
        "jobs",
        sa.Column("dedup_key", sa.String(length=64), nullable=True),
        schema=schema,
    )
    op.create_index(
        "ix_jobs_dedup_key",
        "jobs",
        ["dedup_key"],
        unique=True,
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index("ix_jobs_dedup_key", table_name="jobs", schema=schema)
    op.drop_column("jobs", "dedup_key", schema=schema)
    op.drop_index(
        "ix_dataset_editor_drafts_dataset_owner",
        table_name="dataset_editor_drafts",
        schema=schema,
    )
    op.drop_table("dataset_editor_drafts", schema=schema)

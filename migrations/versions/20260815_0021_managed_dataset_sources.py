"""Добавить состав виртуальных управляемых датасетов.

Revision ID: 20260815_0021
Revises: 20260814_0020
Create Date: 2026-08-15
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260815_0021"
down_revision = "20260814_0020"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    prefix = f"{schema}." if schema else ""
    op.create_table(
        "managed_dataset_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("managed_dataset_id", sa.Uuid(), nullable=False),
        sa.Column("source_dataset_id", sa.Uuid(), nullable=False),
        sa.Column("priority", sa.Integer(), server_default="0", nullable=False),
        sa.Column("object_type_id", sa.Integer(), nullable=False),
        sa.Column("object_type_slug", sa.String(length=180), nullable=False),
        sa.Column("object_type_name", sa.String(length=240), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["managed_dataset_id"],
            [f"{prefix}datasets.id"],
            name="fk_managed_dataset_sources_target",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_dataset_id"],
            [f"{prefix}datasets.id"],
            name="fk_managed_dataset_sources_source",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "managed_dataset_id",
            "source_dataset_id",
            name="uq_managed_dataset_sources_pair",
        ),
        sa.UniqueConstraint(
            "managed_dataset_id",
            "object_type_slug",
            name="uq_managed_dataset_sources_slug",
        ),
        sa.UniqueConstraint(
            "managed_dataset_id",
            "object_type_id",
            name="uq_managed_dataset_sources_type_id",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_managed_dataset_sources_target",
        "managed_dataset_sources",
        ["managed_dataset_id"],
        schema=schema,
    )
    op.create_table(
        "managed_dataset_scenes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("managed_dataset_id", sa.Uuid(), nullable=False),
        sa.Column("annotation_name", sa.String(length=512), nullable=False),
        sa.Column("image_relative_path", sa.String(length=2048), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["managed_dataset_id"],
            [f"{prefix}datasets.id"],
            name="fk_managed_dataset_scenes_target",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "managed_dataset_id",
            "annotation_name",
            name="uq_managed_dataset_scenes_annotation",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_managed_dataset_scenes_target",
        "managed_dataset_scenes",
        ["managed_dataset_id"],
        schema=schema,
    )
    op.create_index(
        "ix_managed_dataset_sources_source",
        "managed_dataset_sources",
        ["source_dataset_id"],
        schema=schema,
    )


def downgrade() -> None:
    op.drop_table("managed_dataset_scenes", schema=_schema())
    op.drop_table("managed_dataset_sources", schema=_schema())

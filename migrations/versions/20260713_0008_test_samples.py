"""Добавить постоянные тестовые выборки и их тайлы.

Revision ID: 20260713_0008
Revises: 20260625_0007
Create Date: 2026-07-13
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260713_0008"
down_revision = "20260625_0007"
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
    op.create_table(
        "test_samples",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("dataset_key", sa.String(length=180), nullable=False),
        sa.Column("dataset_name", sa.String(length=240), nullable=False),
        sa.Column("dataset_version", sa.String(length=160), nullable=True),
        sa.Column("class_key", sa.String(length=180), nullable=False),
        sa.Column("class_name", sa.String(length=240), nullable=False),
        sa.Column("variant_key", sa.String(length=180), nullable=False),
        sa.Column("variant_name", sa.String(length=240), nullable=False),
        sa.Column("tile_width", sa.Integer(), nullable=False),
        sa.Column("tile_height", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("requested_object_count", sa.Integer(), nullable=False),
        sa.Column("actual_object_count", sa.Integer(), nullable=False),
        sa.Column("territory_count", sa.Integer(), nullable=False),
        sa.Column("warnings", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("content_revision", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("evaluated_revision", sa.Integer(), nullable=True),
        sa.Column("metric_status", sa.String(length=32), nullable=False, server_default="unavailable"),
        sa.Column("object_iou_threshold", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("pixel_precision", sa.Float(), nullable=True),
        sa.Column("pixel_recall", sa.Float(), nullable=True),
        sa.Column("pixel_f1", sa.Float(), nullable=True),
        sa.Column("pixel_true_positive", sa.BigInteger(), nullable=True),
        sa.Column("pixel_false_positive", sa.BigInteger(), nullable=True),
        sa.Column("pixel_false_negative", sa.BigInteger(), nullable=True),
        sa.Column("object_precision", sa.Float(), nullable=True),
        sa.Column("object_recall", sa.Float(), nullable=True),
        sa.Column("object_f1", sa.Float(), nullable=True),
        sa.Column("object_true_positive", sa.BigInteger(), nullable=True),
        sa.Column("object_false_positive", sa.BigInteger(), nullable=True),
        sa.Column("object_false_negative", sa.BigInteger(), nullable=True),
        sa.Column(
            "evaluation_pseudo_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("pseudo_markup_results") + ".id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("evaluation_model_name", sa.String(length=160), nullable=True),
        sa.Column("evaluation_markup_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("evaluation_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_index("ix_test_samples_dataset_key", "test_samples", ["dataset_key"], schema=schema)
    op.create_index(
        "ix_test_samples_class_variant",
        "test_samples",
        ["class_key", "variant_key"],
        schema=schema,
    )
    op.create_index("ix_test_samples_created_at", "test_samples", ["created_at"], schema=schema)

    op.create_table(
        "test_sample_tiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "test_sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("test_samples") + ".id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tile_index", sa.Integer(), nullable=False),
        sa.Column("source_name", sa.String(length=512), nullable=False),
        sa.Column("territory", sa.String(length=512), nullable=False),
        sa.Column("object_count", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "test_sample_id",
            "tile_index",
            name="uq_test_sample_tiles_sample_index",
        ),
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_tiles_sample_id",
        "test_sample_tiles",
        ["test_sample_id"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index("ix_test_sample_tiles_sample_id", table_name="test_sample_tiles", schema=schema)
    op.drop_table("test_sample_tiles", schema=schema)
    op.drop_index("ix_test_samples_created_at", table_name="test_samples", schema=schema)
    op.drop_index("ix_test_samples_class_variant", table_name="test_samples", schema=schema)
    op.drop_index("ix_test_samples_dataset_key", table_name="test_samples", schema=schema)
    op.drop_table("test_samples", schema=schema)

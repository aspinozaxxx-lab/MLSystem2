"""Добавить групповые тестовые выборки, основные выборки и тестовый F1 сетей.

Revision ID: 20260713_0009
Revises: 20260713_0008
Create Date: 2026-07-13
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260713_0009"
down_revision = "20260713_0008"
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
    op.add_column(
        "test_samples",
        sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        schema=schema,
    )
    op.create_index(
        "uq_test_samples_primary_dataset_key",
        "test_samples",
        ["dataset_key"],
        unique=True,
        schema=schema,
        postgresql_where=sa.text("is_primary"),
    )

    op.create_table(
        "test_sample_batches",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("active_slot", sa.Integer(), nullable=True, server_default=sa.text("1")),
        sa.Column("tile_size", sa.Integer(), nullable=False),
        sa.Column("image_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("active_slot", name="uq_test_sample_batches_active_slot"),
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_batches_status",
        "test_sample_batches",
        ["status"],
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_batches_created_at",
        "test_sample_batches",
        ["created_at"],
        schema=schema,
    )

    op.create_table(
        "test_sample_batch_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "batch_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("test_sample_batches") + ".id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("dataset_key", sa.String(length=180), nullable=False),
        sa.Column("dataset_name", sa.String(length=240), nullable=False),
        sa.Column("dataset_version", sa.String(length=160), nullable=True),
        sa.Column("class_key", sa.String(length=180), nullable=False),
        sa.Column("class_name", sa.String(length=240), nullable=False),
        sa.Column("variant_key", sa.String(length=180), nullable=False),
        sa.Column("variant_name", sa.String(length=240), nullable=False),
        sa.Column("min_object_count", sa.Integer(), nullable=False),
        sa.Column("metric", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("pool_tile_count", sa.Integer(), nullable=True),
        sa.Column("pool_object_count", sa.Integer(), nullable=True),
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("test_samples") + ".id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("batch_id", "position", name="uq_test_sample_batch_items_position"),
        sa.UniqueConstraint("batch_id", "dataset_key", name="uq_test_sample_batch_items_dataset"),
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_batch_items_batch_id",
        "test_sample_batch_items",
        ["batch_id"],
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_batch_items_status",
        "test_sample_batch_items",
        ["status"],
        schema=schema,
    )

    op.create_table(
        "training_result_test_metrics",
        sa.Column(
            "training_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("training_results") + ".id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "sample_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("test_samples") + ".id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("sample_revision", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="unavailable"),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("jobs") + ".id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("f1", sa.Float(), nullable=True),
        sa.Column("true_positive", sa.BigInteger(), nullable=True),
        sa.Column("false_positive", sa.BigInteger(), nullable=True),
        sa.Column("false_negative", sa.BigInteger(), nullable=True),
        sa.Column(
            "inference_template_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("inference_templates") + ".id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("inference_template_version", sa.Integer(), nullable=True),
        sa.Column("inference_config_hash", sa.String(length=64), nullable=True),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_index(
        "ix_training_result_test_metrics_sample_id",
        "training_result_test_metrics",
        ["sample_id"],
        schema=schema,
    )
    op.create_index(
        "ix_training_result_test_metrics_status",
        "training_result_test_metrics",
        ["status"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "ix_training_result_test_metrics_status",
        table_name="training_result_test_metrics",
        schema=schema,
    )
    op.drop_index(
        "ix_training_result_test_metrics_sample_id",
        table_name="training_result_test_metrics",
        schema=schema,
    )
    op.drop_table("training_result_test_metrics", schema=schema)
    op.drop_index(
        "ix_test_sample_batch_items_status",
        table_name="test_sample_batch_items",
        schema=schema,
    )
    op.drop_index(
        "ix_test_sample_batch_items_batch_id",
        table_name="test_sample_batch_items",
        schema=schema,
    )
    op.drop_table("test_sample_batch_items", schema=schema)
    op.drop_index(
        "ix_test_sample_batches_created_at",
        table_name="test_sample_batches",
        schema=schema,
    )
    op.drop_index(
        "ix_test_sample_batches_status",
        table_name="test_sample_batches",
        schema=schema,
    )
    op.drop_table("test_sample_batches", schema=schema)
    op.drop_index(
        "uq_test_samples_primary_dataset_key",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_column("test_samples", "is_primary", schema=schema)

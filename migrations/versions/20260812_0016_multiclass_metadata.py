"""Добавить схему классов и структурированные метрики.

Revision ID: 20260812_0016
Revises: 20260811_0015
Create Date: 2026-08-12
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260812_0016"
down_revision = "20260811_0015"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def _json_column(name: str, default: str) -> sa.Column:
    return sa.Column(
        name,
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
        server_default=sa.text(default),
    )


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "training_results",
        sa.Column("task", sa.String(length=32), nullable=False, server_default="binary"),
        schema=schema,
    )
    op.add_column("training_results", _json_column("class_schema", "'[]'::jsonb"), schema=schema)
    op.add_column("training_results", _json_column("training_metrics", "'{}'::jsonb"), schema=schema)

    op.add_column(
        "test_samples",
        sa.Column("task", sa.String(length=32), nullable=False, server_default="binary"),
        schema=schema,
    )
    op.add_column("test_samples", _json_column("class_schema", "'[]'::jsonb"), schema=schema)
    op.add_column("test_samples", _json_column("evaluation_metrics", "'{}'::jsonb"), schema=schema)
    op.add_column("test_sample_tiles", _json_column("class_object_counts", "'{}'::jsonb"), schema=schema)
    op.add_column("test_sample_tiles", _json_column("evaluation_metrics", "'{}'::jsonb"), schema=schema)
    op.add_column(
        "training_result_test_metrics",
        _json_column("metrics", "'{}'::jsonb"),
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_column("training_result_test_metrics", "metrics", schema=schema)
    op.drop_column("test_sample_tiles", "evaluation_metrics", schema=schema)
    op.drop_column("test_sample_tiles", "class_object_counts", schema=schema)
    op.drop_column("test_samples", "evaluation_metrics", schema=schema)
    op.drop_column("test_samples", "class_schema", schema=schema)
    op.drop_column("test_samples", "task", schema=schema)
    op.drop_column("training_results", "training_metrics", schema=schema)
    op.drop_column("training_results", "class_schema", schema=schema)
    op.drop_column("training_results", "task", schema=schema)

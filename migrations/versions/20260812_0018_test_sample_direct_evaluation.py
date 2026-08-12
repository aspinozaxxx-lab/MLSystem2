"""Добавить источник прямой оценки тестовых разметок.

Revision ID: 20260812_0018
Revises: 20260812_0017
Create Date: 2026-08-12
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260812_0018"
down_revision = "20260812_0017"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    columns = (
        sa.Column("evaluation_training_result_id", sa.Uuid(), nullable=True),
        sa.Column("evaluation_job_id", sa.Uuid(), nullable=True),
        sa.Column("evaluation_inference_template_id", sa.Uuid(), nullable=True),
        sa.Column("evaluation_inference_template_version", sa.Integer(), nullable=True),
        sa.Column("evaluation_inference_config_hash", sa.String(length=64), nullable=True),
        sa.Column("evaluation_evaluator_version", sa.Integer(), nullable=True),
        sa.Column("evaluation_threshold", sa.Float(), nullable=True),
    )
    for column in columns:
        op.add_column("test_samples", column, schema=schema)
    op.create_foreign_key(
        "fk_test_samples_evaluation_training_result_id_training_results",
        "test_samples",
        "training_results",
        ["evaluation_training_result_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_test_samples_evaluation_job_id_jobs",
        "test_samples",
        "jobs",
        ["evaluation_job_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_test_samples_evaluation_inference_template",
        "test_samples",
        "inference_templates",
        ["evaluation_inference_template_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_test_samples_evaluation_training_result_id",
        "test_samples",
        ["evaluation_training_result_id"],
        schema=schema,
    )
    op.create_index(
        "ix_test_samples_evaluation_job_id",
        "test_samples",
        ["evaluation_job_id"],
        schema=schema,
    )


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "ix_test_samples_evaluation_job_id",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_index(
        "ix_test_samples_evaluation_training_result_id",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_constraint(
        "fk_test_samples_evaluation_inference_template",
        "test_samples",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_test_samples_evaluation_job_id_jobs",
        "test_samples",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_test_samples_evaluation_training_result_id_training_results",
        "test_samples",
        schema=schema,
        type_="foreignkey",
    )
    for name in (
        "evaluation_threshold",
        "evaluation_evaluator_version",
        "evaluation_inference_config_hash",
        "evaluation_inference_template_version",
        "evaluation_inference_template_id",
        "evaluation_job_id",
        "evaluation_training_result_id",
    ):
        op.drop_column("test_samples", name, schema=schema)

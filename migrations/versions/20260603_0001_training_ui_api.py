"""training ui api schema

Revision ID: 20260603_0001
Revises:
Create Date: 2026-06-03
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from mlsystem2.training_ui_api._templates import initial_templates


revision = "20260603_0001"
down_revision = None
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
    if schema:
        op.execute(sa.text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))

    op.create_table(
        "training_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("architecture", sa.String(length=96), nullable=False),
        sa.Column("display_name", sa.String(length=160), nullable=False),
        sa.Column("config_schema", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("default_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("baseline_default_config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("baseline_source", sa.String(length=32), nullable=False),
        sa.Column("source_mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("baseline_source_mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("architecture", name="uq_training_templates_architecture"),
        schema=schema,
    )
    op.create_index("ix_training_templates_architecture", "training_templates", ["architecture"], schema=schema)

    op.create_table(
        "stored_files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("kind", sa.String(length=48), nullable=False),
        sa.Column("original_name", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=160), nullable=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_index("ix_stored_files_kind", "stored_files", ["kind"], schema=schema)

    op.create_table(
        "custom_datasets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=180), nullable=False),
        sa.Column("scenes_file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("stored_files") + ".id"), nullable=False),
        sa.Column("annotation_file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("stored_files") + ".id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )

    op.create_table(
        "queue_controls",
        sa.Column("queue_name", sa.String(length=32), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )

    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("queue_position", sa.Integer(), nullable=False),
        sa.Column("dataset_name", sa.String(length=240), nullable=False),
        sa.Column("training_dataset_name", sa.String(length=240), nullable=True),
        sa.Column("inference_dataset_name", sa.String(length=240), nullable=True),
        sa.Column("model_name", sa.String(length=160), nullable=False),
        sa.Column("architecture", sa.String(length=96), nullable=False),
        sa.Column("tile_size", sa.Integer(), nullable=True),
        sa.Column("mlflow_experiment_id", sa.String(length=64), nullable=True),
        sa.Column("mlflow_experiment_name", sa.String(length=256), nullable=True),
        sa.Column("mlflow_run_name", sa.String(length=256), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("custom_dataset_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("custom_datasets") + ".id"), nullable=True),
        sa.Column("process_pid", sa.Integer(), nullable=True),
        sa.Column("tmp_path", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        schema=schema,
    )
    op.create_index("ix_jobs_type", "jobs", ["type"], schema=schema)
    op.create_index("ix_jobs_status", "jobs", ["status"], schema=schema)
    op.create_index("ix_jobs_queue_position", "jobs", ["queue_position"], schema=schema)

    op.create_table(
        "training_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("class_key", sa.String(length=180), nullable=False),
        sa.Column("class_display_name", sa.String(length=240), nullable=False),
        sa.Column("architecture", sa.String(length=96), nullable=False),
        sa.Column("model_name", sa.String(length=160), nullable=False),
        sa.Column("f1_score", sa.Float(), nullable=True),
        sa.Column("epoch", sa.Integer(), nullable=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("mlflow_run_url", sa.Text(), nullable=True),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("jobs") + ".id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_index("ix_training_results_class_key", "training_results", ["class_key"], schema=schema)
    op.create_index("ix_training_results_status", "training_results", ["status"], schema=schema)

    op.create_table(
        "pseudo_markup_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("training_result_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("training_results") + ".id"), nullable=True),
        sa.Column("class_key", sa.String(length=180), nullable=False),
        sa.Column("source_dataset_name", sa.String(length=240), nullable=False),
        sa.Column("scenes_file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("stored_files") + ".id"), nullable=True),
        sa.Column("geojson_file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("stored_files") + ".id"), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey(_table("jobs") + ".id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_index("ix_pseudo_markup_results_class_key", "pseudo_markup_results", ["class_key"], schema=schema)
    op.create_index("ix_pseudo_markup_results_status", "pseudo_markup_results", ["status"], schema=schema)

    if not op.get_context().as_sql:
        _seed_templates(schema)
        op.bulk_insert(
            sa.table(
                "queue_controls",
                sa.column("queue_name", sa.String),
                sa.column("enabled", sa.Boolean),
                sa.column("updated_at", sa.DateTime(timezone=True)),
                schema=schema,
            ),
            [
                {"queue_name": "training", "enabled": True, "updated_at": datetime.now(timezone.utc)},
                {"queue_name": "inference", "enabled": True, "updated_at": datetime.now(timezone.utc)},
            ],
        )


def downgrade() -> None:
    schema = _schema()
    op.drop_index("ix_pseudo_markup_results_status", table_name="pseudo_markup_results", schema=schema)
    op.drop_index("ix_pseudo_markup_results_class_key", table_name="pseudo_markup_results", schema=schema)
    op.drop_table("pseudo_markup_results", schema=schema)
    op.drop_index("ix_training_results_status", table_name="training_results", schema=schema)
    op.drop_index("ix_training_results_class_key", table_name="training_results", schema=schema)
    op.drop_table("training_results", schema=schema)
    op.drop_index("ix_jobs_queue_position", table_name="jobs", schema=schema)
    op.drop_index("ix_jobs_status", table_name="jobs", schema=schema)
    op.drop_index("ix_jobs_type", table_name="jobs", schema=schema)
    op.drop_table("jobs", schema=schema)
    op.drop_table("queue_controls", schema=schema)
    op.drop_table("custom_datasets", schema=schema)
    op.drop_index("ix_stored_files_kind", table_name="stored_files", schema=schema)
    op.drop_table("stored_files", schema=schema)
    op.drop_index("ix_training_templates_architecture", table_name="training_templates", schema=schema)
    op.drop_table("training_templates", schema=schema)


def _seed_templates(schema: str | None) -> None:
    table = sa.table(
        "training_templates",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("architecture", sa.String),
        sa.column("display_name", sa.String),
        sa.column("config_schema", postgresql.JSONB),
        sa.column("default_config", postgresql.JSONB),
        sa.column("baseline_default_config", postgresql.JSONB),
        sa.column("source", sa.String),
        sa.column("baseline_source", sa.String),
        sa.column("source_mlflow_run_id", sa.String),
        sa.column("baseline_source_mlflow_run_id", sa.String),
        sa.column("is_active", sa.Boolean),
        sa.column("version", sa.Integer),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
        schema=schema,
    )
    now = datetime.now(timezone.utc)
    rows = []
    for payload in initial_templates():
        architecture = payload["architecture"]
        rows.append(
            {
                **payload,
                "id": uuid.uuid5(uuid.NAMESPACE_URL, f"mlsystem2-training-template:{architecture}"),
                "created_at": now,
                "updated_at": now,
            }
        )
    op.bulk_insert(table, rows)

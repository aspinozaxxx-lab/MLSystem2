"""шаблоны инференса

Revision ID: 20260611_0005
Revises: 20260609_0004
Create Date: 2026-06-11
"""

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from typing import Any
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from mlsystem2.training_ui_api._templates import (
    INFERENCE_BASE_DEFAULT_CONFIG, INFERENCE_CONFIG_SCHEMA,
    COMPACT_FILTER_REMOVE, COMPACT_FILTER_KEEP,
)


revision = "20260611_0005"
down_revision = "20260609_0004"
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
        "inference_templates",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("architecture", sa.String(length=96), nullable=False),
        sa.Column("dataset_key", sa.String(length=180), nullable=True),
        sa.Column("dataset_name", sa.String(length=240), nullable=True),
        sa.Column(
            "parent_template_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(_table("inference_templates") + ".id"),
            nullable=True,
        ),
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
        sa.UniqueConstraint("architecture", "dataset_key", name="uq_inference_templates_architecture_dataset"),
        schema=schema,
    )
    op.create_index("ix_inference_templates_architecture", "inference_templates", ["architecture"], schema=schema)
    op.create_index("ix_inference_templates_dataset_key", "inference_templates", ["dataset_key"], schema=schema)
    if not op.get_context().as_sql:
        _seed_inference_templates(schema)


def downgrade() -> None:
    schema = _schema()
    op.drop_index("ix_inference_templates_dataset_key", table_name="inference_templates", schema=schema)
    op.drop_index("ix_inference_templates_architecture", table_name="inference_templates", schema=schema)
    op.drop_table("inference_templates", schema=schema)


def _seed_inference_templates(schema: str | None) -> None:
    table = sa.table(
        "inference_templates",
        sa.column("id", postgresql.UUID(as_uuid=True)),
        sa.column("architecture", sa.String),
        sa.column("dataset_key", sa.String),
        sa.column("dataset_name", sa.String),
        sa.column("parent_template_id", postgresql.UUID(as_uuid=True)),
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
    for payload in _initial_inference_templates():
        architecture = payload["architecture"]
        dataset_key = payload.get("dataset_key")
        template_id = _template_id(architecture, dataset_key)
        parent_template_id = _template_id(architecture, None) if dataset_key else None
        rows.append(
            {
                **payload,
                "id": template_id,
                "parent_template_id": parent_template_id,
                "created_at": now,
                "updated_at": now,
            }
        )
    op.bulk_insert(table, rows)


def _template_id(architecture: str, dataset_key: str | None) -> uuid.UUID:
    if dataset_key:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"mlsystem2-inference-template:{architecture}:{dataset_key}")
    return uuid.uuid5(uuid.NAMESPACE_URL, f"mlsystem2-inference-template:{architecture}")


# Исторические данные этой миграции независимы от новой области шаблонов.
RIVERS_INFERENCE_CONFIG: dict[str, Any] = {
    "postprocess.min_area_m2": 10000.0,
    "postprocess.min_hole_area_m2": 5000.0,
    "postprocess.smooth.enabled": True,
    "postprocess.smooth.iterations": 1,
    "postprocess.smooth.offset": 0.125,
    "postprocess.simplify_m": 1.0,
    "postprocess.filter_compact_objects.enabled": True,
    "postprocess.filter_compact_objects.mode": COMPACT_FILTER_REMOVE,
    "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
    "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
}


LAKES_INFERENCE_CONFIG: dict[str, Any] = {
    "postprocess.filter_compact_objects.enabled": True,
    "postprocess.filter_compact_objects.mode": COMPACT_FILTER_KEEP,
    "postprocess.filter_compact_objects.min_isoperimetric_quotient": 0.25,
    "postprocess.filter_compact_objects.max_bbox_ratio": 3.5,
}


def _initial_inference_templates() -> list[dict[str, Any]]:
    rows = [
        _inference_template(
            "segformer_b0",
            "SegFormer B0 HF",
            source="analogy",
        ),
        _inference_template(
            "external_torchscript",
            "импортированная TorchScript-модель",
            source="manual",
        ),
        _inference_template(
            "smp_deeplabv3plus_resnet50",
            "deeplabV3+",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b0",
            "segformer b0",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b1",
            "segformer b1",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b2",
            "segformer b2",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b3",
            "segformer b3",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet34",
            "unet + resnet34",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet50",
            "unet + resnet50",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet101",
            "unet + resnet101",
            source="analogy",
        ),
        _inference_template(
            "smp_unet_resnet152",
            "unet + resnet152",
            source="analogy",
        ),
        _inference_template(
            "smp_segformer_b2",
            "segformer b2 / Реки\\main",
            source="analogy",
            dataset_key="Реки\\main",
            dataset_name="Реки\\main",
            overrides=RIVERS_INFERENCE_CONFIG,
        ),
        _inference_template(
            "smp_segformer_b2",
            "segformer b2 / Озера\\main",
            source="analogy",
            dataset_key="Озера\\main",
            dataset_name="Озера\\main",
            overrides=LAKES_INFERENCE_CONFIG,
        ),
    ]
    return rows


def _inference_template(
    architecture: str,
    display_name: str,
    *,
    source: str,
    dataset_key: str | None = None,
    dataset_name: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    default_config = deepcopy(INFERENCE_BASE_DEFAULT_CONFIG)
    if overrides:
        default_config.update(overrides)
    return {
        "architecture": architecture,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "parent_template_id": None,
        "display_name": display_name,
        "config_schema": deepcopy(INFERENCE_CONFIG_SCHEMA),
        "default_config": default_config,
        "baseline_default_config": deepcopy(default_config),
        "source": source,
        "baseline_source": source,
        "source_mlflow_run_id": None,
        "baseline_source_mlflow_run_id": None,
        "is_active": True,
        "version": 1,
    }

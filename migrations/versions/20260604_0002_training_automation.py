"""training automation

Revision ID: 20260604_0002
Revises: 20260603_0001
Create Date: 2026-06-04
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260604_0002"
down_revision = "20260603_0001"
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
        "automation_controls",
        sa.Column("key", sa.String(length=32), primary_key=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        schema=schema,
    )
    op.create_table(
        "automation_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("dataset_key", sa.String(length=180), nullable=False),
        sa.Column("architecture", sa.String(length=96), nullable=False),
        sa.Column("training_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("pseudo_markup_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("dataset_key", "architecture", name="uq_automation_rules_dataset_architecture"),
        schema=schema,
    )
    op.create_index("ix_automation_rules_dataset_key", "automation_rules", ["dataset_key"], schema=schema)
    op.create_index("ix_automation_rules_architecture", "automation_rules", ["architecture"], schema=schema)

    _add_automation_columns("jobs", schema)
    _add_automation_columns("training_results", schema)
    _add_automation_columns("pseudo_markup_results", schema)

    if not op.get_context().as_sql:
        op.bulk_insert(
            sa.table(
                "automation_controls",
                sa.column("key", sa.String),
                sa.column("enabled", sa.Boolean),
                sa.column("updated_at", sa.DateTime(timezone=True)),
                schema=schema,
            ),
            [{"key": "automation", "enabled": False, "updated_at": datetime.now(timezone.utc)}],
        )


def downgrade() -> None:
    schema = _schema()
    _drop_automation_columns("pseudo_markup_results", schema)
    _drop_automation_columns("training_results", schema)
    _drop_automation_columns("jobs", schema)

    op.drop_index("ix_automation_rules_architecture", table_name="automation_rules", schema=schema)
    op.drop_index("ix_automation_rules_dataset_key", table_name="automation_rules", schema=schema)
    op.drop_table("automation_rules", schema=schema)
    op.drop_table("automation_controls", schema=schema)


def _add_automation_columns(table_name: str, schema: str | None) -> None:
    op.add_column(
        table_name,
        sa.Column("source", sa.String(length=32), nullable=False, server_default=sa.text("'manual'")),
        schema=schema,
    )
    op.add_column(
        table_name,
        sa.Column("automation_rule_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema=schema,
    )
    op.add_column(
        table_name,
        sa.Column("dataset_key", sa.String(length=180), nullable=True),
        schema=schema,
    )
    op.add_column(
        table_name,
        sa.Column("dataset_version", sa.String(length=160), nullable=True),
        schema=schema,
    )
    op.create_foreign_key(
        f"fk_{table_name}_automation_rule_id",
        table_name,
        "automation_rules",
        ["automation_rule_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
    )
    op.create_index(f"ix_{table_name}_source", table_name, ["source"], schema=schema)
    op.create_index(f"ix_{table_name}_dataset_key", table_name, ["dataset_key"], schema=schema)
    op.create_index(f"ix_{table_name}_dataset_version", table_name, ["dataset_version"], schema=schema)


def _drop_automation_columns(table_name: str, schema: str | None) -> None:
    op.drop_index(f"ix_{table_name}_dataset_version", table_name=table_name, schema=schema)
    op.drop_index(f"ix_{table_name}_dataset_key", table_name=table_name, schema=schema)
    op.drop_index(f"ix_{table_name}_source", table_name=table_name, schema=schema)
    op.drop_constraint(f"fk_{table_name}_automation_rule_id", table_name, schema=schema, type_="foreignkey")
    op.drop_column(table_name, "dataset_version", schema=schema)
    op.drop_column(table_name, "dataset_key", schema=schema)
    op.drop_column(table_name, "automation_rule_id", schema=schema)
    op.drop_column(table_name, "source", schema=schema)

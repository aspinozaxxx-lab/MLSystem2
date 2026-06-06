"""dataset scoped training templates

Revision ID: 20260606_0003
Revises: 20260604_0002
Create Date: 2026-06-06
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "20260606_0003"
down_revision = "20260604_0002"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column("training_templates", sa.Column("dataset_key", sa.String(length=180), nullable=True), schema=schema)
    op.add_column("training_templates", sa.Column("dataset_name", sa.String(length=240), nullable=True), schema=schema)
    op.add_column(
        "training_templates",
        sa.Column("parent_template_id", postgresql.UUID(as_uuid=True), nullable=True),
        schema=schema,
    )
    op.create_foreign_key(
        "fk_training_templates_parent_template_id",
        "training_templates",
        "training_templates",
        ["parent_template_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
    )
    op.drop_constraint("uq_training_templates_architecture", "training_templates", schema=schema, type_="unique")
    op.create_unique_constraint(
        "uq_training_templates_architecture_dataset",
        "training_templates",
        ["architecture", "dataset_key"],
        schema=schema,
    )
    op.create_index("ix_training_templates_dataset_key", "training_templates", ["dataset_key"], schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.drop_index("ix_training_templates_dataset_key", table_name="training_templates", schema=schema)
    op.drop_constraint(
        "uq_training_templates_architecture_dataset",
        "training_templates",
        schema=schema,
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_training_templates_architecture",
        "training_templates",
        ["architecture"],
        schema=schema,
    )
    op.drop_constraint(
        "fk_training_templates_parent_template_id",
        "training_templates",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_column("training_templates", "parent_template_id", schema=schema)
    op.drop_column("training_templates", "dataset_name", schema=schema)
    op.drop_column("training_templates", "dataset_key", schema=schema)

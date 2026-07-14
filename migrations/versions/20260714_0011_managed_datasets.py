"""Добавить управляемый каталог классов и датасетов.

Revision ID: 20260714_0011
Revises: 20260713_0010
Create Date: 2026-07-14
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260714_0011"
down_revision = "20260713_0010"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.create_table(
        "dataset_classes",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=180), nullable=False),
        sa.Column("name", sa.String(length=240), nullable=False),
        sa.Column("quality_metric", sa.String(length=32), server_default="pixel", nullable=False),
        sa.Column("primary_subclass_id", sa.Uuid(), nullable=True),
        sa.Column("primary_subclass_locked", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_dataset_classes_key"),
        sa.UniqueConstraint("name", name="uq_dataset_classes_name"),
        schema=schema,
    )
    op.create_index("ix_dataset_classes_key", "dataset_classes", ["key"], schema=schema)
    op.create_table(
        "dataset_subclasses",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=180), nullable=False),
        sa.Column("class_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=240), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["class_id"],
            [f"{schema + '.' if schema else ''}dataset_classes.id"],
            name="fk_dataset_subclasses_class_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("class_id", "name", name="uq_dataset_subclasses_class_name"),
        sa.UniqueConstraint("key", name="uq_dataset_subclasses_key"),
        schema=schema,
    )
    op.create_index("ix_dataset_subclasses_key", "dataset_subclasses", ["key"], schema=schema)
    op.create_index("ix_dataset_subclasses_class_id", "dataset_subclasses", ["class_id"], schema=schema)
    op.create_foreign_key(
        "fk_dataset_classes_primary_subclass_id",
        "dataset_classes",
        "dataset_subclasses",
        ["primary_subclass_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )
    op.create_table(
        "datasets",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=180), nullable=False),
        sa.Column("subclass_id", sa.Uuid(), nullable=False),
        sa.Column("source_type", sa.String(length=32), server_default="mlmarkup", nullable=False),
        sa.Column("source_path", sa.String(length=1024), nullable=False),
        sa.Column("image_type", sa.String(length=240), server_default="all", nullable=False),
        sa.Column("config_revision", sa.Integer(), server_default="1", nullable=False),
        sa.Column("legacy_version", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(
            ["subclass_id"],
            [f"{schema + '.' if schema else ''}dataset_subclasses.id"],
            name="fk_datasets_subclass_id",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_datasets_key"),
        sa.UniqueConstraint("subclass_id", name="uq_datasets_subclass_id"),
        sa.UniqueConstraint("source_type", "source_path", name="uq_datasets_source"),
        schema=schema,
    )
    op.create_index("ix_datasets_key", "datasets", ["key"], schema=schema)
    op.create_index("ix_datasets_subclass_id", "datasets", ["subclass_id"], schema=schema)

    op.add_column(
        "training_results",
        sa.Column("quality_metric", sa.String(length=32), server_default="pixel", nullable=False),
        schema=schema,
    )
    op.add_column(
        "test_samples",
        sa.Column("quality_metric", sa.String(length=32), server_default="pixel", nullable=False),
        schema=schema,
    )
    for name in (
        "object_precision",
        "object_recall",
        "object_f1",
    ):
        op.add_column(
            "training_result_test_metrics",
            sa.Column(name, sa.Float(), nullable=True),
            schema=schema,
        )
    for name in (
        "object_true_positive",
        "object_false_positive",
        "object_false_negative",
    ):
        op.add_column(
            "training_result_test_metrics",
            sa.Column(name, sa.BigInteger(), nullable=True),
            schema=schema,
        )


def downgrade() -> None:
    schema = _schema()
    for name in (
        "object_false_negative",
        "object_false_positive",
        "object_true_positive",
        "object_f1",
        "object_recall",
        "object_precision",
    ):
        op.drop_column("training_result_test_metrics", name, schema=schema)
    op.drop_column("test_samples", "quality_metric", schema=schema)
    op.drop_column("training_results", "quality_metric", schema=schema)
    op.drop_table("datasets", schema=schema)
    op.drop_constraint(
        "fk_dataset_classes_primary_subclass_id",
        "dataset_classes",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_table("dataset_subclasses", schema=schema)
    op.drop_table("dataset_classes", schema=schema)

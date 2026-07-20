"""Объединить подклассы с датасетами и добавить тип снимков класса.

Revision ID: 20260720_0012
Revises: 20260714_0011
Create Date: 2026-07-20
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260720_0012"
down_revision = "20260714_0011"
branch_labels = None
depends_on = None


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def _table(name: str) -> str:
    schema = _schema()
    return f'"{schema}"."{name}"' if schema else f'"{name}"'


def upgrade() -> None:
    schema = _schema()
    op.drop_index(
        "ix_test_samples_class_variant",
        table_name="test_samples",
        schema=schema,
    )
    op.drop_column("test_samples", "variant_key", schema=schema)
    op.alter_column(
        "test_samples",
        "variant_name",
        new_column_name="dataset_short_name",
        schema=schema,
    )
    op.create_index(
        "ix_test_samples_class_dataset",
        "test_samples",
        ["class_key", "dataset_key"],
        schema=schema,
    )
    op.drop_column("test_sample_batch_items", "variant_key", schema=schema)
    op.alter_column(
        "test_sample_batch_items",
        "variant_name",
        new_column_name="dataset_short_name",
        schema=schema,
    )
    op.add_column(
        "dataset_classes",
        sa.Column("imagery_type", sa.String(length=32), nullable=True),
        schema=schema,
    )
    op.add_column(
        "dataset_classes",
        sa.Column("primary_dataset_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "dataset_classes",
        sa.Column(
            "primary_dataset_locked",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=True,
        ),
        schema=schema,
    )
    op.add_column(
        "datasets",
        sa.Column("class_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "datasets",
        sa.Column("name", sa.String(length=240), nullable=True),
        schema=schema,
    )

    classes = _table("dataset_classes")
    subclasses = _table("dataset_subclasses")
    datasets = _table("datasets")
    op.execute(
        sa.text(
            f"""
DO $$
DECLARE
    problem_class text;
BEGIN
    SELECT c.name
      INTO problem_class
      FROM {classes} AS c
      JOIN {subclasses} AS s ON s.class_id = c.id
      JOIN {datasets} AS d ON d.subclass_id = s.id
     WHERE d.image_type NOT IN ('all', 'kanopus', 'orto', 'ortho')
     LIMIT 1;
    IF problem_class IS NOT NULL THEN
        RAISE EXCEPTION 'Неподдерживаемый тип снимков в классе %', problem_class;
    END IF;

    SELECT c.name
      INTO problem_class
      FROM {classes} AS c
      JOIN {subclasses} AS s ON s.class_id = c.id
      JOIN {datasets} AS d ON d.subclass_id = s.id
     GROUP BY c.id, c.name
    HAVING COUNT(
        DISTINCT CASE
            WHEN d.image_type IN ('all', 'kanopus') THEN 'kanopus'
            ELSE 'ortho'
        END
    ) > 1
     LIMIT 1;
    IF problem_class IS NOT NULL THEN
        RAISE EXCEPTION 'В классе % смешаны типы снимков', problem_class;
    END IF;
END $$
"""
        )
    )
    op.execute(
        sa.text(
            f"""
UPDATE {datasets} AS d
   SET class_id = s.class_id,
       name = s.name
  FROM {subclasses} AS s
 WHERE s.id = d.subclass_id
"""
        )
    )
    op.execute(
        sa.text(
            f"""
UPDATE {classes} AS c
   SET imagery_type = COALESCE(
           (
               SELECT CASE
                          WHEN d.image_type IN ('all', 'kanopus') THEN 'kanopus'
                          ELSE 'ortho'
                      END
                 FROM {datasets} AS d
                WHERE d.class_id = c.id
                LIMIT 1
           ),
           'kanopus'
       ),
       primary_dataset_id = (
           SELECT d.id
             FROM {datasets} AS d
            WHERE d.subclass_id = c.primary_subclass_id
            LIMIT 1
       ),
       primary_dataset_locked = c.primary_subclass_locked
"""
        )
    )

    op.alter_column("dataset_classes", "imagery_type", nullable=False, schema=schema)
    op.alter_column("dataset_classes", "primary_dataset_locked", nullable=False, schema=schema)
    op.alter_column("datasets", "class_id", nullable=False, schema=schema)
    op.alter_column("datasets", "name", nullable=False, schema=schema)
    op.create_check_constraint(
        "ck_dataset_classes_imagery_type",
        "dataset_classes",
        "imagery_type IN ('kanopus', 'ortho')",
        schema=schema,
    )
    op.create_foreign_key(
        "fk_datasets_class_id",
        "datasets",
        "dataset_classes",
        ["class_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_datasets_class_name",
        "datasets",
        ["class_id", "name"],
        schema=schema,
    )
    op.create_index("ix_datasets_class_id", "datasets", ["class_id"], schema=schema)
    op.create_foreign_key(
        "fk_dataset_classes_primary_dataset_id",
        "dataset_classes",
        "datasets",
        ["primary_dataset_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="SET NULL",
    )

    op.drop_constraint(
        "fk_dataset_classes_primary_subclass_id",
        "dataset_classes",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_datasets_subclass_id",
        "datasets",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_datasets_subclass_id",
        "datasets",
        schema=schema,
        type_="unique",
    )
    op.drop_index("ix_datasets_subclass_id", table_name="datasets", schema=schema)
    op.drop_column("datasets", "subclass_id", schema=schema)
    op.drop_column("datasets", "image_type", schema=schema)
    op.drop_column("dataset_classes", "primary_subclass_id", schema=schema)
    op.drop_column("dataset_classes", "primary_subclass_locked", schema=schema)
    op.drop_table("dataset_subclasses", schema=schema)


def downgrade() -> None:
    schema = _schema()
    op.drop_index(
        "ix_test_samples_class_dataset",
        table_name="test_samples",
        schema=schema,
    )
    op.add_column(
        "test_samples",
        sa.Column("variant_key", sa.String(length=180), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"UPDATE {_table('test_samples')} SET variant_key = dataset_key"
        )
    )
    op.alter_column("test_samples", "variant_key", nullable=False, schema=schema)
    op.alter_column(
        "test_samples",
        "dataset_short_name",
        new_column_name="variant_name",
        schema=schema,
    )
    op.create_index(
        "ix_test_samples_class_variant",
        "test_samples",
        ["class_key", "variant_key"],
        schema=schema,
    )
    op.add_column(
        "test_sample_batch_items",
        sa.Column("variant_key", sa.String(length=180), nullable=True),
        schema=schema,
    )
    op.execute(
        sa.text(
            f"UPDATE {_table('test_sample_batch_items')} SET variant_key = dataset_key"
        )
    )
    op.alter_column(
        "test_sample_batch_items",
        "variant_key",
        nullable=False,
        schema=schema,
    )
    op.alter_column(
        "test_sample_batch_items",
        "dataset_short_name",
        new_column_name="variant_name",
        schema=schema,
    )
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
    op.create_index(
        "ix_dataset_subclasses_class_id",
        "dataset_subclasses",
        ["class_id"],
        schema=schema,
    )
    op.add_column(
        "dataset_classes",
        sa.Column("primary_subclass_id", sa.Uuid(), nullable=True),
        schema=schema,
    )
    op.add_column(
        "dataset_classes",
        sa.Column(
            "primary_subclass_locked",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=True,
        ),
        schema=schema,
    )
    op.add_column("datasets", sa.Column("subclass_id", sa.Uuid(), nullable=True), schema=schema)
    op.add_column(
        "datasets",
        sa.Column("image_type", sa.String(length=240), nullable=True),
        schema=schema,
    )

    classes = _table("dataset_classes")
    subclasses = _table("dataset_subclasses")
    datasets = _table("datasets")
    op.execute(
        sa.text(
            f"""
INSERT INTO {subclasses} (id, key, class_id, name, created_at, updated_at)
SELECT id, key, class_id, name, created_at, updated_at
  FROM {datasets}
"""
        )
    )
    op.execute(
        sa.text(
            f"""
UPDATE {datasets} AS d
   SET subclass_id = d.id,
       image_type = CASE WHEN c.imagery_type = 'ortho' THEN 'orto' ELSE 'kanopus' END
  FROM {classes} AS c
 WHERE c.id = d.class_id
"""
        )
    )
    op.execute(
        sa.text(
            f"""
UPDATE {classes}
   SET primary_subclass_id = primary_dataset_id,
       primary_subclass_locked = primary_dataset_locked
"""
        )
    )
    op.alter_column("dataset_classes", "primary_subclass_locked", nullable=False, schema=schema)
    op.alter_column("datasets", "subclass_id", nullable=False, schema=schema)
    op.alter_column("datasets", "image_type", nullable=False, schema=schema)
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
    op.create_foreign_key(
        "fk_datasets_subclass_id",
        "datasets",
        "dataset_subclasses",
        ["subclass_id"],
        ["id"],
        source_schema=schema,
        referent_schema=schema,
        ondelete="RESTRICT",
    )
    op.create_unique_constraint(
        "uq_datasets_subclass_id",
        "datasets",
        ["subclass_id"],
        schema=schema,
    )
    op.create_index("ix_datasets_subclass_id", "datasets", ["subclass_id"], schema=schema)

    op.drop_constraint(
        "fk_dataset_classes_primary_dataset_id",
        "dataset_classes",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "fk_datasets_class_id",
        "datasets",
        schema=schema,
        type_="foreignkey",
    )
    op.drop_constraint(
        "uq_datasets_class_name",
        "datasets",
        schema=schema,
        type_="unique",
    )
    op.drop_index("ix_datasets_class_id", table_name="datasets", schema=schema)
    op.drop_constraint(
        "ck_dataset_classes_imagery_type",
        "dataset_classes",
        schema=schema,
        type_="check",
    )
    op.drop_column("datasets", "class_id", schema=schema)
    op.drop_column("datasets", "name", schema=schema)
    op.drop_column("dataset_classes", "primary_dataset_id", schema=schema)
    op.drop_column("dataset_classes", "primary_dataset_locked", schema=schema)
    op.drop_column("dataset_classes", "imagery_type", schema=schema)

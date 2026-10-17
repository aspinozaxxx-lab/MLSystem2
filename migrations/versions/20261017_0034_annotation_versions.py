"""Снимки, варианты разметки и выбор вариантов в датасетах."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20261017_0034"
down_revision = "20261015_0033"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    prefix = f"{schema}." if schema else ""
    json_type = sa.JSON().with_variant(JSONB, "postgresql")
    op.add_column("dataset_editor_drafts", sa.Column("annotation_versions", json_type, nullable=True), schema=schema)
    op.create_table(
        "annotation_images",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("imagery_type", sa.String(32), nullable=False),
        sa.Column("relative_path", sa.String(2048), nullable=False),
        sa.Column("annotation_name", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("imagery_type", "relative_path", name="uq_annotation_images_path"),
        schema=schema,
    )
    op.create_table(
        "annotation_versions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("image_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_images.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("class_id", sa.Uuid(), sa.ForeignKey(prefix + "dataset_classes.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("name", sa.String(240), nullable=False),
        sa.Column("geojson", json_type, nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema=schema,
    )
    op.create_index("ix_annotation_versions_class_image", "annotation_versions", ["class_id", "image_id"], schema=schema)
    op.create_table(
        "dataset_annotations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("dataset_id", sa.Uuid(), sa.ForeignKey(prefix + "datasets.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_images.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("annotation_name", sa.String(512), nullable=False),
        sa.Column("materialized_hash", sa.String(64), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.UniqueConstraint("dataset_id", "image_id", name="uq_dataset_annotations_image"),
        sa.UniqueConstraint("dataset_id", "annotation_name", name="uq_dataset_annotations_name"),
        schema=schema,
    )
    op.create_index("ix_dataset_annotations_version", "dataset_annotations", ["version_id"], schema=schema)
    op.create_table(
        "annotation_publications",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("username", sa.String(180), nullable=False),
        sa.Column("payload", json_type, nullable=False),
        sa.Column("commit_sha", sa.String(64), nullable=True),
        sa.Column("completed", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        schema=schema,
    )


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        versions = sa.table("annotation_versions", sa.column("id"), schema=schema)
        if op.get_bind().execute(sa.select(versions.c.id).limit(1)).first():
            raise RuntimeError("Откат удалит версии разметки. Сначала экспортируйте их и освободите каталог версий.")
    for table in ("annotation_publications", "dataset_annotations", "annotation_versions", "annotation_images"):
        op.drop_table(table, schema=schema)
    op.drop_column("dataset_editor_drafts", "annotation_versions", schema=schema)

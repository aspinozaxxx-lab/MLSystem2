"""Редактируемые тестовые области и фиксированные версии исходной разметки."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20261029_0037"
down_revision = "20261019_0036"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    prefix = f"{schema}." if schema else ""
    json_type = sa.JSON().with_variant(JSONB, "postgresql")
    op.create_table(
        "test_sample_sources",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("test_sample_id", sa.Uuid(), sa.ForeignKey(prefix + "test_samples.id", ondelete="CASCADE"), nullable=False),
        sa.Column("image_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_images.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version_info", json_type, nullable=False),
        sa.Column("geojson", json_type, nullable=False),
        sa.Column("revision", sa.String(64), nullable=False),
        sa.Column("raster_revision", sa.String(128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("test_sample_id", "image_id", name="uq_test_sample_source_image"),
        schema=schema,
    )
    op.add_column("test_sample_tiles", sa.Column("source_scene_id", sa.Uuid(), nullable=True), schema=schema)
    op.add_column("test_sample_tiles", sa.Column("source_region_id", sa.String(128), nullable=True), schema=schema)


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        sources = sa.table("test_sample_sources", sa.column("id"), schema=schema)
        if op.get_bind().execute(sa.select(sources.c.id).limit(1)).first():
            raise RuntimeError("Откат удалит исходные тестовые области. Сначала сохраните их резервную копию.")
    op.drop_column("test_sample_tiles", "source_region_id", schema=schema)
    op.drop_column("test_sample_tiles", "source_scene_id", schema=schema)
    op.drop_table("test_sample_sources", schema=schema)

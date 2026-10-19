"""История опубликованных полигонов, без записи личных черновиков."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20261019_0036"
down_revision = "20261018_0035"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    prefix = f"{schema}." if schema else ""
    json_type = sa.JSON().with_variant(JSONB, "postgresql")
    op.create_table(
        "annotation_feature_events",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("publication_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_publications.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("version_id", sa.Uuid(), sa.ForeignKey(prefix + "annotation_versions.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("feature_id", sa.Text(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("username", sa.String(180), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature", json_type, nullable=True),
        sa.Column("crs", json_type, nullable=True),
        sa.UniqueConstraint("publication_id", "version_id", "feature_id", "action", name="uq_annotation_feature_event"),
        schema=schema,
    )
    op.create_index("ix_annotation_feature_history", "annotation_feature_events", ["version_id", "feature_id", "recorded_at"], schema=schema)


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        events = sa.table("annotation_feature_events", sa.column("id"), schema=schema)
        if op.get_bind().execute(sa.select(events.c.id).limit(1)).first():
            raise RuntimeError("Откат удалит историю полигонов. Сначала сохраните её резервную копию.")
    op.drop_table("annotation_feature_events", schema=schema)

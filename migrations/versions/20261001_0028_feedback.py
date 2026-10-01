"""Обращения пользователей и этапы их реализации."""

import os

import sqlalchemy as sa
from alembic import op

revision = "20261001_0028"
down_revision = "20260825_0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("submission_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(24), nullable=False),
        sa.Column("title", sa.String(160), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("author", sa.String(240), nullable=False),
        sa.Column("page_path", sa.String(1000), nullable=False),
        sa.Column("page_title", sa.String(160), nullable=False),
        sa.Column("app_version", sa.String(40)),
        sa.Column("credit_name", sa.String(80)),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("preparation", sa.Text(), nullable=False),
        sa.Column("progress", sa.Text(), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("approved_by", sa.String(240)),
        sa.Column("approval_note", sa.Text()),
        sa.Column("news_slug", sa.String(160)),
        sa.Column("commit_sha", sa.String(40)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("author", "submission_id", name="uq_feedback_submission"),
        sa.CheckConstraint("status IN ('waiting', 'preparing', 'implementing', 'implemented')", name="ck_feedback_status"),
        sa.CheckConstraint("status NOT IN ('implementing', 'implemented') OR approved_at IS NOT NULL", name="ck_feedback_approval"),
        schema=os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None,
    )


def downgrade() -> None:
    op.drop_table("feedback", schema=os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None)

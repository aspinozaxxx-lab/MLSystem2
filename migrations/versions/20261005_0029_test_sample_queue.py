"""Отдельная очередь создания тестовых разметок и настройки датасета."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261005_0029"
down_revision = "20261001_0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.add_column("datasets", sa.Column("test_sample_settings", postgresql.JSONB()), schema=schema)
    op.add_column(
        "test_sample_batches",
        sa.Column("queue_position", sa.BigInteger(), nullable=False, server_default="0"),
        schema=schema,
    )
    op.add_column(
        "test_sample_batches",
        sa.Column("cancel_requested", sa.Boolean(), nullable=False, server_default=sa.false()),
        schema=schema,
    )
    op.add_column(
        "test_sample_batch_items",
        sa.Column("use_optimization", sa.Boolean(), nullable=False, server_default=sa.true()),
        schema=schema,
    )
    op.alter_column(
        "test_sample_batches",
        "active_slot",
        existing_type=sa.Integer(),
        server_default=None,
        schema=schema,
    )
    op.create_index(
        "ix_test_sample_batches_queue_position",
        "test_sample_batches",
        ["queue_position"],
        schema=schema,
    )
    batches = sa.table(
        "test_sample_batches", sa.column("status"), sa.column("active_slot"), schema=schema
    )
    op.execute(batches.update().where(batches.c.status == "queued").values(active_slot=None))


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.alter_column(
        "test_sample_batches",
        "active_slot",
        existing_type=sa.Integer(),
        server_default=sa.text("1"),
        schema=schema,
    )
    op.drop_index(
        "ix_test_sample_batches_queue_position", table_name="test_sample_batches", schema=schema
    )
    op.drop_column("test_sample_batch_items", "use_optimization", schema=schema)
    op.drop_column("test_sample_batches", "cancel_requested", schema=schema)
    op.drop_column("test_sample_batches", "queue_position", schema=schema)
    op.drop_column("datasets", "test_sample_settings", schema=schema)

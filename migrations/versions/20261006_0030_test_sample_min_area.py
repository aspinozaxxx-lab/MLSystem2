"""Минимальная площадь объектов в задании создания тестовой разметки."""

import os

import sqlalchemy as sa
from alembic import op

revision = "20261006_0030"
down_revision = "20261005_0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.add_column(
        "test_sample_batch_items",
        sa.Column("min_object_area_m2", sa.Float(), nullable=False, server_default="0"),
        schema=schema,
    )


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.drop_column("test_sample_batch_items", "min_object_area_m2", schema=schema)

"""Удалить обращения пользователей вместе с их стадиями и подтверждениями."""

import os
from importlib import import_module

from alembic import op

revision = "20261015_0033"
down_revision = "20261012_0032"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.drop_table("feedback", schema=schema)


def downgrade() -> None:
    # Откат возвращает только пустую схему, удалённые обращения не восстанавливаются.
    import_module("migrations.versions.20261001_0028_feedback").upgrade()

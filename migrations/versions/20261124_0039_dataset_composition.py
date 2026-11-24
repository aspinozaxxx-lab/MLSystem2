"""Режим составного датасета и параметры наследования разметки."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20261124_0039"
down_revision = "20261119_0038"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    data_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    op.add_column("datasets", sa.Column("composition", data_type, nullable=True), schema=schema)


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        datasets = sa.table("datasets", sa.column("composition"), schema=schema)
        populated = sa.select(sa.literal(1)).where(sa.cast(datasets.c.composition, sa.Text) != "null").limit(1)
        if op.get_bind().execute(populated).first():
            raise RuntimeError("Для отката требуется сохранить параметры составных датасетов.")
    op.drop_column("datasets", "composition", schema=schema)

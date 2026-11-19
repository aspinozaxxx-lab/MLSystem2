"""Одна личная заметка и её текущие вложения, без истории."""

import os

import sqlalchemy as sa
from alembic import op

revision = "20261119_0038"
down_revision = "20261029_0037"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.create_table(
        "personal_note",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_personal_note_singleton"),
        schema=schema,
    )
    op.create_table(
        "personal_note_attachments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("original_name", sa.String(255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        schema=schema,
    )


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    if not op.get_context().as_sql:
        for name in ("personal_note", "personal_note_attachments"):
            table = sa.table(name, sa.column("id"), schema=schema)
            if op.get_bind().execute(sa.select(table.c.id).limit(1)).first():
                raise RuntimeError("Откат удалит личную заметку. Сначала удалите её данные явно.")
    op.drop_table("personal_note_attachments", schema=schema)
    op.drop_table("personal_note", schema=schema)

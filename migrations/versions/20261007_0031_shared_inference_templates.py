"""Общие именованные шаблоны инференса и привязки классов."""

import os

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

revision = "20261007_0031"
down_revision = "20261006_0030"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.alter_column("inference_templates", "architecture", nullable=True, schema=schema)
    op.add_column("inference_templates", sa.Column("description", sa.Text()), schema=schema)
    op.add_column("inference_templates", sa.Column("archived_at", sa.DateTime(timezone=True)), schema=schema)
    op.add_column("dataset_classes", sa.Column("inference_template_id", postgresql.UUID(as_uuid=True)), schema=schema)
    op.create_foreign_key("fk_dataset_classes_inference_template", "dataset_classes", "inference_templates",
                          ["inference_template_id"], ["id"], source_schema=schema, referent_schema=schema)
    op.create_index("ix_dataset_classes_inference_template_id", "dataset_classes", ["inference_template_id"], schema=schema)
    table = sa.table("inference_templates", sa.column("archived_at"), schema=schema)
    op.execute(table.update().values(archived_at=sa.func.now()))
    op.create_check_constraint(
        "ck_inference_templates_scope", "inference_templates",
        "archived_at IS NOT NULL OR (architecture IS NULL AND dataset_key IS NULL "
        "AND dataset_name IS NULL AND parent_template_id IS NULL)", schema=schema,
    )
    if not op.get_context().as_sql:
        from mlsystem2.training_ui_api._database import configure_schema
        from mlsystem2.training_ui_api._inference_templates import migrate_inference_templates

        configure_schema(schema or "")
        with Session(bind=op.get_bind()) as session:
            migrate_inference_templates(session)
            session.flush()


def downgrade() -> None:
    # Обратный перенос не может восстановить новые классовые настройки по архитектурам.
    raise RuntimeError("Обратный перенос шаблонов инференса требует восстановления резервной копии.")

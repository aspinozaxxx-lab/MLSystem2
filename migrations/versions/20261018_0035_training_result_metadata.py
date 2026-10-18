"""Комментарии, постоянные номера в датасете и скрытый архив обучений."""

import os

import sqlalchemy as sa
from alembic import op

revision = "20261018_0035"
down_revision = "20261017_0034"
branch_labels = None
depends_on = None


def upgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    op.add_column("training_results", sa.Column("dataset_number", sa.Integer(), nullable=True), schema=schema)
    op.add_column("training_results", sa.Column("comment", sa.Text(), nullable=False, server_default=""), schema=schema)
    op.add_column("training_results", sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True), schema=schema)
    op.add_column("training_results", sa.Column("archived_by", sa.String(180), nullable=True), schema=schema)
    op.create_table("training_result_counters",
        sa.Column("dataset_key", sa.String(180), primary_key=True),
        sa.Column("last_number", sa.Integer(), nullable=False), schema=schema)
    results = sa.table("training_results", sa.column("id"), sa.column("class_key"),
        sa.column("created_at"), sa.column("status"), sa.column("dataset_number"), schema=schema)
    ranked = sa.select(results.c.id, sa.func.row_number().over(
        partition_by=results.c.class_key, order_by=(results.c.created_at, results.c.id),
    ).label("number")).where(results.c.status != "cancelled").cte("ranked")
    op.execute(results.update().where(results.c.status != "cancelled").values(
        dataset_number=sa.select(ranked.c.number).where(ranked.c.id == results.c.id).scalar_subquery(),
    ))
    counters = sa.table("training_result_counters", sa.column("dataset_key"), sa.column("last_number"), schema=schema)
    op.execute(counters.insert().from_select(["dataset_key", "last_number"],
        sa.select(results.c.class_key, sa.func.max(results.c.dataset_number))
        .where(results.c.dataset_number.is_not(None)).group_by(results.c.class_key)))
    op.create_index("uq_training_results_dataset_number", "training_results", ["class_key", "dataset_number"], unique=True, schema=schema)


def downgrade() -> None:
    schema = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui") or None
    results = sa.table("training_results", sa.column("comment"), sa.column("archived_at"), schema=schema)
    if not op.get_context().as_sql and op.get_bind().scalar(sa.select(sa.func.count()).select_from(results).where(
        (results.c.comment != "") | results.c.archived_at.is_not(None),
    )):
        raise RuntimeError("Откат удалит комментарии или сведения об архиве. Сначала сохраните их отдельно.")
    op.drop_index("uq_training_results_dataset_number", table_name="training_results", schema=schema)
    op.drop_table("training_result_counters", schema=schema)
    for name in ("archived_by", "archived_at", "comment", "dataset_number"):
        op.drop_column("training_results", name, schema=schema)

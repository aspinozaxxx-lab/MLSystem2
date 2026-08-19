"""Сохранить историческое имя модели датасета.

Revision ID: 20260819_0023
Revises: 20260819_0022
Create Date: 2026-08-19
"""

from __future__ import annotations

import os

import sqlalchemy as sa
from alembic import op


revision = "20260819_0023"
down_revision = "20260819_0022"
branch_labels = None
depends_on = None


_MODEL_NAME_STEMS = {
    ("Абразия", "main"): "abrasion",
    ("Ветровая эрозия", "main"): "wind_erosion",
    ("Водная эрозия", "main"): "water_erosion",
    ("Вырубки", "main"): "deforestation",
    ("Вырубки", "strict"): "deforestationStrict",
    ("Вырубки", "test"): "deforestation",
    ("Гари", "main"): "burnt_forests",
    ("Границы леса", "main"): "forest",
    ("ЗУ500", "main"): "zu500",
    ("Заболачивание", "main"): "swampings",
    ("Засоления", "main"): "salty",
    ("Захламнения", "main"): "landfills",
    ("Захламнения", "test"): "landfills",
    ("Карьеры", "main"): "careers",
    ("ОКС500", "main"): "oks500",
    ("Обвально-оползневые и осыпные", "main"): "landslides",
    ("Озера", "main"): "lakes",
    ("Опустынивание", "main"): "desertification",
    ("Опустынивание и ветровая эрозия", "main"): (
        "desertification_wind_erosion"
    ),
    ("Пашни", "main"): "areas_of_used_arable_land",
    ("Переувлажнения", "main"): "floodings",
    ("Переувлажнения", "test"): "floodings",
    ("Переувлажнения и заболачивания", "main"): "floodings_swampings",
    ("Переувлажнения и заболачивания", "test"): "floodings_swampings",
    ("Разрушки", "main"): "damaged_oks",
    ("Разрушки", "test"): "damaged_oks",
    ("Реки", "main"): "rivers",
    ("Реки", "test"): "rivers",
}


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


def upgrade() -> None:
    schema = _schema()
    op.add_column(
        "datasets",
        sa.Column("model_name_stem", sa.String(length=160), nullable=True),
        schema=schema,
    )
    datasets = sa.table(
        "datasets",
        sa.column("class_id", sa.Uuid()),
        sa.column("name", sa.String()),
        sa.column("model_name_stem", sa.String()),
        schema=schema,
    )
    classes = sa.table(
        "dataset_classes",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String()),
        schema=schema,
    )
    connection = op.get_bind()
    for (class_name, dataset_name), stem in _MODEL_NAME_STEMS.items():
        class_ids = sa.select(classes.c.id).where(classes.c.name == class_name)
        connection.execute(
            sa.update(datasets)
            .where(
                datasets.c.class_id.in_(class_ids),
                sa.or_(
                    datasets.c.name == dataset_name,
                    datasets.c.name.like(f"{dataset_name} [legacy %"),
                ),
                datasets.c.model_name_stem.is_(None),
            )
            .values(model_name_stem=stem)
        )


def downgrade() -> None:
    op.drop_column("datasets", "model_name_stem", schema=_schema())

"""Вспомогательные функции footprints сцен."""

from __future__ import annotations

from .contracts import SceneFootprint, SceneRecord


def calculate_scene_footprint(record: SceneRecord) -> SceneFootprint:
    raise NotImplementedError(
        "Реальное извлечение GIS footprints намеренно оставлено для шага миграции из старого проекта."
    )


def build_placeholder_footprints(records: list[SceneRecord]) -> list[SceneFootprint]:
    return [
        SceneFootprint(
            scene_id=record.scene_id,
            image_uri=record.image_uri,
            geometry_wkt="POLYGON EMPTY",
        )
        for record in records
    ]

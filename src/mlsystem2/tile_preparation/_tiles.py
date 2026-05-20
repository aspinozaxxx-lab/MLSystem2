"""Заглушки функций извлечения тайлов."""

from __future__ import annotations

from mlsystem2.dataset_preparing.contracts import SceneRecord


def extract_tile(scene: SceneRecord, x: int, y: int) -> object:
    raise NotImplementedError(
        "Реальное извлечение тайлов с учетом соседних снимков намеренно оставлено для шага миграции из старого проекта."
    )

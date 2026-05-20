"""Заглушка ввода-вывода чекпойнтов."""

from __future__ import annotations

from .contracts import CheckpointArtifact, LoadCheckpointRequest, LoadedCheckpoint, SaveCheckpointRequest


def load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint:
    raise NotImplementedError(
        "Реальная загрузка чекпойнтов намеренно оставлена для шага миграции моделей."
    )


def save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact:
    raise NotImplementedError(
        "Реальное сохранение чекпойнтов намеренно оставлено для шага миграции моделей."
    )

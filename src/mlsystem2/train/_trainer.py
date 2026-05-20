"""Заглушка цикла обучения."""

from __future__ import annotations

from .contracts import TrainProgressSink, TrainRequest, TrainResult


def train_model(
    request: TrainRequest,
    progress_sink: TrainProgressSink | None = None,
) -> TrainResult:
    raise NotImplementedError(
        "Реальный цикл обучения на PyTorch намеренно оставлен для шага миграции из старого проекта."
    )

"""Заглушка цикла обучения."""

from __future__ import annotations

from .contracts import TrainProgressSink, TrainRequest, TrainResult


def train_model(
    request: TrainRequest,
    progress_sink: TrainProgressSink | None = None,
) -> TrainResult:
    del request, progress_sink
    raise NotImplementedError(
        "Реальный цикл обучения на готовых train_loader и val_loader еще не реализован."
    )

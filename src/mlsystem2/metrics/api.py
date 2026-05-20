"""Публичный фасад метрик."""

from __future__ import annotations

from ._history import summarize_epoch_metrics as _summarize_epoch_metrics
from ._pixel import compute_pixel_f1 as _compute_pixel_f1
from .contracts import EpochMetrics, MetricsSummary, PixelF1Request, PixelF1Result


def compute_pixel_f1(request: PixelF1Request) -> PixelF1Result:
    return _compute_pixel_f1(request)


def summarize_epoch_metrics(history: list[EpochMetrics]) -> MetricsSummary:
    return _summarize_epoch_metrics(history)


__all__ = ["compute_pixel_f1", "summarize_epoch_metrics"]

"""Вспомогательные функции истории метрик."""

from __future__ import annotations

from .contracts import EpochMetrics, MetricsSummary


def summarize_epoch_metrics(history: list[EpochMetrics]) -> MetricsSummary:
    if not history:
        return MetricsSummary(
            epochs_total=0,
            best_f1_pixel=None,
            final_f1_pixel=None,
            total_epoch_time_sec=0.0,
        )
    return MetricsSummary(
        epochs_total=len(history),
        best_f1_pixel=max(item.f1_pixel for item in history),
        final_f1_pixel=history[-1].f1_pixel,
        total_epoch_time_sec=sum(item.epoch_time_sec for item in history),
    )

"""Заглушки источников с ограниченной предварительной подготовкой."""

from __future__ import annotations

from collections.abc import Iterator

from .contracts import (
    TileBatch,
    TilePreparationReport,
    TileSourceBundle,
    TileSourceRequest,
)


class EmptyTileBatchSource:
    def __init__(self, report: TilePreparationReport) -> None:
        self._report = report

    def __iter__(self) -> Iterator[TileBatch]:
        return iter(())

    def close(self) -> None:
        return None

    def profile_snapshot(self) -> TilePreparationReport:
        return self._report


def build_tile_sources(request: TileSourceRequest) -> TileSourceBundle:
    report = TilePreparationReport(
        train_batches_prepared=0,
        val_batches_prepared=0,
        queue_capacity=request.config.prefetch_batches,
        worker_count=request.config.prefetch_workers,
        warnings=[
            "Источники тайлов являются пустыми итераторами-заглушками до миграции предварительной подготовки с учетом соседних снимков."
        ],
    )
    source = EmptyTileBatchSource(report)
    return TileSourceBundle(train=source, val=source, report=report)

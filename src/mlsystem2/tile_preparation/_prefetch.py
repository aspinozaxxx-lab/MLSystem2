"""Заглушки источников с ограниченной предварительной подготовкой."""

from __future__ import annotations

from collections.abc import Iterator

from .contracts import (
    TileBatch,
    TilePreparationError,
    TilePreparationReport,
    TileSourceBundle,
    TileSourceRequest,
)
from ._tiles import validate_vrt_xml


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
    try:
        validate_vrt_xml(request.dataset.train_vrt_xml)
        validate_vrt_xml(request.dataset.val_vrt_xml)
    except Exception as exc:  # noqa: BLE001
        raise TilePreparationError("Не удалось открыть VRT XML для подготовки тайлов") from exc

    report = TilePreparationReport(
        train_batches_prepared=0,
        val_batches_prepared=0,
        queue_capacity=request.prefetch_batches,
        worker_count=request.prefetch_workers,
        warnings=[
            "Реальная нарезка батчей из VRT еще не реализована."
        ],
    )
    source = EmptyTileBatchSource(report)
    return TileSourceBundle(train=source, val=source, report=report)

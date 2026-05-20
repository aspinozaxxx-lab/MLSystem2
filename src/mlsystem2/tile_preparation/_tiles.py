"""Открытие VRT-источников для нарезки тайлов."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rasterio.io import DatasetReader, MemoryFile


@contextmanager
def open_vrt_xml(vrt_xml: str) -> Iterator[DatasetReader]:
    with MemoryFile(vrt_xml.encode("utf-8")) as memory_file:
        with memory_file.open() as dataset:
            yield dataset


def validate_vrt_xml(vrt_xml: str) -> None:
    with open_vrt_xml(vrt_xml) as dataset:
        dataset.read(1, window=((0, min(dataset.height, 1)), (0, min(dataset.width, 1))))

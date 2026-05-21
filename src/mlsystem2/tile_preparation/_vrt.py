"""Открытие VRT XML из памяти."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rasterio.io import DatasetReader, MemoryFile


@contextmanager
def open_vrt_xml(vrt_xml: str) -> Iterator[DatasetReader]:
    with MemoryFile(vrt_xml.encode("utf-8")) as memory_file:
        with memory_file.open() as dataset:
            yield dataset


def open_vrt_reader(vrt_xml: str) -> tuple[MemoryFile, DatasetReader]:
    memory_file = MemoryFile(vrt_xml.encode("utf-8"))
    try:
        dataset = memory_file.open()
    except Exception:
        memory_file.close()
        raise
    return memory_file, dataset

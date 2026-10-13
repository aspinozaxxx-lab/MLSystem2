"""Точная статистика объектных целей без передачи растров между процессами."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import copy
import logging
import math
import multiprocessing
from multiprocessing.util import Finalize
import time

import numpy as np
from rasterio.windows import Window

from ._object_targets import boundary_targets_v2, rasterize_objects
from .contracts import HARD_NEGATIVE_LABEL


LOGGER = logging.getLogger(__name__)
_WORKER_DATASET = None


def _initialize_worker(dataset):
    global _WORKER_DATASET
    _WORKER_DATASET = dataset
    # GDAL надо закрыть до разрушения модулей Python: поздний __del__ уже небезопасен.
    Finalize(None, dataset.close, exitpriority=10)


def _window_statistics(dataset, index):
    item = dataset._windows[index]
    source = dataset._open_dataset(item.scene_index)
    size = dataset._tile_size
    window = Window(item.window.x, item.window.y, size, size)
    invalid = (
        dataset._region_invalid(item.scene_index, source, window)
        if dataset._scenes[item.scene_index].region_geometry is not None
        else dataset._read_invalid_data_pixels(source, window)
    )
    mask = dataset._read_supervision_mask(item.scene_index, source, window, invalid)[0]
    instances, _ = rasterize_objects(dataset._positive_index(item.scene_index), source,
        window, size, invalid | (mask < 0), collect_overlaps=False)
    boundary, usable = boundary_targets_v2(instances, ~invalid)
    labels = (instances > 0).astype(np.int32)
    labels[boundary > 0] = 2
    counts = np.bincount(labels[usable], minlength=3)
    ratio = float(np.count_nonzero((mask > 0) & ~invalid) / max(1, np.count_nonzero(~invalid)))
    return ratio, bool(np.any(mask == HARD_NEGATIVE_LABEL)), counts


def _statistics_block(bounds):
    start, end = bounds
    ratios, negatives = [], []
    counts = np.zeros(3, dtype=np.int64)
    for index in range(start, end):
        ratio, negative, current = _window_statistics(_WORKER_DATASET, index)
        ratios.append(ratio)
        negatives.append(negative)
        counts += current
    return start, ratios, negatives, counts


def object_statistics(dataset, requested_workers):
    """Вернуть прежние частоты/веса по всем train-окнам в исходном порядке."""
    length = len(dataset._windows)
    workers = min(max(1, requested_workers), max(1, length))
    ratios = np.empty(length, dtype=np.float64)
    negatives = np.empty(length, dtype=bool)
    counts = np.zeros(3, dtype=np.int64)
    started = last_report = time.monotonic()
    completed = 0
    LOGGER.info("object f1 v2: начало точной статистики train, окон=%s, процессов=%s", length, workers)

    def progress():
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 15 or completed == length:
            LOGGER.info("object f1 v2: статистика train %s/%s окон (%.1f%%), %.1f с",
                        completed, length, 100 * completed / max(1, length), now - started)
            last_report = now

    if workers == 1:
        for index in range(length):
            ratios[index], negatives[index], current = _window_statistics(dataset, index)
            counts += current
            completed += 1
            progress()
    else:
        # В копии нет открытых GDAL-дескрипторов и индексов; аугментации здесь не нужны.
        snapshot = copy.copy(dataset)
        snapshot._notebook_transform = snapshot._zone_transform = None
        chunk = min(16, max(1, math.ceil(length / (workers * 4))))
        blocks = iter((start, min(start + chunk, length)) for start in range(0, length, chunk))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_initialize_worker, initargs=(snapshot,)) as pool:
            pending = deque()
            for bounds in blocks:
                pending.append(pool.submit(_statistics_block, bounds))
                if len(pending) == workers * 2:
                    break
            try:
                while pending:
                    done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.remove(future)
                        start, block_ratios, block_negatives, current = future.result()
                        end = start + len(block_ratios)
                        ratios[start:end] = block_ratios
                        negatives[start:end] = block_negatives
                        counts += current
                        completed += len(block_ratios)
                        bounds = next(blocks, None)
                        if bounds is not None:
                            pending.append(pool.submit(_statistics_block, bounds))
                    progress()
            finally:
                for future in pending:
                    future.cancel()
    LOGGER.info("object f1 v2: статистика train готова за %.1f с, пикселей фон/внутренность/граница=%s",
                time.monotonic() - started, counts.tolist())
    return ratios, negatives, counts

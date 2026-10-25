"""Точная train-статистика целей, автобаланса и нормализации без передачи растров."""

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
from .contracts import HARD_NEGATIVE_LABEL, TilePreparationError


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
    supervision = dataset._read_supervision_mask(item.scene_index, source, window, invalid)
    if dataset._pipeline_variant in {"next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "next_gen4_consistency_v7"}:
        semantic = supervision if dataset.uses_multiclass_masks else supervision[0]
        negatives = (dataset._read_class_hard_negative_masks(item.scene_index, source, window, invalid, supervision)
                     if dataset._has_class_specific_hard_negatives else None)
        valid = ~invalid
        positive = bool(np.any((semantic > 0) & valid))
        usable = valid if negatives is None else valid & ~negatives.any(axis=0)
        counts = np.bincount(np.maximum(semantic[usable], 0).astype(np.int64), minlength=_class_count(dataset))
        hard = bool(np.any(semantic == HARD_NEGATIVE_LABEL))
        ratio = float(np.count_nonzero((semantic > 0) & valid) / max(1, np.count_nonzero(valid)))
        histogram = None
        if dataset._pipeline_variant in {"next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "next_gen4_consistency_v7"} and dataset._collect_band_histogram_enabled:
            image = dataset._read_image_raw(source, window, dataset._scene_nodata[item.scene_index])
            if image.dtype != np.uint8:
                raise TilePreparationError("Глобальная нормализация ожидает исходные каналы uint8")
            histogram = np.stack([np.bincount(channel[valid], minlength=256) for channel in image]).astype(np.uint64)
        return ratio, hard or bool(negatives is not None and negatives.any()), counts, {
            "positive": positive, "valid_pixels": int(valid.sum()), "hard_negative": hard,
            **({"dense_valid_pixels": int(usable.sum()),
                "dense_positive_fraction": float(np.count_nonzero((semantic > 0) & usable) / max(1, usable.sum()))}
               if dataset._pipeline_variant in {"next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "next_gen4_consistency_v7"} else {}),
            "partial_negative": bool(negatives is not None and negatives.any()),
            "scene_id": item.scene_id,
            "group": f"{item.scene_id}:{item.window.x // (4 * size)}:{item.window.y // (4 * size)}",
        }, histogram
    mask = supervision[0]
    instances, _ = rasterize_objects(dataset._positive_index(item.scene_index), source,
        window, size, invalid | (mask < 0), collect_overlaps=False)
    boundary, usable = boundary_targets_v2(instances, ~invalid)
    labels = (instances > 0).astype(np.int32)
    labels[boundary > 0] = 2
    counts = np.bincount(labels[usable], minlength=3)
    ratio = float(np.count_nonzero((mask > 0) & ~invalid) / max(1, np.count_nonzero(~invalid)))
    return ratio, bool(np.any(mask == HARD_NEGATIVE_LABEL)), counts, None, None


def _class_count(dataset):
    if dataset._pipeline_variant == "object_f1_v2":
        return 3
    return 1 + len(dataset._classes or dataset._class_annotations) if dataset.uses_multiclass_masks else 2


def _statistics_block(bounds):
    start, end = bounds
    ratios, negatives, tiles = [], [], []
    counts = np.zeros(_class_count(_WORKER_DATASET), dtype=np.int64)
    histogram = np.zeros((_WORKER_DATASET.channel_count, 256), dtype=np.uint64)
    for index in range(start, end):
        ratio, negative, current, tile, bands = _window_statistics(_WORKER_DATASET, index)
        ratios.append(ratio)
        negatives.append(negative)
        counts += current
        tiles.append(tile)
        if bands is not None:
            histogram += bands
    return start, ratios, negatives, counts, tiles, histogram


def training_statistics(dataset, requested_workers):
    """Вернуть прежние частоты/веса по всем train-окнам в исходном порядке."""
    length = len(dataset._windows)
    workers = min(max(1, requested_workers), max(1, length))
    ratios = np.empty(length, dtype=np.float64)
    negatives = np.empty(length, dtype=bool)
    counts = np.zeros(_class_count(dataset), dtype=np.int64)
    tiles = [None] * length
    histogram = np.zeros((dataset.channel_count, 256), dtype=np.uint64)
    profile = dataset._pipeline_variant.replace("_", "-") if dataset._pipeline_variant in {"next_gen3", "next_gen4", "next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "next_gen4_consistency_v7"} else "object f1 v2"
    started = last_report = time.monotonic()
    completed = 0
    LOGGER.info("%s: начало точной статистики train, окон=%s, процессов=%s", profile, length, workers)

    def progress():
        nonlocal last_report
        now = time.monotonic()
        if now - last_report >= 15 or completed == length:
            LOGGER.info("%s: статистика train %s/%s окон (%.1f%%), %.1f с",
                        profile, completed, length, 100 * completed / max(1, length), now - started)
            last_report = now

    if workers == 1:
        for index in range(length):
            ratios[index], negatives[index], current, tiles[index], bands = _window_statistics(dataset, index)
            counts += current
            if bands is not None:
                histogram += bands
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
                        start, block_ratios, block_negatives, current, block_tiles, bands = future.result()
                        end = start + len(block_ratios)
                        ratios[start:end] = block_ratios
                        negatives[start:end] = block_negatives
                        tiles[start:end] = block_tiles
                        counts += current
                        histogram += bands
                        completed += len(block_ratios)
                        bounds = next(blocks, None)
                        if bounds is not None:
                            pending.append(pool.submit(_statistics_block, bounds))
                    progress()
            finally:
                for future in pending:
                    future.cancel()
    LOGGER.info("%s: статистика train готова за %.1f с, пикселей по классам=%s",
                profile, time.monotonic() - started, counts.tolist())
    if dataset._pipeline_variant in {"next_gen4_global_norm", "next_gen4_augmentation_v3", "next_gen4_augmentation_v4", "next_gen4_consistency_v1", "next_gen4_consistency_v2", "next_gen4_consistency_v3", "next_gen4_consistency_v4", "next_gen4_consistency_v5", "next_gen4_consistency_v6", "next_gen4_consistency_v7"} and dataset._collect_band_histogram_enabled:
        dataset._band_histogram = {"counts": histogram.tolist(), "bins": list(range(256)),
            "source": "retained_train_windows", "valid_pixel_count": int(histogram[0].sum()),
            "window_count": length, "scene_ids": sorted({item.scene_id for item in dataset._windows})}
    return ratios, negatives, counts, tiles

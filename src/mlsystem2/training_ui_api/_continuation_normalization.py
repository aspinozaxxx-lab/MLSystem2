"""Ограниченная по объёму диагностика входных данных; не калибрует сеть и не считает F1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling

from mlsystem2.dataset_preparing.api import resolve_scene_images
from mlsystem2.dataset_preparing.contracts import SceneImageResolutionRequest
from mlsystem2.models.api import compatible_training_pipelines

from ._markup_lock import markup_read_lock, test_sample_storage_lock
from ._training_continuation import (
    CONFIG_ARTIFACT, RESOLVED_CONFIG_ARTIFACT, _current_dataset, _download, _global_pipeline,
    _pipeline_spec, _source, _test_sample_files, _training_test_sample,
)
from .contracts import TrainingContinuationNormalizationAdvice, TrainingUIAPIError


def _report(config, run_id, artifact):
    root = config.scratch_root / "continuation-advice" / hashlib.sha256(run_id.encode()).hexdigest()
    cached = root / Path(artifact).name
    if not cached.is_file():
        cached = _download(config, run_id, artifact, root)
    return yaml.safe_load(cached.read_text(encoding="utf-8"))


def _data_images(session, result, job, config, selection):
    channels = int(job.config.get("train.input_channels") or 4)
    if selection.dataset_source == "test_sample":
        sample = _training_test_sample(session, result, selection.test_sample_id)
        files = _test_sample_files(sample, config, channels)
        return [(image, str(tile.source_scene_id or tile.source_name)) for tile, image, _ in files]
    if selection.dataset_source in {"current", "dataset"}:
        dataset = _current_dataset(session, result, job, config, dataset_key=selection.dataset_key)
        data = dataset.model_dump()
    else:
        path = Path(job.tmp_path) / "run.yml" if job.tmp_path else None
        if path is not None and path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))["dataset"]
        else:
            try:
                data = _report(config, result.mlflow_run_id, RESOLVED_CONFIG_ARTIFACT)["dataset"]
            except TrainingUIAPIError:
                data = _report(config, result.mlflow_run_id, CONFIG_ARTIFACT)["dataset"]
    resolved = resolve_scene_images(SceneImageResolutionRequest(
        images_dir=data["images_dir"], scenes_file=data.get("scenes_file"), annotations_dir=data.get("annotations_dir"),
    ))
    if resolved.missing_scenes or resolved.ambiguous_scenes or not resolved.images:
        raise TrainingUIAPIError("Не все исходные снимки доступны для предварительной оценки.")
    return [(Path(image.image_path), image.scene_id) for image in resolved.images]


def _sample_images(images, channels, previous):
    # Равномерный детерминированный отбор по именам, максимум 24 × 256 × 256 пикселей.
    selected = sorted(images, key=lambda item: hashlib.sha256(str(item[0]).encode()).hexdigest())[:24]
    counts = np.zeros((channels, 256), dtype=np.int64)
    valid_pixels, estimated, affected = 0, 0.0, set()
    low = np.asarray(previous.get("low", [0] * channels), dtype=float)
    high = np.asarray(previous.get("high", [255] * channels), dtype=float)
    for path, source in selected:
        with rasterio.open(path) as raster:
            if raster.count != channels or any(dtype != "uint8" for dtype in raster.dtypes):
                raise TrainingUIAPIError("Оценка нормализации требует совместимые uint8-снимки.")
            height, width = min(256, raster.height), min(256, raster.width)
            pixels = raster.read(out_shape=(channels, height, width), resampling=Resampling.nearest)
            valid = raster.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
            valid &= np.any(pixels != 0, axis=0)
            values = pixels[:, valid]
            count = values.shape[1]
            valid_pixels += count
            estimated += raster.width * raster.height * count / (width * height)
            if not count:
                continue
            clipped = ((values < low[:, None]) | (values > high[:, None])).mean(axis=1)
            if np.max(clipped) >= 0.05:
                affected.add(source)
            for channel in range(channels):
                counts[channel] += np.bincount(values[channel], minlength=256)
    fractions = []
    for channel in range(channels):
        outside = (np.arange(256) < low[channel]) | (np.arange(256) > high[channel])
        fractions.append(float(counts[channel, outside].sum() / max(1, counts[channel].sum())))
    return {"scene_count": len(images), "source_count": len({source for _, source in images}),
            "sampled_scene_count": len(selected), "sampled_valid_pixels": valid_pixels,
            "estimated_valid_pixels": int(estimated * len(images) / max(1, len(selected))),
            "clipped_fractions": fractions}, len(affected)


def _recommend(statistics, previous, affected):
    original = int(previous.get("valid_pixel_count") or 0)
    volume = statistics["estimated_valid_pixels"]
    small = (statistics["source_count"] < 6 or statistics["sampled_valid_pixels"] < 10000
             or (original > 0 and volume < original * 0.25))
    if small:
        return "preserve", "Набор невелик или содержит мало разных исходников. Его диапазоны могут быть случайными; сохраняйте нормализацию исходной сети."
    if max(statistics["clipped_fractions"], default=0) < 0.05:
        return "preserve", "Прежние границы покрывают не менее 95% проверенных значений каждого канала. Явных оснований менять привычный сети масштаб нет."
    if affected < max(3, statistics["sampled_scene_count"] / 2):
        return "preserve", "Сильное обрезание значений видно лишь на части исходников. Пересчёт по всему набору может ухудшить остальные; сначала проверьте эти снимки."
    return "recompute", "Набор достаточно велик и разнообразен, а минимум в одном канале более 5% значений выходит за прежние границы на многих снимках. Имеет смысл сравнить отдельный этап с пересчётом, проверив качество и на прежних территориях."


def normalization_advice(session, result_id, selection, config):
    result, job = _source(session, result_id)
    variant = selection.pipeline_variant or job.config.get("train.pipeline_variant") or "legacy"
    if variant not in compatible_training_pipelines(_pipeline_spec(result, job)):
        raise TrainingUIAPIError("Конвейер несовместим с выбранной сетью.")
    if not _global_pipeline(variant):
        return TrainingContinuationNormalizationAdvice(status="not_applicable", reason="У этого конвейера нет общих min/max, которые нужно пересчитывать по датасету.")
    try:
        source_global = _global_pipeline(str(job.config.get("train.pipeline_variant") or "legacy"))
        previous = _report(config, result.mlflow_run_id, "reports/preprocessing.json") if source_global else {}
        if source_global and (previous.get("mode") != "global_minmax" or len(previous.get("low", [])) != int(job.config.get("train.input_channels") or 4)):
            raise TrainingUIAPIError("Исходные границы нормализации недоступны для сравнения.")
        if selection.dataset_source == "test_sample":
            with test_sample_storage_lock(config, selection.test_sample_id, wait=False):
                images = _data_images(session, result, job, config, selection)
                statistics, affected = _sample_images(images, int(job.config.get("train.input_channels") or 4), previous)
        else:
            with markup_read_lock(config):
                images = _data_images(session, result, job, config, selection)
                statistics, affected = _sample_images(images, int(job.config.get("train.input_channels") or 4), previous)
        if not statistics["sampled_valid_pixels"]:
            raise TrainingUIAPIError("В проверенных снимках нет полезных пикселей.")
        if source_global:
            recommendation, reason = _recommend(statistics, previous, affected)
        else:
            recommendation, reason = "recompute", "Выбран переход от оконной к глобальной нормализации. Прежних общих границ нет: новый профиль требует калибровки на train. Для небольшого набора безопаснее также сравнить исходный конвейер."
        return TrainingContinuationNormalizationAdvice(status="available", recommendation=recommendation, reason=reason,
            **statistics, original_valid_pixels=previous.get("valid_pixel_count"), warnings=[
                "Это оценка входных данных, а не прогноз F1. Разные исходники могут относиться к одной территории.",
                "Просмотренная статистика не используется для обучения. При явном пересчёте границы определяются только по train нового этапа.",
            ])
    except (OSError, ValueError, KeyError, TrainingUIAPIError) as exc:
        detail = str(exc) if isinstance(exc, TrainingUIAPIError) else "не удалось прочитать сохранённые данные"
        return TrainingContinuationNormalizationAdvice(status="unavailable", recommendation="preserve" if source_global else "recompute",
            reason=f"Оценка пока недоступна: {detail}. " + ("По умолчанию сохраняйте исходную нормализацию." if source_global else "Новый глобальный профиль потребует калибровки только по train."))

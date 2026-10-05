"""Авторизованный просмотр сохранённого тестового расчёта конкретной сети."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._models import JobRow, TrainingResultRow, TrainingResultTestMetricRow
from ._pseudo_viewer import _raster_metadata
from ._raster_http import raster_revision
from ._test_f1_artifacts import test_f1_layers
from ._test_samples import queue_training_result_test_f1, training_result_test_f1_info
from .contracts import JobSource, TestF1SceneInfo, TestF1ScoreInfo, TestF1ViewInfo, TrainingUIAPIError


def store_test_f1_view(
    job: JobRow, config: TrainingUIAPIConfig, previous: dict[str, str],
    *, selected_slugs: list[str] | None,
) -> dict[str, str]:
    """Опубликовать маски до очистки scratch; сохранить другие классы при частичной оценке."""
    references = {key: value for key, value in previous.items() if selected_slugs and key not in selected_slugs}
    source = Path(job.tmp_path) / "scratch" / "test_f1_view" if job.tmp_path else None
    if source is None or not (source / "manifest.json").is_file():
        return references
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    storage_root = (config.stored_files_root / "test-f1").resolve()
    target = (storage_root / str(job.id)).resolve()
    temporary = target.with_name(f".{job.id}.tmp")
    if not target.is_relative_to(storage_root) or not temporary.resolve().is_relative_to(storage_root):
        raise RuntimeError("Каталог визуализации находится вне хранилища.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    if target.exists():
        shutil.rmtree(target)
    temporary.replace(target)
    for scene in manifest["scenes"]:
        references[str(scene.get("target_class_slug") or "foreground")] = str(job.id)
    return references


def _result(session: Session, result_id: uuid.UUID) -> TrainingResultRow:
    result = session.get(TrainingResultRow, result_id)
    if result is None or result.status != "ok":
        raise TrainingUIAPIError("Успешная сеть для просмотра тестового F1 не найдена.")
    return result


def _saved_scenes(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID):
    metric = session.get(TrainingResultTestMetricRow, result_id)
    references = dict((metric.metrics or {}).get("viewer_jobs") or {}) if metric else {}
    scenes: list[tuple[Path, dict[str, Any]]] = []
    warnings: list[str] = []
    for job_id in dict.fromkeys(references.values()):
        try:
            canonical_id = str(uuid.UUID(str(job_id)))
            root = (config.stored_files_root / "test-f1" / canonical_id).resolve()
            if not root.is_relative_to((config.stored_files_root / "test-f1").resolve()):
                raise ValueError
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("version") != 1 or manifest.get("training_result_id") != str(result_id):
                raise ValueError
            for scene in manifest["scenes"]:
                key = str(scene.get("target_class_slug") or "foreground")
                if references.get(key) != job_id:
                    continue
                scene_id = str(scene["id"])
                if not scene_id.startswith("tile-") or not scene_id[5:].isdigit():
                    raise ValueError
                path = (root / f"{scene_id}.npz").resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError
                scenes.append((root, {**scene, "public_id": f"{canonical_id}_{scene_id}"}))
        except (OSError, ValueError, KeyError, TypeError):
            warnings.append("Часть сохранённой визуализации недоступна. Подготовьте её заново.")
    if metric and (metric.metrics or {}).get("aggregation") == "macro":
        scope = list((metric.metrics or {}).get("test_samples") or [])
        required = {str(item["class_slug"]) for item in scope}
        if not required:
            required = set((metric.metrics or {}).get("pixel", {}).get("per_class", {}))
        if not required.issubset(references):
            warnings.append("Часть сохранённой визуализации отсутствует для других классов. Подготовьте её заново.")
        for item in scope:
            saved = [scene for _, scene in scenes if scene.get("target_class_slug") == item["class_slug"]]
            if sorted(scene["source_tile_index"] for scene in saved) != sorted(item["tile_indices"]):
                warnings.append("Состав сохранённой визуализации неполный. Подготовьте её заново.")
    return scenes, list(dict.fromkeys(warnings))


def _image_path(config: TrainingUIAPIConfig, scene: dict[str, Any]) -> Path:
    path = Path(scene["image_path"]).resolve()
    if not path.is_relative_to((config.stored_files_root / "test-samples").resolve()):
        raise TrainingUIAPIError("Снимок находится вне каталога тестовых разметок.")
    if not path.is_file():
        raise TrainingUIAPIError("Исходный тестовый снимок удалён. Сохранённые слои доступны без подложки.")
    if raster_revision(path) != scene["image_revision"]:
        raise HTTPException(412, "Тестовый TIFF изменился после расчёта F1. Подготовьте визуализацию заново.")
    return path


def _score(report: dict[str, Any], prefix: str = "") -> TestF1ScoreInfo:
    tp, fp, fn = (int(report.get(f"{prefix}{key}") or 0) for key in (
        "true_positive", "false_positive", "false_negative",
    ))
    return TestF1ScoreInfo(
        true_positive=tp, false_positive=fp, false_negative=fn,
        precision=tp / (tp + fp) if tp + fp else 0,
        recall=tp / (tp + fn) if tp + fn else 0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
    )


def test_f1_view(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID) -> TestF1ViewInfo:
    result = _result(session, result_id)
    metric = training_result_test_f1_info(session, result, config)
    saved, warnings = _saved_scenes(session, config, result_id)
    complete = not warnings
    scenes = []
    base = f"/api/v1/results/training/{result_id}/test-f1"
    for _, scene in saved:
        available = True
        metadata = (scene["bounds"], False, False, None)
        try:
            path = _image_path(config, scene)
            stat = path.stat()
            metadata = _raster_metadata(str(path), stat.st_mtime_ns, stat.st_size)
        except (OSError, ValueError, TrainingUIAPIError, HTTPException) as exc:
            available = False
            warnings.append(f"{scene['sample_name']}, тайл {scene['source_tile_index']}: "
                            + (str(exc.detail) if isinstance(exc, HTTPException) else str(exc)))
        scene_id = scene["public_id"]
        bounds, has_alpha, has_nir, nodata = metadata
        scenes.append(TestF1SceneInfo(
            id=scene_id, name=f"Тайл {scene['source_tile_index']:03d}",
            sample_name=scene["sample_name"], sample_revision=scene["sample_revision"],
            raster_url=f"{base}/raster/{scene_id}?v={scene['image_revision']}",
            footprint_url=f"{base}/footprint/{scene_id}?v={scene['image_revision']}",
            layers_url=f"{base}/layers/{scene_id}", raster_available=available,
            bounds=bounds, has_alpha=has_alpha, has_nir=has_nir, nodata=nodata,
            target_class_id=scene.get("target_class_id"), class_schema=scene["class_schema"],
            pixel=_score(scene["report"]), objects=_score(scene["report"], "object_"),
            metrics=scene["report"].get("metrics") or {},
        ))
    status = "ready" if scenes and complete else "missing"
    if metric is None or metric.status == "unavailable":
        status = "ready" if scenes and complete else "missing" if scenes else "unavailable"
    elif metric.status in {"queued", "running", "error"}:
        status = metric.status
    return TestF1ViewInfo(
        training_result_id=result.id, model_name=result.model_name,
        training_dataset_name=result.class_display_name, metric=metric, status=status,
        scenes=scenes, warnings=list(dict.fromkeys(warnings)),
    )


def prepare_test_f1_view(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID) -> TestF1ViewInfo:
    # Блокировка сети делает два одновременных нажатия одним заданием.
    result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.id == result_id).with_for_update())
    if result is None or result.status != "ok":
        raise TrainingUIAPIError("Успешная сеть для просмотра тестового F1 не найдена.")
    view = test_f1_view(session, config, result_id)
    if view.status not in {"ready", "queued", "running"}:
        queue_training_result_test_f1(session, result, config, source=JobSource.MANUAL, force=True)
        session.commit()
    return test_f1_view(session, config, result_id)


def _scene(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID, scene_id: str):
    _result(session, result_id)
    scenes, _ = _saved_scenes(session, config, result_id)
    for root, scene in scenes:
        if scene["public_id"] == scene_id:
            return root, scene
    raise TrainingUIAPIError("Снимок не входит в сохранённый расчёт этой сети.")


def test_f1_raster(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID, scene_id: str) -> Path:
    _, scene = _scene(session, config, result_id, scene_id)
    return _image_path(config, scene)


def test_f1_scene_layers(
    session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID, scene_id: str, class_id: int | None,
) -> dict[str, Any]:
    root, scene = _scene(session, config, result_id, scene_id)
    target = scene.get("target_class_id")
    allowed = {int(item["id"]) for item in scene["class_schema"]}
    if target is not None:
        if class_id is not None and class_id != target:
            raise TrainingUIAPIError("Этот снимок оценивает только класс своей тестовой выборки.")
        class_id = target
    elif class_id is not None and class_id not in allowed:
        raise TrainingUIAPIError("Класс отсутствует в сохранённом расчёте.")
    path = root / f"{scene['id']}.npz"
    stat = path.stat()
    return test_f1_layers(str(path), stat.st_mtime_ns, stat.st_size, tuple(scene["transform"]), scene["crs"], class_id)

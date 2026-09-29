"""Просмотр готовой псевдоразметки на исходных снимках без создания растровой мозаики."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from functools import lru_cache
from pathlib import Path

import rasterio
from rasterio.enums import ColorInterp
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import mapping
from sqlalchemy.orm import Session

from mlsystem2.dataset_preparing.api import resolve_scene_images
from mlsystem2.dataset_preparing.contracts import SceneImageResolutionRequest

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import dataset_class_row, find_managed_dataset
from ._datasets import RASTER_SUFFIXES, imagery_images_dir
from ._models import JobRow, PseudoMarkupResultRow
from ._raster_http import raster_revision
from ._raster_valid_data import valid_data_footprint
from .contracts import PseudoMarkupSceneInfo, PseudoMarkupViewInfo, TrainingUIAPIError


def pseudo_markup_view(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID) -> PseudoMarkupViewInfo:
    result = _ready_result(session, result_id)
    paths, cached_warnings = _result_images(session, config, result)
    warnings = list(cached_warnings)
    scenes = []
    for path in paths:
        try:
            stat = path.stat()
            bounds, has_alpha, has_nir, nodata = _raster_metadata(str(path), stat.st_mtime_ns, stat.st_size)
            scene_id = _scene_id(path)
            scenes.append(PseudoMarkupSceneInfo(
                id=scene_id, name=path.relative_to(config.images_root.resolve()).as_posix(),
                raster_url=f"/api/v1/results/pseudo-markup/{result.id}/raster/{scene_id}?v={raster_revision(path)}",
                footprint_url=f"/api/v1/results/pseudo-markup/{result.id}/footprint/{scene_id}?v={raster_revision(path)}",
                bounds=bounds, has_alpha=has_alpha, has_nir=has_nir, nodata=nodata,
            ))
        except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
            warnings.append(f"Снимок {path.name} недоступен для просмотра: {exc}")
    model = result.training_result
    dataset = find_managed_dataset(session, config, model.dataset_key or model.class_key) if model else None
    return PseudoMarkupViewInfo(
        id=result.id, training_result_id=result.training_result_id,
        model_name=model.model_name if model else "Сеть не указана",
        training_dataset_name=dataset.name if dataset else (model.class_display_name if model else "Не указан"),
        source_dataset_name=result.source_dataset_name, created_at=result.created_at,
        geojson_url=f"/api/v1/files/{result.geojson_file_id}/download",
        object_count=result.geojson_file.object_count, expected_image_count=result.image_count,
        scenes=scenes, warnings=warnings,
    )


def pseudo_markup_raster(session: Session, config: TrainingUIAPIConfig, result_id: uuid.UUID, scene_id: str) -> Path:
    result = _ready_result(session, result_id)
    paths, _ = _result_images(session, config, result)
    for path in paths:
        if _scene_id(path) == scene_id and path.is_file() and path.resolve().is_relative_to(config.images_root.resolve()):
            return path
    raise TrainingUIAPIError("Снимок не входит в эту псевдоразметку или больше недоступен")


def _ready_result(session: Session, result_id: uuid.UUID) -> PseudoMarkupResultRow:
    result = session.get(PseudoMarkupResultRow, result_id)
    if result is None or result.status != "ok" or result.geojson_file is None:
        raise TrainingUIAPIError("Готовая псевдоразметка не найдена")
    if not Path(result.geojson_file.path).is_file():
        raise TrainingUIAPIError("Файл псевдоразметки больше недоступен")
    return result


def pseudo_markup_footprint(path: Path) -> dict:
    """Контур фактических данных; общий с редактором ограниченный кэш масок TIFF."""
    geometry = valid_data_footprint(path)
    with rasterio.open(path) as source:
        geometry = transform_geom(source.crs, "EPSG:3857", mapping(geometry))
    return {"type": "Feature", "properties": {}, "geometry": geometry}


def _scene_id(path: Path) -> str:
    return hashlib.sha256(path.as_posix().encode()).hexdigest()[:24]


def _result_images(session, config, result):
    job = session.get(JobRow, result.job_id) if result.job_id else None
    state = (job.config or {}) if job else {}
    warnings = []
    processed = state.get("pseudo_processed_images")
    if processed is None and job and job.tmp_path:
        report_path = Path(job.tmp_path) / "scratch" / "report.json"
        try:
            report = _read_report(str(report_path), report_path.stat().st_mtime_ns)
            processed = [item["image"] for item in report.get("scenes", [])
                         if item.get("status") == "ok" and item.get("image")]
            if not processed:
                processed = None
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    if processed is not None:
        candidates = [Path(value) for value in processed]
    else:
        class_row = dataset_class_row(session, result.dataset_key or result.class_key)
        imagery_type = state.get("imagery_type") or (class_row.imagery_type if class_row else "kanopus")
        root = Path(state.get("images_root") or imagery_images_dir(config.images_root, imagery_type))
        if not root.resolve().is_relative_to(config.images_root.resolve()):
            return [], ["Корень исходных снимков недоступен для просмотра."]
        scenes_file = Path(result.scenes_file.path) if result.scenes_file else None
        if scenes_file is not None and scenes_file.is_file():
            stat = scenes_file.stat()
            annotations = tuple(str(value) for value in state.get("annotation_files", []) if Path(value).is_file())
            candidates, problems = _resolve_saved_scenes(str(root), str(scenes_file), stat.st_mtime_ns,
                                                       stat.st_size, annotations, int(time.monotonic() // 60))
            warnings.extend(problems)
        else:
            dataset = find_managed_dataset(session, config, result.dataset_key) if result.dataset_key else None
            if dataset and result.dataset_version and dataset.version == result.dataset_version and dataset.annotations_dir:
                resolved = resolve_scene_images(SceneImageResolutionRequest(images_dir=dataset.images_dir,
                                                                             annotations_dir=dataset.annotations_dir))
                candidates = [Path(item.image_path) for item in resolved.images]
                warnings.extend(f"Снимок не найден: {value}" for value in resolved.missing_scenes)
                warnings.extend(f"Неоднозначный снимок: {value}" for value in resolved.ambiguous_scenes)
            else:
                return [], ["Сохранённый состав снимков недоступен. Псевдоразметка показана без подложки."]
    return _safe_images(tuple(str(path) for path in candidates), str(config.images_root.resolve()), int(time.monotonic() // 60), tuple(warnings))


@lru_cache(maxsize=64)
def _safe_images(candidates, allowed_root, cache_minute, warnings):
    safe = []
    warnings = list(warnings)
    for value in dict.fromkeys(candidates):
        path = Path(value).resolve()
        if path.suffix.lower() not in RASTER_SUFFIXES or not path.is_relative_to(allowed_root):
            warnings.append(f"Снимок {path.name} находится вне каталога подготовленных снимков.")
        elif not path.is_file():
            warnings.append(f"Исходный снимок больше недоступен: {path.name}")
        else:
            safe.append(path)
    return safe, warnings


@lru_cache(maxsize=32)
def _resolve_saved_scenes(root, scenes_file, mtime_ns, size, annotations, cache_minute):
    result = resolve_scene_images(SceneImageResolutionRequest(images_dir=root, scenes_file=scenes_file,
                                                              annotation_files=list(annotations)))
    warnings = [f"Снимок не найден: {name}" for name in result.missing_scenes]
    warnings.extend(f"Неоднозначный снимок: {name}" for name in result.ambiguous_scenes)
    paths = []
    for item in result.images:
        if any((Path(root) / entry).is_dir() for entry in item.request_scenes):
            warnings.append("Для старого результата не сохранён точный состав папки снимков; её подложка пропущена.")
        else:
            paths.append(Path(item.image_path))
    return paths, list(dict.fromkeys(warnings))


@lru_cache(maxsize=16)
def _read_report(path, mtime_ns):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=512)
def _raster_metadata(path, mtime_ns, size):
    with rasterio.open(path) as source:
        if source.crs is None or source.count < 3:
            raise ValueError("Нужны RGB-каналы и система координат")
        bounds = transform_bounds(source.crs, "EPSG:3857", *source.bounds, densify_pts=21)
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError("Некорректная привязка снимка")
        nodata = float(source.nodata) if source.nodata is not None and math.isfinite(source.nodata) else None
        has_alpha = source.count >= 4 and source.colorinterp[3] == ColorInterp.alpha
        has_nir = source.count >= 4 and not has_alpha
        return bounds, has_alpha, has_nir, nodata

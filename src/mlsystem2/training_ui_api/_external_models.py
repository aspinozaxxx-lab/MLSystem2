"""Приватная поддержка импортированных TorchScript-моделей Training UI."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Callable, Literal
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
import rasterio
from rasterio import features as rasterio_features
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform
from rasterio.windows import Window
from scipy.optimize import differential_evolution
from shapely import make_valid
from shapely.affinity import translate
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.strtree import STRtree


EXTERNAL_ARCHITECTURE = "external_torchscript"
EXTERNAL_MANIFEST_KEY = "external_model"
ExternalAdapter = Literal["detectron2_instances", "oks_multiclass_footprints"]


class ExternalModelError(RuntimeError):
    """Ошибка внешней модели или её манифеста."""


class ExternalModelManifest(BaseModel):
    """Версионированный снимок параметров импортированной модели."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    adapter: ExternalAdapter
    artifact_path: str
    archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_member: str
    model_root: str
    input_channels: Literal[3] = 3
    target_resolution_m: float = Field(gt=0.0)
    tile_size: int = Field(gt=0)
    stride: int = Field(gt=0)
    context: int = Field(ge=0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    min_area_m2: float = Field(default=0.0, ge=0.0)
    min_hole_area_m2: float = Field(default=0.0, ge=0.0)
    nms_iou_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    nms_relative_intersection: float | None = Field(default=None, ge=0.0, le=1.0)
    max_shift_m: float | None = Field(default=None, gt=0.0)
    shift_iterations: int | None = Field(default=None, gt=0)
    shift_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    correction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_profile(self) -> "ExternalModelManifest":
        _safe_archive_path(self.artifact_path, "artifact_path")
        _safe_archive_path(self.model_member, "model_member")
        root = _safe_archive_path(self.model_root, "model_root")
        if "/" in root:
            raise ValueError("model_root должен быть именем корневой папки ZIP")
        if not self.model_member.startswith(f"{root}/"):
            raise ValueError("model_member должен находиться внутри model_root")
        if self.stride > self.tile_size:
            raise ValueError("stride не может превышать tile_size")
        if self.adapter == "detectron2_instances":
            if self.tile_size != self.stride + 2 * self.context:
                raise ValueError("Для instance-профиля tile_size должен включать core и контекст")
            if self.score_threshold is None:
                raise ValueError("Для instance-профиля нужен score_threshold")
            if self.nms_iou_threshold is None or self.nms_relative_intersection is None:
                raise ValueError("Для instance-профиля нужны параметры NMS")
        else:
            if self.tile_size != self.stride + 2 * self.context:
                raise ValueError("Для ОКС-профиля tile_size должен включать core и контекст")
            if any(
                value is None
                for value in (
                    self.max_shift_m,
                    self.shift_iterations,
                    self.shift_confidence,
                    self.correction_confidence,
                )
            ):
                raise ValueError("Для ОКС-профиля нужны параметры коррекции footprint")
        return self


@dataclass
class LoadedExternalModel:
    manifest: ExternalModelManifest
    torch: Any
    model: Any
    device: str


@dataclass(frozen=True)
class _PredictedGeometry:
    geometry: BaseGeometry
    score: float | None = None
    properties: dict[str, Any] | None = None


@dataclass(frozen=True)
class _GeometryPrediction:
    geometries: list[_PredictedGeometry]
    crs: object


@dataclass(frozen=True)
class ExternalTestPrediction:
    mask: np.ndarray
    instances: np.ndarray


def external_model_manifest(
    config: dict[str, Any] | None,
    *,
    architecture: str | None = None,
) -> ExternalModelManifest | None:
    """Прочитать манифест из snapshot задания и проверить согласованность архитектуры."""

    raw = (config or {}).get(EXTERNAL_MANIFEST_KEY)
    if raw is None:
        if architecture == EXTERNAL_ARCHITECTURE:
            raise ExternalModelError("Для external_torchscript отсутствует external_model")
        return None
    if architecture is not None and architecture != EXTERNAL_ARCHITECTURE:
        raise ExternalModelError("external_model допустим только для external_torchscript")
    try:
        return ExternalModelManifest.model_validate(raw)
    except Exception as exc:
        raise ExternalModelError(f"Некорректный манифест внешней модели: {exc}") from exc


def external_model_payload(manifest: ExternalModelManifest | None) -> dict[str, Any] | None:
    return manifest.model_dump(mode="json") if manifest is not None else None


def external_result_manifest(session: Any, result: Any) -> ExternalModelManifest | None:
    """Прочитать неизменяемый манифест из исходного training job результата."""

    if str(getattr(result, "architecture", "")) != EXTERNAL_ARCHITECTURE:
        return None
    job_id = getattr(result, "job_id", None)
    if job_id is None:
        raise ExternalModelError("У внешнего результата отсутствует исходное training job")
    from ._models import JobRow

    source_job = session.get(JobRow, job_id)
    if source_job is None:
        raise ExternalModelError("Исходное training job внешнего результата не найдено")
    return external_model_manifest(
        dict(source_job.config or {}),
        architecture=str(result.architecture),
    )


def validate_external_archive(path: Path, manifest: ExternalModelManifest) -> None:
    """Проверить хэш, структуру и безопасность исходного Triton ZIP."""

    if not path.is_file():
        raise ExternalModelError(f"Архив внешней модели не найден: {path}")
    actual_hash = _file_sha256(path)
    if actual_hash != manifest.archive_sha256:
        raise ExternalModelError(
            f"SHA-256 архива не совпадает: {actual_hash} != {manifest.archive_sha256}"
        )
    try:
        with zipfile.ZipFile(path) as archive:
            names: set[str] = set()
            for item in archive.infolist():
                normalized = _safe_archive_path(item.filename, "путь ZIP")
                if normalized in names:
                    raise ExternalModelError(f"ZIP содержит повторяющийся путь: {normalized}")
                if normalized != manifest.model_root and not normalized.startswith(
                    f"{manifest.model_root}/"
                ):
                    raise ExternalModelError(
                        f"ZIP содержит файл вне корневой папки модели: {normalized}"
                    )
                names.add(normalized)
            required = {
                manifest.model_member,
                f"{manifest.model_root}/config.pbtxt",
            }
            missing = sorted(required - names)
            if missing:
                raise ExternalModelError(
                    f"ZIP не содержит обязательные файлы: {', '.join(missing)}"
                )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ExternalModelError("Не удалось прочитать ZIP внешней модели") from exc


def load_external_model(
    archive_path: Path,
    manifest: ExternalModelManifest,
    *,
    device: str,
    scratch_root: Path,
) -> LoadedExternalModel:
    """Безопасно извлечь TorchScript и загрузить его на выбранное устройство."""

    validate_external_archive(archive_path, manifest)
    model_path = scratch_root / "external-model" / "model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(manifest.model_member) as source, model_path.open("wb") as target:
                shutil.copyfileobj(source, target)
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ExternalModelError("Не удалось извлечь model.pt из внешнего ZIP") from exc
    try:
        import torch

        if manifest.adapter == "detectron2_instances":
            import torchvision  # noqa: F401

            try:
                import detectron2.layers  # type: ignore[import-not-found]  # noqa: F401
            except ImportError:
                pass

            if not hasattr(torch.ops.torchvision, "nms"):
                raise ExternalModelError("TorchVision не зарегистрировал оператор torchvision::nms")
        model = torch.jit.load(str(model_path), map_location=torch.device("cpu"))
        model.to(torch.device(device))
        model.eval()
    except ExternalModelError:
        raise
    except Exception as exc:
        raise ExternalModelError("Не удалось загрузить TorchScript внешней модели") from exc
    return LoadedExternalModel(manifest=manifest, torch=torch, model=model, device=device)


def predict_external_scene(
    loaded: LoadedExternalModel,
    *,
    image_path: Path,
    scene: str,
    config: dict[str, Any],
    aoi_wgs84: BaseGeometry | None = None,
    geometry_postprocessor: Callable[[BaseGeometry, object], BaseGeometry] | None = None,
) -> list[dict[str, Any]]:
    """Распознать один снимок и вернуть совместимые WGS84 GeoJSON features."""

    prediction = _predict_geometries(loaded, image_path, aoi_wgs84=aoi_wgs84)
    prediction = _postprocess_prediction_geometries(prediction, geometry_postprocessor)
    source_crs = str(prediction.crs)
    transformer = Transformer.from_crs(prediction.crs, "EPSG:4326", always_xy=True)
    output: list[dict[str, Any]] = []
    for item in prediction.geometries:
        geometry = transform_geometry(transformer.transform, item.geometry)
        properties = {
            "_x_res": loaded.manifest.target_resolution_m,
            "_y_res": loaded.manifest.target_resolution_m,
            "_crs": source_crs,
            "scene_id": scene,
            "class_key": config.get("class_key"),
            "class_name": config.get("class_name"),
            "source_model": config.get("source_model"),
            "source_run_id": config.get("mlflow_run_id"),
            "source_checkpoint": config.get("checkpoint_uri"),
            "source_threshold": loaded.manifest.score_threshold,
            "confidence": item.score,
            "external_adapter": loaded.manifest.adapter,
        }
        properties.update(item.properties or {})
        output.append(
            {
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": properties,
            }
        )
    return output


def predict_external_test_tile(
    loaded: LoadedExternalModel,
    image_path: Path,
    *,
    geometry_postprocessor: Callable[[BaseGeometry, object], BaseGeometry] | None = None,
) -> ExternalTestPrediction:
    """Распознать тестовый TIFF и вернуть маски в исходной пиксельной сетке."""

    prediction = _predict_geometries(loaded, image_path)
    prediction = _postprocess_prediction_geometries(prediction, geometry_postprocessor)
    with rasterio.open(image_path) as source:
        if source.crs is None:
            raise ExternalModelError("У тестового TIFF отсутствует CRS")
        transformer = (
            Transformer.from_crs(prediction.crs, source.crs, always_xy=True)
            if PyprojCRS.from_user_input(prediction.crs)
            != PyprojCRS.from_user_input(source.crs)
            else None
        )
        shapes: list[tuple[BaseGeometry, int]] = []
        for index, item in enumerate(prediction.geometries, start=1):
            geometry = (
                transform_geometry(transformer.transform, item.geometry)
                if transformer is not None
                else item.geometry
            )
            if not geometry.is_empty:
                shapes.append((geometry, index))
        instances = rasterio_features.rasterize(
            shapes,
            out_shape=(source.height, source.width),
            transform=source.transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        ).astype(np.int64, copy=False)
        valid = source.dataset_mask() > 0
        instances[~valid] = 0
    return ExternalTestPrediction(mask=(instances > 0).astype(np.uint8), instances=instances)


def _postprocess_prediction_geometries(
    prediction: _GeometryPrediction,
    postprocessor: Callable[[BaseGeometry, object], BaseGeometry] | None,
) -> _GeometryPrediction:
    if postprocessor is None:
        return prediction
    output: list[_PredictedGeometry] = []
    for item in prediction.geometries:
        geometry = _polygons_geometry(make_valid(postprocessor(item.geometry, prediction.crs)))
        if geometry.is_empty:
            continue
        output.append(
            _PredictedGeometry(
                geometry=geometry,
                score=item.score,
                properties=item.properties,
            )
        )
    return _GeometryPrediction(geometries=output, crs=prediction.crs)


def merge_external_instance_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Убрать дубли instance-полигонов разных снимков, не объединяя соседние объекты."""

    if len(features) < 2:
        return features
    geometries = [make_valid(shape(item["geometry"])) for item in features]
    scores = [float((item.get("properties") or {}).get("confidence") or 0.0) for item in features]
    keep = _nms_keep_indexes(geometries, scores, iou_threshold=0.75, relative_threshold=0.75)
    occupied: BaseGeometry = GeometryCollection()
    output: list[dict[str, Any]] = []
    for index in sorted(keep, key=lambda item: (-scores[item], item)):
        geometry = geometries[index]
        if not occupied.is_empty and geometry.intersects(occupied):
            geometry = make_valid(geometry.difference(occupied))
        geometry = _polygons_geometry(geometry)
        if geometry.is_empty:
            continue
        output.append({**features[index], "geometry": mapping(geometry)})
        occupied = geometry if occupied.is_empty else unary_union((occupied, geometry))
    return output


def _predict_geometries(
    loaded: LoadedExternalModel,
    image_path: Path,
    *,
    aoi_wgs84: BaseGeometry | None = None,
) -> _GeometryPrediction:
    with _open_resampled_dataset(
        image_path,
        loaded.manifest.target_resolution_m,
    ) as dataset:
        processing_window = _aoi_processing_window(
            dataset,
            aoi_wgs84,
            padding_pixels=loaded.manifest.context,
        )
        if processing_window is None:
            if dataset.crs is None:
                raise ExternalModelError(f"У снимка отсутствует CRS: {image_path}")
            return _GeometryPrediction(geometries=[], crs=dataset.crs)
        if loaded.manifest.adapter == "detectron2_instances":
            geometries = _predict_detectron2_geometries(
                loaded,
                dataset,
                image_path,
                processing_window,
            )
        else:
            geometries = _predict_oks_geometries(
                loaded,
                dataset,
                image_path,
                processing_window,
            )
        if dataset.crs is None:
            raise ExternalModelError(f"У снимка отсутствует CRS: {image_path}")
        return _GeometryPrediction(geometries=geometries, crs=dataset.crs)


def _predict_detectron2_geometries(
    loaded: LoadedExternalModel,
    dataset,
    image_path: Path,
    processing_window: Window,
) -> list[_PredictedGeometry]:
    manifest = loaded.manifest
    output: list[_PredictedGeometry] = []
    grid_window = _aligned_grid_window(dataset, processing_window, manifest.stride)
    for y in range(int(grid_window.row_off), int(grid_window.row_off + grid_window.height), manifest.stride):
        for x in range(int(grid_window.col_off), int(grid_window.col_off + grid_window.width), manifest.stride):
            window = Window(
                x - manifest.context,
                y - manifest.context,
                manifest.tile_size,
                manifest.tile_size,
            )
            image, valid = _read_rgb_window(dataset, image_path, window, manifest.tile_size)
            if not np.any(valid):
                continue
            tensor = loaded.torch.as_tensor(
                image,
                dtype=loaded.torch.uint8,
                device=loaded.torch.device(loaded.device),
            )
            with loaded.torch.no_grad():
                result = loaded.model(tensor)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise ExternalModelError("ЗУ500 должна возвращать tuple(instance_masks, scores)")
            raw_masks, raw_scores = result
            masks = raw_masks.detach().cpu().numpy().astype(bool, copy=False)
            scores = raw_scores.detach().cpu().numpy().astype(np.float32, copy=False)
            if (
                masks.ndim != 3
                or scores.ndim != 1
                or masks.shape[0] != scores.shape[0]
                or masks.shape[1:] != (manifest.tile_size, manifest.tile_size)
                or not np.all(np.isfinite(scores))
            ):
                raise ExternalModelError("ЗУ500 вернула несовместимые размеры masks/scores")
            window_transform = dataset.window_transform(window)
            for mask, raw_score in zip(masks, scores, strict=True):
                score = float(raw_score)
                if score < float(manifest.score_threshold or 0.0):
                    continue
                mask = mask & valid
                geometry = _largest_mask_geometry(mask, window_transform)
                if geometry is None:
                    continue
                geometry = geometry.simplify(manifest.target_resolution_m, preserve_topology=True)
                if geometry.area < 10.0 * manifest.target_resolution_m**2:
                    continue
                output.append(_PredictedGeometry(geometry=geometry, score=score))
    if not output:
        return []
    geometries = [item.geometry for item in output]
    scores = [float(item.score or 0.0) for item in output]
    keep = _nms_keep_indexes(
        geometries,
        scores,
        iou_threshold=float(manifest.nms_iou_threshold or 0.75),
        relative_threshold=float(manifest.nms_relative_intersection or 0.75),
    )
    selected = [output[index] for index in sorted(keep, key=lambda item: (-scores[item], item))]
    return _correct_instance_topology(
        selected,
        min_area_m2=manifest.min_area_m2,
        min_hole_area_m2=manifest.min_hole_area_m2,
    )


def _predict_oks_geometries(
    loaded: LoadedExternalModel,
    dataset,
    image_path: Path,
    processing_window: Window,
) -> list[_PredictedGeometry]:
    manifest = loaded.manifest
    grid_window = _aligned_grid_window(dataset, processing_window, manifest.stride)
    roof_parts: list[BaseGeometry] = []
    wall_parts: list[BaseGeometry] = []
    valid_parts: list[BaseGeometry] = []
    mask_height = int(grid_window.height)
    mask_width = int(grid_window.width)
    row_start = int(grid_window.row_off)
    column_start = int(grid_window.col_off)
    for y in range(row_start, row_start + mask_height, manifest.stride):
        for x in range(column_start, column_start + mask_width, manifest.stride):
            window = Window(
                x - manifest.context,
                y - manifest.context,
                manifest.tile_size,
                manifest.tile_size,
            )
            image, valid = _read_rgb_window(dataset, image_path, window, manifest.tile_size)
            if not np.any(valid):
                continue
            tensor = loaded.torch.as_tensor(
                image[None, ...],
                dtype=loaded.torch.uint8,
                device=loaded.torch.device(loaded.device),
            )
            with loaded.torch.no_grad():
                result = loaded.model(tensor)
            labels = result.detach().cpu().numpy() if hasattr(result, "detach") else np.asarray(result)
            if labels.shape != (1, manifest.tile_size, manifest.tile_size):
                raise ExternalModelError("ОКС500 должна возвращать labels формы [B,H,W]")
            labels = labels[0]
            if np.any(labels < 0) or np.any(labels > 4):
                raise ExternalModelError("ОКС500 вернула неизвестные номера классов")
            core_height = min(manifest.stride, dataset.height - y)
            core_width = min(manifest.stride, dataset.width - x)
            ys = slice(manifest.context, manifest.context + core_height)
            xs = slice(manifest.context, manifest.context + core_width)
            core = labels[ys, xs]
            core_valid = valid[ys, xs]
            core_window = Window(x, y, core_width, core_height)
            core_transform = dataset.window_transform(core_window)
            roof_parts.extend(
                _mask_geometries(
                    (np.isin(core, (1, 2)) & core_valid).astype(np.uint8),
                    core_transform,
                )
            )
            wall_parts.extend(
                _mask_geometries(
                    ((core == 4) & core_valid).astype(np.uint8),
                    core_transform,
                )
            )
            if np.all(core_valid):
                valid_parts.append(box(*rasterio.windows.bounds(core_window, dataset.transform)))
            elif np.any(core_valid):
                valid_parts.extend(
                    _mask_geometries(core_valid.astype(np.uint8), core_transform)
                )
    roofs = list(_iter_polygons(make_valid(unary_union(roof_parts)))) if roof_parts else []
    roofs = [
        _fill_small_holes(make_valid(item), manifest.min_hole_area_m2)
        for item in roofs
        if manifest.min_area_m2 <= item.area <= 100_000.0
    ]
    walls = list(_iter_polygons(make_valid(unary_union(wall_parts)))) if wall_parts else []
    shifted = _shift_roofs_to_footprints(roofs, walls, manifest)
    valid_geometry = make_valid(unary_union(valid_parts)) if valid_parts else GeometryCollection()
    output: list[_PredictedGeometry] = []
    for geometry, confidence in shifted:
        geometry = _polygons_geometry(make_valid(geometry.intersection(valid_geometry)))
        if geometry.is_empty or geometry.area < 5.0:
            continue
        output.append(
            _PredictedGeometry(
                geometry=geometry,
                properties={"_shift_confidence": confidence},
            )
        )
    return output


def _shift_roofs_to_footprints(
    roofs: list[BaseGeometry],
    walls: list[BaseGeometry],
    manifest: ExternalModelManifest,
) -> list[tuple[BaseGeometry, float]]:
    if not roofs:
        return []
    wall_tree = STRtree(walls) if walls else None
    shifts: list[tuple[float, float, float]] = []
    for index, roof in enumerate(roofs):
        nearby = _tree_geometries(wall_tree, walls, roof.buffer(1.0)) if wall_tree is not None else []
        detected_wall = unary_union(nearby) if nearby else GeometryCollection()
        if detected_wall.is_empty:
            shifts.append((0.0, 0.0, 1.0))
            continue
        bounds = _shift_bounds(roof, detected_wall, float(manifest.max_shift_m or 50.0))

        def objective(coords: np.ndarray) -> float:
            generated = _wall_from_roof_shift(roof, float(coords[0]), float(coords[1]))
            return 1.0 - _geometry_iou(detected_wall, generated)

        result = differential_evolution(
            objective,
            bounds,
            tol=1.0,
            maxiter=int(manifest.shift_iterations or 50),
            seed=index,
            workers=1,
            updating="immediate",
            polish=False,
        )
        shifts.append((float(result.x[0]), float(result.x[1]), float(1.0 - result.fun)))

    initial = [
        translate(roof, xoff=x, yoff=y)
        if confidence >= float(manifest.shift_confidence or 0.2)
        else roof
        for roof, (x, y, confidence) in zip(roofs, shifts, strict=True)
    ]
    footprint_tree = STRtree(initial)
    corrected: list[tuple[BaseGeometry, float]] = []
    for index, (roof, footprint, shift) in enumerate(zip(roofs, initial, shifts, strict=True)):
        x, y, confidence = shift
        correction_x = 0.0
        correction_y = 0.0
        correction_confidence = -3.0
        best_neighbor_confidence = confidence
        for other_index in _tree_indexes(footprint_tree, footprint):
            if other_index == index or shifts[other_index][2] <= best_neighbor_confidence:
                continue
            candidate_x, candidate_y, candidate_confidence = shifts[other_index]
            correction_x = candidate_x - x
            correction_y = candidate_y - y
            best_neighbor_confidence = candidate_confidence
        if correction_x != 0.0 or correction_y != 0.0:
            nearby = (
                _tree_geometries(wall_tree, walls, roof.buffer(1.0))
                if wall_tree is not None
                else []
            )
            detected_wall = unary_union(nearby) if nearby else GeometryCollection()
            correction_confidence = _geometry_iou(
                detected_wall,
                _wall_from_roof_shift(footprint, correction_x, correction_y),
            )
        geometry = (
            translate(footprint, xoff=correction_x, yoff=correction_y)
            if correction_confidence >= float(manifest.correction_confidence or 0.05)
            else footprint
        )
        final_confidence = max(confidence, correction_confidence)
        corrected.append((make_valid(geometry), final_confidence))
    return corrected


def _correct_instance_topology(
    items: list[_PredictedGeometry],
    *,
    min_area_m2: float,
    min_hole_area_m2: float,
) -> list[_PredictedGeometry]:
    accepted: list[_PredictedGeometry] = []
    occupied: BaseGeometry = GeometryCollection()
    for item in items:
        geometry = make_valid(item.geometry)
        if not occupied.is_empty and geometry.intersects(occupied):
            geometry = make_valid(geometry.difference(occupied))
        polygons = list(_iter_polygons(geometry))
        polygons = [
            _fill_small_holes(polygon, min_hole_area_m2)
            for polygon in polygons
            if polygon.area >= min_area_m2
        ]
        if not polygons:
            continue
        geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
        accepted.append(
            _PredictedGeometry(
                geometry=geometry,
                score=item.score,
                properties=item.properties,
            )
        )
        occupied = geometry if occupied.is_empty else unary_union((occupied, geometry))
    return accepted


def _nms_keep_indexes(
    geometries: list[BaseGeometry],
    scores: list[float],
    *,
    iou_threshold: float,
    relative_threshold: float,
) -> set[int]:
    if len(geometries) < 2:
        return set(range(len(geometries)))
    tree = STRtree(geometries)
    dropped: set[int] = set()
    for left, geometry in enumerate(geometries):
        for right in _tree_indexes(tree, geometry):
            if right <= left or left in dropped or right in dropped:
                continue
            intersection = geometry.intersection(geometries[right]).area
            if intersection <= 0.0:
                continue
            union = geometry.union(geometries[right]).area
            if union <= 0.0 or intersection / union <= iou_threshold:
                continue
            if scores[left] < scores[right]:
                candidate = left
            elif scores[right] < scores[left]:
                candidate = right
            else:
                candidate = left if geometry.area < geometries[right].area else right
            candidate_area = geometries[candidate].area
            if candidate_area > 0.0 and intersection / candidate_area > relative_threshold:
                dropped.add(candidate)
    return set(range(len(geometries))) - dropped


def _read_rgb_window(
    dataset,
    image_path: Path,
    window: Window,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset.count < 3:
        raise ExternalModelError(f"Для внешней RGB-модели недостаточно каналов: {image_path}")
    raster_window = Window(0, 0, dataset.width, dataset.height)
    try:
        clipped = window.intersection(raster_window)
    except rasterio.errors.WindowError:
        return (
            np.zeros((3, tile_size, tile_size), dtype=np.uint8),
            np.zeros((tile_size, tile_size), dtype=bool),
        )
    height = int(clipped.height)
    width = int(clipped.width)
    source = dataset.read(
        indexes=_rgb_indexes(dataset),
        window=clipped,
        out_shape=(3, height, width),
        out_dtype="float32",
        resampling=Resampling.bilinear,
    )
    clipped_valid = dataset.dataset_mask(
        window=clipped,
        out_shape=(height, width),
        resampling=Resampling.nearest,
    ) > 0
    alpha_indexes = [
        index
        for index, interpretation in enumerate(dataset.colorinterp, start=1)
        if interpretation == ColorInterp.alpha
    ]
    if alpha_indexes:
        alpha = dataset.read(
            alpha_indexes[0],
            window=clipped,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
        clipped_valid &= alpha > 0
    image = np.zeros((3, tile_size, tile_size), dtype=np.float32)
    valid = np.zeros((tile_size, tile_size), dtype=bool)
    y = int(clipped.row_off - window.row_off)
    x = int(clipped.col_off - window.col_off)
    image[:, y : y + height, x : x + width] = source
    valid[y : y + height, x : x + width] = clipped_valid
    image[:, ~valid] = 0
    return np.clip(image, 0, 255).astype(np.uint8, copy=False), valid


def _rgb_indexes(dataset) -> tuple[int, int, int]:
    interpretations = list(dataset.colorinterp)
    rgb = []
    for interpretation in (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
        try:
            rgb.append(interpretations.index(interpretation) + 1)
        except ValueError:
            return (1, 2, 3)
    return tuple(rgb)


def _largest_mask_geometry(mask: np.ndarray, transform: Affine) -> BaseGeometry | None:
    geometries = _mask_geometries(mask.astype(np.uint8, copy=False), transform)
    return max(geometries, key=lambda item: item.area) if geometries else None


def _mask_geometries(mask: np.ndarray, transform: Affine) -> list[BaseGeometry]:
    output: list[BaseGeometry] = []
    for geometry, value in rasterio_features.shapes(
        mask.astype(np.uint8, copy=False),
        mask=mask > 0,
        transform=transform,
    ):
        if int(value) <= 0:
            continue
        normalized = make_valid(shape(geometry))
        output.extend(polygon for polygon in _iter_polygons(normalized) if not polygon.is_empty)
    return output


def _fill_small_holes(geometry: BaseGeometry, threshold: float) -> BaseGeometry:
    if threshold <= 0.0:
        return geometry
    polygons = []
    for polygon in _iter_polygons(make_valid(geometry)):
        holes = [ring.coords for ring in polygon.interiors if Polygon(ring).area >= threshold]
        polygons.append(Polygon(polygon.exterior.coords, holes))
    if not polygons:
        return GeometryCollection()
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def _wall_from_roof_shift(roof: BaseGeometry, x: float, y: float) -> BaseGeometry:
    footprint = translate(roof, xoff=x, yoff=y)
    swept = _sweep_geometry(footprint, -x, -y)
    return make_valid(swept).difference(roof)


def _sweep_geometry(geometry: BaseGeometry, x: float, y: float) -> BaseGeometry:
    polygons: list[BaseGeometry] = []
    for polygon in _iter_polygons(geometry):
        rings = [polygon.exterior, *polygon.interiors]
        for ring in rings:
            coordinates = list(ring.coords)
            for first, second in zip(coordinates, coordinates[1:]):
                polygons.append(
                    Polygon(
                        [
                            first,
                            (first[0] + x, first[1] + y),
                            (second[0] + x, second[1] + y),
                            second,
                        ]
                    )
                )
    return unary_union(polygons) if polygons else GeometryCollection()


def _shift_bounds(
    roof: BaseGeometry,
    walls: BaseGeometry,
    max_shift: float,
) -> list[tuple[float, float]]:
    roof_bounds = roof.bounds
    wall_bounds = walls.bounds
    left = max(min(0.0, wall_bounds[0] - roof_bounds[0]), -max_shift)
    right = min(max(0.0, wall_bounds[2] - roof_bounds[2]), max_shift)
    bottom = max(min(0.0, wall_bounds[1] - roof_bounds[1]), -max_shift)
    top = min(max(0.0, wall_bounds[3] - roof_bounds[3]), max_shift)
    if math.isclose(left, right):
        if right < max_shift:
            right = min(max_shift, right + 1e-6)
        else:
            left = max(-max_shift, left - 1e-6)
    if math.isclose(bottom, top):
        if top < max_shift:
            top = min(max_shift, top + 1e-6)
        else:
            bottom = max(-max_shift, bottom - 1e-6)
    return [(left, right), (bottom, top)]


def _geometry_iou(left: BaseGeometry, right: BaseGeometry) -> float:
    if left.is_empty or right.is_empty:
        return 0.0
    intersection = left.intersection(right).area
    union = left.union(right).area
    return intersection / union if union > 0.0 else 0.0


def _tree_indexes(tree: STRtree, geometry: BaseGeometry) -> list[int]:
    result = tree.query(geometry, predicate="intersects")
    if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.integer):
        return [int(item) for item in result]
    source = list(tree.geometries)
    by_identity = {id(item): index for index, item in enumerate(source)}
    return [by_identity[id(item)] for item in result if id(item) in by_identity]


def _tree_geometries(
    tree: STRtree | None,
    geometries: list[BaseGeometry],
    query: BaseGeometry,
) -> list[BaseGeometry]:
    if tree is None:
        return []
    return [geometries[index] for index in _tree_indexes(tree, query)]


def _iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def _polygons_geometry(geometry: BaseGeometry) -> BaseGeometry:
    polygons = [polygon for polygon in _iter_polygons(geometry) if not polygon.is_empty]
    if not polygons:
        return GeometryCollection()
    return polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)


def _aoi_processing_window(
    dataset,
    aoi_wgs84: BaseGeometry | None,
    *,
    padding_pixels: int,
) -> Window | None:
    if aoi_wgs84 is None:
        return Window(0, 0, dataset.width, dataset.height)
    if dataset.crs is None:
        raise ExternalModelError("У исходного снимка отсутствует CRS")
    transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
    raster_aoi = transform_geometry(transformer.transform, aoi_wgs84)
    try:
        window = rasterio_features.geometry_window(
            dataset,
            [mapping(raster_aoi)],
            pad_x=padding_pixels,
            pad_y=padding_pixels,
            boundless=False,
        )
    except rasterio.errors.WindowError:
        return None
    return window.round_offsets().round_lengths()


def _aligned_grid_window(dataset, window: Window, stride: int) -> Window:
    column_start = max(0, int(math.floor(float(window.col_off) / stride)) * stride)
    row_start = max(0, int(math.floor(float(window.row_off) / stride)) * stride)
    column_stop = min(
        dataset.width,
        int(math.ceil(float(window.col_off + window.width) / stride)) * stride,
    )
    row_stop = min(
        dataset.height,
        int(math.ceil(float(window.row_off + window.height) / stride)) * stride,
    )
    return Window(
        column_start,
        row_start,
        max(0, column_stop - column_start),
        max(0, row_stop - row_start),
    )


class _ResampledDataset:
    def __init__(self, image_path: Path, resolution: float) -> None:
        self.image_path = image_path
        self.resolution = resolution
        self.stack = ExitStack()
        self.dataset = None

    def __enter__(self):
        source = self.stack.enter_context(rasterio.open(self.image_path))
        if source.crs is None:
            raise ExternalModelError(f"Нельзя привести разрешение снимка без CRS: {self.image_path}")
        target_crs = _metric_target_crs(source)
        transform, width, height = calculate_default_transform(
            source.crs,
            target_crs,
            source.width,
            source.height,
            *source.bounds,
            resolution=self.resolution,
        )
        self.dataset = self.stack.enter_context(
            WarpedVRT(
                source,
                crs=target_crs,
                transform=transform,
                width=max(1, width),
                height=max(1, height),
                resampling=Resampling.bilinear,
            )
        )
        return self.dataset

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stack.close()


def _open_resampled_dataset(image_path: Path, resolution: float) -> _ResampledDataset:
    return _ResampledDataset(image_path, resolution)


def _metric_target_crs(dataset) -> object:
    source_crs = PyprojCRS.from_user_input(dataset.crs)
    if source_crs.is_projected and source_crs.axis_info:
        factor = source_crs.axis_info[0].unit_conversion_factor
        if factor is not None and math.isclose(float(factor), 1.0, rel_tol=1e-6):
            return dataset.crs
    center_x = (dataset.bounds.left + dataset.bounds.right) / 2.0
    center_y = (dataset.bounds.bottom + dataset.bounds.top) / 2.0
    lon, lat = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True).transform(
        center_x,
        center_y,
    )
    zone = max(1, min(60, int((float(lon) + 180.0) // 6.0) + 1))
    return PyprojCRS.from_epsg((32600 if float(lat) >= 0.0 else 32700) + zone)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_archive_path(value: str, field_name: str) -> str:
    raw = value.strip()
    if raw.startswith(("/", "\\")) or "\x00" in raw:
        raise ValueError(f"{field_name} содержит небезопасный путь")
    normalized = raw.replace("\\", "/").rstrip("/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"{field_name} содержит небезопасный путь")
    return path.as_posix()


__all__ = [
    "EXTERNAL_ARCHITECTURE",
    "EXTERNAL_MANIFEST_KEY",
    "ExternalModelError",
    "ExternalModelManifest",
    "ExternalTestPrediction",
    "LoadedExternalModel",
    "external_model_manifest",
    "external_model_payload",
    "external_result_manifest",
    "load_external_model",
    "merge_external_instance_features",
    "predict_external_scene",
    "predict_external_test_tile",
    "validate_external_archive",
]

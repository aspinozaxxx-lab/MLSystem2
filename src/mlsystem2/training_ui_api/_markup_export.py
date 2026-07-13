"""Самостоятельное формирование временного набора тестовой разметки."""

from __future__ import annotations

import json
import re
import shutil
import uuid
import warnings
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import rasterize
from rasterio.shutil import copy as raster_copy
from rasterio.windows import Window, bounds as window_bounds
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from ._config import TrainingUIAPIConfig
from ._datasets import find_dataset, resolve_scenes_file_images
from .contracts import (
    MarkupExportInfo,
    MarkupExportRequest,
    MarkupExportTileInfo,
    TrainingUIAPIError,
)


EXPORT_TTL = timedelta(hours=1)
EXPORT_ROOT_NAME = "markup-exports"
MANIFEST_NAME = "manifest.json"
ARCHIVE_NAME = "archive.zip"
_ORIGIN_FRACTIONS = (
    (0.5, 0.5),
    (0.25, 0.5),
    (0.75, 0.5),
    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.25),
    (0.25, 0.75),
    (0.75, 0.25),
    (0.75, 0.75),
)
_MILP_TIME_LIMIT_SECONDS = 15.0


class MarkupExportUnavailable(FileNotFoundError):
    """Временный экспорт не найден или уже просрочен."""


@dataclass(frozen=True)
class MarkupExportArtifact:
    info: MarkupExportInfo
    archive_path: Path
    archive_filename: str
    preview_paths: dict[int, Path]


@dataclass(frozen=True)
class _AnnotationFeature:
    source_index: int
    geometry: BaseGeometry
    properties: dict[str, Any]
    feature_id: Any
    has_feature_id: bool


@dataclass(frozen=True)
class _AnnotationSet:
    payload: dict[str, Any]
    crs: PyprojCRS
    features: tuple[_AnnotationFeature, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _TransformedAnnotations:
    geometries: tuple[BaseGeometry, ...]
    tree: STRtree


@dataclass(frozen=True)
class _Candidate:
    source_path: Path
    source_name: str
    territory: str
    column: int
    row: int
    raster_crs: PyprojCRS
    raster_footprint: BaseGeometry
    annotation_footprint: BaseGeometry
    feature_positions: tuple[int, ...]

    @property
    def object_count(self) -> int:
        return len(self.feature_positions)


def build_markup_export(
    request: MarkupExportRequest,
    config: TrainingUIAPIConfig,
) -> MarkupExportInfo:
    cleanup_expired_markup_exports(config)
    dataset = find_dataset(config.mlmarkup_root, request.dataset_key)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError("Для экспорта разметки нужен существующий датасет MLMarkup.")
    if dataset.diagnostics:
        raise TrainingUIAPIError("; ".join(dataset.diagnostics))
    if not dataset.scenes_file or not dataset.annotation_file:
        raise TrainingUIAPIError("У датасета должны быть TXT со сценами и один positive GeoJSON.")

    source_paths = resolve_scenes_file_images(Path(dataset.scenes_file), config.images_root)
    if not source_paths:
        raise TrainingUIAPIError(
            "Для датасета не найдены снимки в MLSYSTEM2_IMAGES_ROOT."
        )
    annotations = _load_annotations(Path(dataset.annotation_file))
    candidates = _build_candidates(
        source_paths=source_paths,
        images_root=config.images_root,
        annotations=annotations,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
    )
    if len(candidates) < request.image_count:
        raise TrainingUIAPIError(
            "Недостаточно полностью валидных тайлов с объектами: "
            f"найдено {len(candidates)}, требуется {request.image_count}."
        )

    selected_indices = _select_candidates(candidates, request, allow_touching=False)
    touching_allowed = False
    if selected_indices is None:
        selected_indices = _select_candidates(candidates, request, allow_touching=True)
        touching_allowed = selected_indices is not None
    if selected_indices is None:
        raise TrainingUIAPIError(
            "Невозможно сформировать заданное количество непересекающихся тайлов "
            "без повторного использования объектов."
        )

    selected = [candidates[index] for index in selected_indices]
    selected.sort(
        key=lambda item: (
            item.territory.casefold(),
            item.source_name.casefold(),
            item.row,
            item.column,
        )
    )
    warnings = list(annotations.warnings)
    actual_object_count = sum(item.object_count for item in selected)
    if actual_object_count != request.object_count:
        warnings.append(
            "Точное число объектов недостижимо: "
            f"запрошено {request.object_count}, сформировано {actual_object_count}."
        )
    if touching_allowed:
        warnings.append(
            "Для формирования полного набора разрешено касание границ тайлов; "
            "перекрытий между тайлами нет."
        )

    export_id = uuid.uuid4()
    export_root = _export_root(config)
    export_root.mkdir(parents=True, exist_ok=True)
    building_root = export_root / f".building-{export_id}"
    final_root = export_root / str(export_id)
    building_root.mkdir(parents=False, exist_ok=False)
    try:
        tile_infos, preview_files = _write_selected_tiles(
            output_root=building_root,
            export_id=export_id,
            selected=selected,
            annotations=annotations,
            tile_width=request.tile_width,
            tile_height=request.tile_height,
        )
        dataset_stem = _safe_name(
            (dataset.class_name or dataset.name.split("\\", maxsplit=1)[0]).casefold(),
            fallback="markup",
        )
        archive_filename = f"{dataset_stem}_test_markup.zip"
        _zip_tile_files(building_root, building_root / ARCHIVE_NAME)
        expires_at = _utc_now() + EXPORT_TTL
        info = MarkupExportInfo(
            id=export_id,
            dataset_key=dataset.key,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            tile_width=request.tile_width,
            tile_height=request.tile_height,
            image_count=len(tile_infos),
            requested_object_count=request.object_count,
            actual_object_count=actual_object_count,
            territory_count=len({item.territory for item in selected}),
            warnings=warnings,
            expires_at=expires_at,
            download_url=f"/api/v1/markup-export/{export_id}/download",
            tiles=tile_infos,
        )
        _write_manifest(
            building_root / MANIFEST_NAME,
            info=info,
            archive_filename=archive_filename,
            preview_files=preview_files,
        )
        building_root.replace(final_root)
        return info
    except TrainingUIAPIError:
        shutil.rmtree(building_root, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(building_root, ignore_errors=True)
        raise TrainingUIAPIError(f"Не удалось сформировать экспорт разметки: {exc}") from exc


def load_markup_export(
    export_id: uuid.UUID,
    config: TrainingUIAPIConfig,
) -> MarkupExportArtifact:
    cleanup_expired_markup_exports(config)
    root = _export_root(config) / str(export_id)
    manifest_path = root / MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = MarkupExportInfo.model_validate(payload["info"])
        archive_filename = str(payload["archive_filename"])
        preview_files = {
            int(index): _safe_child(root, str(filename))
            for index, filename in dict(payload["preview_files"]).items()
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise MarkupExportUnavailable(str(export_id)) from exc
    if info.expires_at <= _utc_now():
        shutil.rmtree(root, ignore_errors=True)
        raise MarkupExportUnavailable(str(export_id))
    archive_path = _safe_child(root, ARCHIVE_NAME)
    if not archive_path.is_file() or any(not path.is_file() for path in preview_files.values()):
        raise MarkupExportUnavailable(str(export_id))
    return MarkupExportArtifact(
        info=info,
        archive_path=archive_path,
        archive_filename=archive_filename,
        preview_paths=preview_files,
    )


def cleanup_expired_markup_exports(
    config: TrainingUIAPIConfig,
    *,
    now: datetime | None = None,
    remove_incomplete: bool = False,
) -> None:
    root = _export_root(config)
    if not root.exists():
        return
    current = now or _utc_now()
    building_deadline = current - EXPORT_TTL
    for child in root.iterdir():
        if child.is_symlink():
            child.unlink(missing_ok=True)
            continue
        if not child.is_dir():
            continue
        if child.name.startswith(".building-"):
            if remove_incomplete:
                shutil.rmtree(child, ignore_errors=True)
                continue
            try:
                modified_at = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if modified_at <= building_deadline:
                shutil.rmtree(child, ignore_errors=True)
            continue
        manifest_path = child / MANIFEST_NAME
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            expires_at = MarkupExportInfo.model_validate(payload["info"]).expires_at
        except (OSError, ValueError, KeyError, TypeError):
            try:
                modified_at = datetime.fromtimestamp(child.stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue
            if modified_at <= building_deadline:
                shutil.rmtree(child, ignore_errors=True)
            continue
        if expires_at <= current:
            shutil.rmtree(child, ignore_errors=True)


def _load_annotations(path: Path) -> _AnnotationSet:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise TrainingUIAPIError(
            f"Не удалось прочитать GeoJSON положительной разметки: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise TrainingUIAPIError(
            "GeoJSON положительной разметки должен быть FeatureCollection."
        )
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise TrainingUIAPIError(
            "В GeoJSON положительной разметки отсутствует список features."
        )
    crs = _geojson_crs(payload)
    features: list[_AnnotationFeature] = []
    skipped = 0
    for source_index, raw_feature in enumerate(raw_features):
        if not isinstance(raw_feature, dict) or not raw_feature.get("geometry"):
            skipped += 1
            continue
        try:
            geometry = _polygonal_geometry(shape(raw_feature["geometry"]))
        except Exception:  # noqa: BLE001
            geometry = None
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            skipped += 1
            continue
        properties = raw_feature.get("properties")
        features.append(
            _AnnotationFeature(
                source_index=source_index,
                geometry=geometry,
                properties=dict(properties) if isinstance(properties, dict) else {},
                feature_id=raw_feature.get("id"),
                has_feature_id="id" in raw_feature,
            )
        )
    if not features:
        raise TrainingUIAPIError(
            "GeoJSON положительной разметки не содержит валидных полигональных объектов."
        )
    warnings = (
        (f"Пропущено некорректных или неполигональных объектов: {skipped}.",)
        if skipped
        else ()
    )
    return _AnnotationSet(
        payload=payload,
        crs=crs,
        features=tuple(features),
        warnings=warnings,
    )


def _geojson_crs(payload: dict[str, Any]) -> PyprojCRS:
    raw_crs = payload.get("crs")
    if raw_crs is None:
        return PyprojCRS.from_epsg(4326)
    value: Any = raw_crs
    if isinstance(raw_crs, dict):
        properties = raw_crs.get("properties")
        if isinstance(properties, dict):
            value = properties.get("name") or properties.get("href")
        if not value:
            value = raw_crs.get("name")
    try:
        return PyprojCRS.from_user_input(value)
    except Exception as exc:  # noqa: BLE001
        raise TrainingUIAPIError(
            f"Не удалось определить CRS GeoJSON положительной разметки: {value}"
        ) from exc


def _build_candidates(
    *,
    source_paths: list[Path],
    images_root: Path,
    annotations: _AnnotationSet,
    tile_width: int,
    tile_height: int,
) -> list[_Candidate]:
    transformed_cache: dict[str, _TransformedAnnotations] = {}
    candidates: dict[tuple[Path, int, int], _Candidate] = {}
    root = Path(images_root).resolve()
    for source_path in source_paths:
        try:
            relative_path = Path(source_path).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        try:
            with rasterio.open(source_path) as dataset:
                if dataset.crs is None or dataset.width < tile_width or dataset.height < tile_height:
                    continue
                raster_crs = PyprojCRS.from_user_input(dataset.crs)
                transformed = _annotations_for_crs(
                    annotations,
                    raster_crs,
                    transformed_cache,
                )
                image_footprint = box(*dataset.bounds)
                feature_indices = transformed.tree.query(
                    image_footprint,
                    predicate="intersects",
                )
                for raw_feature_index in feature_indices:
                    feature_index = int(raw_feature_index)
                    clipped_to_image = transformed.geometries[feature_index].intersection(
                        image_footprint
                    )
                    if clipped_to_image.is_empty:
                        continue
                    point = clipped_to_image.representative_point()
                    pixel_column, pixel_row = (~dataset.transform) * (point.x, point.y)
                    for x_fraction, y_fraction in _ORIGIN_FRACTIONS:
                        column = _clamped_origin(
                            pixel_column - tile_width * x_fraction,
                            dataset.width - tile_width,
                        )
                        row = _clamped_origin(
                            pixel_row - tile_height * y_fraction,
                            dataset.height - tile_height,
                        )
                        key = (Path(source_path).resolve(), column, row)
                        if key in candidates:
                            break
                        window = Window(column, row, tile_width, tile_height)
                        if not _window_is_fully_valid(dataset, window):
                            continue
                        raster_footprint = box(*window_bounds(window, dataset.transform))
                        object_indices = tuple(
                            sorted(
                                int(index)
                                for index in transformed.tree.query(
                                    raster_footprint,
                                    predicate="intersects",
                                )
                                if transformed.geometries[int(index)]
                                .intersection(raster_footprint)
                                .area
                                > 0.0
                            )
                        )
                        if not object_indices:
                            continue
                        annotation_footprint = _transform_between_crs(
                            raster_footprint,
                            raster_crs,
                            annotations.crs,
                        )
                        candidates[key] = _Candidate(
                            source_path=Path(source_path).resolve(),
                            source_name=relative_path.as_posix(),
                            territory=(
                                relative_path.parent.as_posix()
                                if relative_path.parent.as_posix() != "."
                                else "корень"
                            ),
                            column=column,
                            row=row,
                            raster_crs=raster_crs,
                            raster_footprint=raster_footprint,
                            annotation_footprint=annotation_footprint,
                            feature_positions=object_indices,
                        )
                        break
        except (OSError, rasterio.errors.RasterioError):
            continue
    return sorted(
        candidates.values(),
        key=lambda item: (
            item.territory.casefold(),
            item.source_name.casefold(),
            item.row,
            item.column,
        ),
    )


def _annotations_for_crs(
    annotations: _AnnotationSet,
    target_crs: PyprojCRS,
    cache: dict[str, _TransformedAnnotations],
) -> _TransformedAnnotations:
    key = target_crs.to_wkt()
    cached = cache.get(key)
    if cached is not None:
        return cached
    geometries = tuple(
        _transform_between_crs(feature.geometry, annotations.crs, target_crs)
        for feature in annotations.features
    )
    result = _TransformedAnnotations(geometries=geometries, tree=STRtree(geometries))
    cache[key] = result
    return result


def _transform_between_crs(
    geometry: BaseGeometry,
    source_crs: PyprojCRS,
    target_crs: PyprojCRS,
) -> BaseGeometry:
    if source_crs == target_crs:
        return geometry
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def _clamped_origin(value: float, maximum: int) -> int:
    return max(0, min(maximum, int(round(value))))


def _window_is_fully_valid(
    dataset: rasterio.io.DatasetReader,
    window: Window,
    *,
    image: np.ndarray | None = None,
) -> bool:
    valid_mask = dataset.dataset_mask(window=window)
    expected_shape = (int(window.height), int(window.width))
    if valid_mask.shape != expected_shape or not bool(np.all(valid_mask != 0)):
        return False
    pixels = dataset.read(window=window) if image is None else image
    if pixels.shape != (dataset.count, *expected_shape):
        return False
    if not bool(np.all(np.isfinite(pixels))):
        return False
    return not bool(np.any(np.all(pixels == 0, axis=0)))


def _select_candidates(
    candidates: list[_Candidate],
    request: MarkupExportRequest,
    *,
    allow_touching: bool,
) -> list[int] | None:
    candidate_count = len(candidates)
    territories = sorted({item.territory for item in candidates}, key=str.casefold)
    sources = sorted({item.source_name for item in candidates}, key=str.casefold)
    territory_index = {value: index for index, value in enumerate(territories)}
    source_index = {value: index for index, value in enumerate(sources)}
    territory_offset = candidate_count
    source_offset = territory_offset + len(territories)
    deviation_index = source_offset + len(sources)
    variable_count = deviation_index + 1

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(
        coefficients: list[tuple[int, float]],
        *,
        minimum: float = -np.inf,
        maximum: float = np.inf,
    ) -> None:
        row_index = len(lower)
        lower.append(minimum)
        upper.append(maximum)
        for column, value in coefficients:
            rows.append(row_index)
            columns.append(column)
            values.append(float(value))

    add_constraint(
        [(index, 1.0) for index in range(candidate_count)],
        minimum=request.image_count,
        maximum=request.image_count,
    )
    for left, right in _candidate_conflicts(candidates, allow_touching=allow_touching):
        add_constraint([(left, 1.0), (right, 1.0)], maximum=1.0)

    for territory, index in territory_index.items():
        candidate_indices = [
            candidate_index
            for candidate_index, item in enumerate(candidates)
            if item.territory == territory
        ]
        variable_index = territory_offset + index
        add_constraint(
            [(variable_index, 1.0)]
            + [(candidate_index, -1.0) for candidate_index in candidate_indices],
            maximum=0.0,
        )
        add_constraint(
            [(candidate_index, 1.0) for candidate_index in candidate_indices]
            + [(variable_index, -float(request.image_count))],
            maximum=0.0,
        )

    for source, index in source_index.items():
        candidate_indices = [
            candidate_index
            for candidate_index, item in enumerate(candidates)
            if item.source_name == source
        ]
        variable_index = source_offset + index
        add_constraint(
            [(variable_index, 1.0)]
            + [(candidate_index, -1.0) for candidate_index in candidate_indices],
            maximum=0.0,
        )
        add_constraint(
            [(candidate_index, 1.0) for candidate_index in candidate_indices]
            + [(variable_index, -float(request.image_count))],
            maximum=0.0,
        )

    object_counts = np.asarray([item.object_count for item in candidates], dtype=float)
    add_constraint(
        [(index, object_counts[index]) for index in range(candidate_count)]
        + [(deviation_index, -1.0)],
        maximum=float(request.object_count),
    )
    add_constraint(
        [(index, -object_counts[index]) for index in range(candidate_count)]
        + [(deviation_index, -1.0)],
        maximum=float(-request.object_count),
    )

    matrix = coo_matrix(
        (values, (rows, columns)),
        shape=(len(lower), variable_count),
        dtype=float,
    ).tocsr()
    common_constraints: list[LinearConstraint] = [
        LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    ]
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    upper_bounds[deviation_index] = float(
        request.object_count + sum(sorted(object_counts, reverse=True)[: request.image_count]) + 1
    )
    integrality = np.ones(variable_count, dtype=int)
    integrality[deviation_index] = 0
    bounds = Bounds(lower_bounds, upper_bounds)

    territory_objective = np.zeros(variable_count, dtype=float)
    territory_objective[territory_offset:source_offset] = -1.0
    territory_result = _run_milp(
        territory_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=common_constraints,
        allow_infeasible=True,
    )
    if territory_result is None:
        return None
    territory_optimum = int(
        round(float(np.sum(territory_result[territory_offset:source_offset])))
    )
    territory_constraint = _single_constraint(
        variable_count,
        [(index, 1.0) for index in range(territory_offset, source_offset)],
        minimum=territory_optimum,
        maximum=territory_optimum,
    )

    source_objective = np.zeros(variable_count, dtype=float)
    source_objective[source_offset:deviation_index] = -1.0
    source_result = _run_milp(
        source_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[*common_constraints, territory_constraint],
    )
    if source_result is None:
        return None
    source_optimum = int(round(float(np.sum(source_result[source_offset:deviation_index]))))
    source_constraint = _single_constraint(
        variable_count,
        [(index, 1.0) for index in range(source_offset, deviation_index)],
        minimum=source_optimum,
        maximum=source_optimum,
    )

    deviation_objective = np.zeros(variable_count, dtype=float)
    deviation_objective[deviation_index] = 1.0
    deviation_result = _run_milp(
        deviation_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[*common_constraints, territory_constraint, source_constraint],
    )
    if deviation_result is None:
        return None
    deviation_optimum = int(round(float(deviation_result[deviation_index])))
    deviation_constraint = _single_constraint(
        variable_count,
        [(deviation_index, 1.0)],
        minimum=0.0,
        maximum=float(deviation_optimum) + 1e-6,
    )

    density_objective = np.zeros(variable_count, dtype=float)
    density_objective[:candidate_count] = -object_counts
    density_result = _run_milp(
        density_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[
            *common_constraints,
            territory_constraint,
            deviation_constraint,
            source_constraint,
        ],
    )
    if density_result is None:
        return None
    object_optimum = int(
        round(
            sum(
                candidates[index].object_count
                for index in range(candidate_count)
                if density_result[index] > 0.5
            )
        )
    )
    object_constraint = _single_constraint(
        variable_count,
        [(index, object_counts[index]) for index in range(candidate_count)],
        minimum=object_optimum,
        maximum=object_optimum,
    )
    stable_objective = np.zeros(variable_count, dtype=float)
    stable_objective[:candidate_count] = np.arange(1, candidate_count + 1, dtype=float)
    stable_result = _run_milp(
        stable_objective,
        integrality=integrality,
        bounds=bounds,
        constraints=[
            *common_constraints,
            territory_constraint,
            deviation_constraint,
            source_constraint,
            object_constraint,
        ],
    )
    if stable_result is None:
        return None
    selected = [index for index in range(candidate_count) if stable_result[index] > 0.5]
    return selected if len(selected) == request.image_count else None


def _candidate_conflicts(
    candidates: list[_Candidate],
    *,
    allow_touching: bool,
) -> set[tuple[int, int]]:
    conflicts: set[tuple[int, int]] = set()
    by_feature: dict[int, list[int]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidates):
        for feature_index in candidate.feature_positions:
            by_feature[feature_index].append(candidate_index)
    for candidate_indices in by_feature.values():
        for left, right in combinations(candidate_indices, 2):
            conflicts.add((min(left, right), max(left, right)))

    footprints = [item.annotation_footprint for item in candidates]
    tree = STRtree(footprints)
    for left, footprint in enumerate(footprints):
        for raw_right in tree.query(footprint, predicate="intersects"):
            right = int(raw_right)
            if right <= left:
                continue
            if allow_touching and footprint.intersection(footprints[right]).area <= 0.0:
                continue
            conflicts.add((left, right))
    return conflicts


def _run_milp(
    objective: np.ndarray,
    *,
    integrality: np.ndarray,
    bounds: Bounds,
    constraints: list[LinearConstraint],
    allow_infeasible: bool = False,
) -> np.ndarray | None:
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"presolve": True, "time_limit": _MILP_TIME_LIMIT_SECONDS},
    )
    if allow_infeasible and result.status == 2:
        return None
    if not result.success or result.x is None:
        raise TrainingUIAPIError(
            "Не удалось оптимально подобрать тестовые тайлы за допустимое время."
        )
    return np.asarray(result.x)


def _single_constraint(
    variable_count: int,
    coefficients: list[tuple[int, float]],
    *,
    minimum: float,
    maximum: float,
) -> LinearConstraint:
    matrix = coo_matrix(
        (
            [float(value) for _, value in coefficients],
            ([0] * len(coefficients), [index for index, _ in coefficients]),
        ),
        shape=(1, variable_count),
        dtype=float,
    ).tocsr()
    return LinearConstraint(matrix, [minimum], [maximum])


def _write_selected_tiles(
    *,
    output_root: Path,
    export_id: uuid.UUID,
    selected: list[_Candidate],
    annotations: _AnnotationSet,
    tile_width: int,
    tile_height: int,
) -> tuple[list[MarkupExportTileInfo], dict[int, str]]:
    tile_infos: list[MarkupExportTileInfo] = []
    preview_files: dict[int, str] = {}
    for index, candidate in enumerate(selected, start=1):
        base_name = f"tile_{index:03d}"
        tif_path = output_root / f"{base_name}.tif"
        geojson_path = output_root / f"{base_name}.geojson"
        mask_path = output_root / f"{base_name}_mask.png"
        preview_path = output_root / f"{base_name}_preview.png"
        window = Window(candidate.column, candidate.row, tile_width, tile_height)
        with rasterio.open(candidate.source_path) as source:
            image = source.read(window=window)
            if not _window_is_fully_valid(source, window, image=image):
                raise TrainingUIAPIError(
                    f"Тайл {base_name} пересекает чёрную область или область без данных."
                )
            transform = source.window_transform(window)
            _write_geotiff(source, window, image, tif_path)

        clipped_features = _clipped_features(candidate, annotations)
        if len(clipped_features) != candidate.object_count:
            raise TrainingUIAPIError(
                f"Не удалось сохранить все объекты разметки для тайла {base_name}."
            )
        _write_geojson(geojson_path, annotations.payload, clipped_features)
        raster_geometries = [
            _transform_between_crs(
                feature.geometry.intersection(candidate.annotation_footprint),
                annotations.crs,
                candidate.raster_crs,
            )
            for feature in (
                annotations.features[position] for position in candidate.feature_positions
            )
        ]
        mask = rasterize(
            [(geometry, 255) for geometry in raster_geometries if not geometry.is_empty],
            out_shape=(tile_height, tile_width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        overlay = _overlay_image(image, mask)
        _write_png(mask_path, mask[np.newaxis, :, :])
        _write_png(preview_path, overlay.transpose(2, 0, 1))
        preview_files[index] = preview_path.name
        tile_infos.append(
            MarkupExportTileInfo(
                index=index,
                source_name=candidate.source_name,
                territory=candidate.territory,
                object_count=candidate.object_count,
                preview_url=(
                    f"/api/v1/markup-export/{export_id}/tiles/{index}/preview"
                ),
            )
        )
    return tile_infos, preview_files


def _clipped_features(
    candidate: _Candidate,
    annotations: _AnnotationSet,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for position in candidate.feature_positions:
        feature = annotations.features[position]
        clipped = _polygonal_geometry(
            feature.geometry.intersection(candidate.annotation_footprint)
        )
        if clipped is None or clipped.is_empty or clipped.area <= 0.0:
            continue
        if isinstance(clipped, Polygon):
            clipped = MultiPolygon([clipped])
        payload: dict[str, Any] = {
            "type": "Feature",
            "properties": feature.properties,
            "geometry": mapping(clipped),
        }
        if feature.has_feature_id:
            payload["id"] = feature.feature_id
        result.append(payload)
    return result


def _polygonal_geometry(geometry: BaseGeometry) -> Polygon | MultiPolygon | None:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [
            part
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]
        if not polygons:
            return None
        merged = unary_union(polygons)
        return merged if isinstance(merged, (Polygon, MultiPolygon)) else None
    return None


def _write_geotiff(
    source: rasterio.io.DatasetReader,
    window: Window,
    image: np.ndarray,
    output_path: Path,
) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.source.tif")
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        width=int(window.width),
        height=int(window.height),
        count=source.count,
        transform=source.window_transform(window),
    )
    with rasterio.open(temporary_path, "w", **profile) as target:
        target.write(image)
        default_tags = source.tags()
        if default_tags:
            target.update_tags(**default_tags)
        for namespace in source.tag_namespaces():
            if namespace in {"IMAGE_STRUCTURE", "DERIVED_SUBDATASETS"}:
                continue
            tags = source.tags(ns=namespace)
            if tags:
                target.update_tags(ns=namespace, **tags)
        for band in range(1, source.count + 1):
            tags = source.tags(band)
            if tags:
                target.update_tags(band, **tags)
            description = source.descriptions[band - 1]
            if description:
                target.set_band_description(band, description)
        target.colorinterp = source.colorinterp
        target.scales = source.scales
        target.offsets = source.offsets

    block_size = int(source.block_shapes[0][1]) if source.block_shapes else 512
    block_size = max(128, block_size)
    compression = source.compression.name.upper() if source.compression else "DEFLATE"
    interleave = source.interleaving.name.upper() if source.interleaving else "PIXEL"
    try:
        raster_copy(
            temporary_path,
            output_path,
            driver="COG",
            BLOCKSIZE=block_size,
            COMPRESS=compression,
            INTERLEAVE=interleave,
            OVERVIEWS="AUTO",
            RESAMPLING="NEAREST",
            BIGTIFF="IF_SAFER",
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_geojson(
    path: Path,
    source_payload: dict[str, Any],
    features: list[dict[str, Any]],
) -> None:
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if "crs" in source_payload:
        payload["crs"] = source_payload["crs"]
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_png(path: Path, data: np.ndarray) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(
            path,
            "w",
            driver="PNG",
            width=int(data.shape[2]),
            height=int(data.shape[1]),
            count=int(data.shape[0]),
            dtype="uint8",
        ) as dataset:
            dataset.write(data.astype(np.uint8, copy=False))


def _overlay_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = _preview_rgb(image)
    edge = _mask_edge(mask)
    result = preview.copy()
    result[edge] = np.asarray([255, 255, 0], dtype=np.uint8)
    return result


def _preview_rgb(image: np.ndarray) -> np.ndarray:
    if image.shape[0] >= 3:
        channels = image[:3]
    elif image.shape[0] == 2:
        channels = np.stack([image[0], image[1], image[1]], axis=0)
    elif image.shape[0] == 1:
        channels = np.repeat(image, 3, axis=0)
    else:
        raise TrainingUIAPIError("Исходный снимок не содержит каналов.")
    stretched = [_stretch_channel(channel) for channel in channels]
    return np.stack(stretched, axis=2)


def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    image = channel.astype(np.float32, copy=False)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    nonzero = finite[finite > 0]
    values = nonzero if nonzero.size else finite
    minimum = float(np.percentile(values, 1))
    maximum = float(np.percentile(values, 99))
    if maximum <= minimum:
        minimum = float(values.min())
        maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - minimum) / (maximum - minimum) * 255.0, 0, 255).astype(
        np.uint8
    )


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    positive = mask > 0
    if not bool(np.any(positive)):
        return np.zeros(mask.shape, dtype=bool)
    edge = np.zeros(mask.shape, dtype=bool)
    for _ in range(2):
        if not bool(np.any(positive)):
            break
        interior = _mask_interior(positive)
        edge |= positive & ~interior
        positive = interior
    return edge


def _mask_interior(positive: np.ndarray) -> np.ndarray:
    padded = np.pad(positive, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )


def _zip_tile_files(output_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_root.iterdir(), key=lambda item: item.name.casefold()):
            if path.is_file() and path.suffix.lower() in {".tif", ".geojson", ".png"}:
                archive.write(path, path.name)


def _write_manifest(
    path: Path,
    *,
    info: MarkupExportInfo,
    archive_filename: str,
    preview_files: dict[int, str],
) -> None:
    payload = {
        "info": info.model_dump(mode="json"),
        "archive_filename": archive_filename,
        "preview_files": {str(index): name for index, name in preview_files.items()},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_name(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._-")
    return (normalized or fallback)[:120]


def _safe_child(root: Path, name: str) -> Path:
    root_resolved = root.resolve()
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise MarkupExportUnavailable(name) from exc
    return candidate


def _export_root(config: TrainingUIAPIConfig) -> Path:
    return Path(config.scratch_root) / EXPORT_ROOT_NAME


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


__all__ = [
    "MarkupExportArtifact",
    "MarkupExportUnavailable",
    "build_markup_export",
    "cleanup_expired_markup_exports",
    "load_markup_export",
]

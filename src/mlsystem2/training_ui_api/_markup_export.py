"""Самостоятельное формирование временного набора тестовой разметки."""

from __future__ import annotations

import json
import re
import shutil
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
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
from shapely.validation import make_valid
from sqlalchemy.orm import Session

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_dataset
from ._datasets import find_dataset, imagery_images_dir, resolve_scenes_file_images
from ._raster_index import load_raster_index
from .contracts import (
    DatasetInfo,
    ImageryType,
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
_MILP_MAXIMUM_TIME_LIMIT_SECONDS = 60.0


class MarkupExportUnavailable(FileNotFoundError):
    """Временный экспорт не найден или уже просрочен."""


class _MilpTimeLimitError(TrainingUIAPIError):
    """Оптимизация не завершилась за допустимое время."""


@dataclass(frozen=True)
class MarkupExportArtifact:
    info: MarkupExportInfo
    archive_path: Path
    archive_filename: str
    preview_paths: dict[int, Path]


@dataclass(frozen=True)
class SceneListExportArtifact:
    filename: str
    content: bytes
    footprints_filename: str
    footprints_content: bytes
    archive_filename: str
    archive_content: bytes
    scene_count: int


@dataclass(frozen=True)
class _SceneListMatch:
    scene_id: str
    relative_path: Path
    footprint_wgs84: BaseGeometry


@dataclass(frozen=True)
class IntersectingImage:
    """Servernyi TIFF so stabilnym otnositelnym ID."""

    source_id: str
    path: Path


@dataclass(frozen=True)
class IntersectingImages:
    """Rezultat prostranstvennogo otbora i pokrytiya AOI."""

    images: tuple[IntersectingImage, ...]
    coverage_percent: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedMarkupTile:
    index: int
    source_name: str
    territory: str
    object_count: int
    preview_filename: str


@dataclass(frozen=True)
class GeneratedMarkupFiles:
    dataset_key: str
    dataset_name: str
    dataset_short_name: str
    dataset_version: str | None
    class_key: str
    class_name: str
    tile_width: int
    tile_height: int
    requested_object_count: int
    actual_object_count: int
    territory_count: int
    warnings: tuple[str, ...]
    tiles: tuple[GeneratedMarkupTile, ...]


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


def find_intersecting_images(
    aoi_wgs84: BaseGeometry,
    images_root: Path,
    *,
    index_path: Path | None = None,
    index_workers: int = 8,
) -> IntersectingImages:
    """Naiti originalnye TIFF, peresekayushchie AOI."""

    root = images_root.resolve()
    if not root.is_dir():
        return IntersectingImages(
            images=(),
            coverage_percent=0.0,
            warnings=("Папка исходных снимков недоступна.",),
        )
    index = load_raster_index(root, cache_path=index_path, workers=index_workers)
    matches: list[IntersectingImage] = []
    coverage_parts: list[BaseGeometry] = []
    warnings_list: list[str] = list(index.warnings)
    for entry in index.entries:
        source_id = PurePosixPath(entry.relative_path).with_suffix("").as_posix()
        resolved_path = root.joinpath(*PurePosixPath(entry.relative_path).parts)
        try:
            raster_crs = PyprojCRS.from_user_input(entry.crs)
            to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
            raster_aoi = transform_geometry(to_raster.transform, aoi_wgs84)
            raster_footprint = box(*entry.bounds)
            if (
                raster_footprint.is_empty
                or not raster_footprint.is_valid
                or raster_footprint.area <= 0
            ):
                raise ValueError("некорректный географический контур")
            if raster_aoi.intersection(raster_footprint).area <= 0:
                continue
            coverage_parts.append(box(*entry.wgs84_bounds).intersection(aoi_wgs84))
            matches.append(IntersectingImage(source_id=source_id, path=resolved_path))
        except Exception as exc:  # noqa: BLE001
            warnings_list.append(f"Пропущен нечитаемый снимок {source_id}: {exc}.")
    aoi_area = _geodesic_geometry_area(aoi_wgs84)
    covered_area = (
        _geodesic_geometry_area(unary_union(coverage_parts).intersection(aoi_wgs84))
        if coverage_parts
        else 0.0
    )
    coverage_percent = min(100.0, max(0.0, covered_area * 100.0 / aoi_area)) if aoi_area else 0.0
    if matches and coverage_percent < 99.99:
        warnings_list.append(
            f"Исходные снимки покрывают {coverage_percent:.2f}% зоны интереса."
        )
    return IntersectingImages(
        images=tuple(matches),
        coverage_percent=round(coverage_percent, 6),
        warnings=tuple(warnings_list),
    )


def _geodesic_geometry_area(geometry: BaseGeometry) -> float:
    """Poschitat ploshchad WGS84-geometrii v kvadratnyh metrah."""

    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


def build_scene_list_export(
    *,
    imagery_type: ImageryType,
    geojson_filename: str,
    geojson_bytes: bytes,
    config: TrainingUIAPIConfig,
) -> SceneListExportArtifact:
    """Сформировать TXT и GeoJSON футпринтов пересекающихся сцен."""

    download_filename = _scene_list_download_filename(geojson_filename)
    download_stem = PurePosixPath(download_filename).stem
    footprints_filename = f"{download_stem}_футпринты.geojson"
    archive_filename = f"{download_stem}.zip"
    annotations = _load_uploaded_annotations(geojson_bytes)
    images_root = imagery_images_dir(config.images_root, imagery_type.value)
    if not images_root.is_dir():
        raise TrainingUIAPIError(
            f"Папка снимков для типа «{_imagery_type_name(imagery_type)}» не найдена."
        )

    root = images_root.resolve()
    source_paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    paths_by_scene_id: dict[str, list[tuple[str, Path]]] = {}
    transformed_cache: dict[str, _TransformedAnnotations] = {}
    matches: list[_SceneListMatch] = []
    wgs84 = PyprojCRS.from_epsg(4326)
    for source_path in source_paths:
        try:
            resolved_path = source_path.resolve(strict=True)
            relative_path = resolved_path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise TrainingUIAPIError(
                f"Снимок выходит за пределы папки подготовленных изображений: {source_path.name}."
            ) from exc
        scene_id = relative_path.with_suffix("").as_posix()
        paths_by_scene_id.setdefault(scene_id.casefold(), []).append(
            (scene_id, relative_path)
        )
        try:
            with rasterio.open(resolved_path) as dataset:
                if dataset.crs is None:
                    raise TrainingUIAPIError(
                        f"У снимка отсутствует CRS: {relative_path.as_posix()}."
                    )
                raster_crs = PyprojCRS.from_user_input(dataset.crs)
                transformed = _annotations_for_crs(
                    annotations,
                    raster_crs,
                    transformed_cache,
                )
                image_footprint = box(*dataset.bounds)
                raster_footprint = Polygon(
                    (
                        dataset.transform * (0, 0),
                        dataset.transform * (dataset.width, 0),
                        dataset.transform * (dataset.width, dataset.height),
                        dataset.transform * (0, dataset.height),
                    )
                )
                if (
                    image_footprint.is_empty
                    or not image_footprint.is_valid
                    or image_footprint.area <= 0
                    or raster_footprint.is_empty
                    or not raster_footprint.is_valid
                    or raster_footprint.area <= 0
                ):
                    raise TrainingUIAPIError(
                        f"У снимка некорректный географический контур: {relative_path.as_posix()}."
                    )
                feature_indices = transformed.tree.query(
                    image_footprint,
                    predicate="intersects",
                )
                has_objects = any(
                    transformed.geometries[int(index)].intersection(image_footprint).area > 0.0
                    for index in feature_indices
                )
        except TrainingUIAPIError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TrainingUIAPIError(
                f"Не удалось прочитать снимок {relative_path.as_posix()}: {exc}"
            ) from exc
        if has_objects:
            footprint_wgs84 = _transform_between_crs(
                raster_footprint,
                raster_crs,
                wgs84,
            )
            if footprint_wgs84.is_empty:
                raise TrainingUIAPIError(
                    f"Не удалось преобразовать контур снимка в WGS84: {relative_path.as_posix()}."
                )
            matches.append(
                _SceneListMatch(
                    scene_id=scene_id,
                    relative_path=relative_path,
                    footprint_wgs84=footprint_wgs84,
                )
            )

    matched_scene_ids = {item.scene_id.casefold() for item in matches}
    ambiguous = [
        paths_by_scene_id[scene_id]
        for scene_id in sorted(matched_scene_ids)
        if len(paths_by_scene_id[scene_id]) > 1
    ]
    if ambiguous:
        details = "; ".join(
            ", ".join(path.as_posix() for _, path in items)
            for items in ambiguous
        )
        raise TrainingUIAPIError(
            "У выбранных сцен совпадают относительные пути без расширения. "
            f"Список сцен был бы неоднозначным: {details}"
        )

    sorted_matches = sorted(
        matches,
        key=lambda item: (item.scene_id.casefold(), item.scene_id),
    )
    scene_names = [item.scene_id for item in sorted_matches]
    text = "\n".join(scene_names)
    if text:
        text += "\n"
    content = text.encode("utf-8")
    footprints_content = (
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": item.scene_id,
                        "properties": {
                            "scene_id": item.scene_id,
                            "relative_path": item.relative_path.as_posix(),
                            "filename": item.relative_path.name,
                            "imagery_type": imagery_type.value,
                        },
                        "geometry": mapping(item.footprint_wgs84),
                    }
                    for item in sorted_matches
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    archive_buffer = BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(download_filename, content)
        archive.writestr(footprints_filename, footprints_content)
    return SceneListExportArtifact(
        filename=download_filename,
        content=content,
        footprints_filename=footprints_filename,
        footprints_content=footprints_content,
        archive_filename=archive_filename,
        archive_content=archive_buffer.getvalue(),
        scene_count=len(scene_names),
    )


def build_markup_export(
    request: MarkupExportRequest,
    config: TrainingUIAPIConfig,
    session: Session | None = None,
) -> MarkupExportInfo:
    cleanup_expired_markup_exports(config)
    export_id = uuid.uuid4()
    export_root = _export_root(config)
    export_root.mkdir(parents=True, exist_ok=True)
    building_root = export_root / f".building-{export_id}"
    final_root = export_root / str(export_id)
    building_root.mkdir(parents=False, exist_ok=False)
    try:
        dataset = (
            find_managed_dataset(session, config, request.dataset_key)
            if session is not None
            else find_dataset(config.mlmarkup_root, request.dataset_key, config.images_root)
        )
        generated = generate_markup_files(request, config, building_root, dataset=dataset)
        tile_infos = [
            MarkupExportTileInfo(
                index=tile.index,
                source_name=tile.source_name,
                territory=tile.territory,
                object_count=tile.object_count,
                preview_url=(
                    f"/api/v1/markup-export/{export_id}/tiles/{tile.index}/preview"
                ),
            )
            for tile in generated.tiles
        ]
        preview_files = {
            tile.index: tile.preview_filename for tile in generated.tiles
        }
        dataset_stem = _safe_name(
            generated.class_name.casefold(),
            fallback="markup",
        )
        archive_filename = f"{dataset_stem}_test_markup.zip"
        _zip_tile_files(building_root, building_root / ARCHIVE_NAME)
        expires_at = _utc_now() + EXPORT_TTL
        info = MarkupExportInfo(
            id=export_id,
            dataset_key=generated.dataset_key,
            dataset_name=generated.dataset_name,
            dataset_version=generated.dataset_version,
            tile_width=generated.tile_width,
            tile_height=generated.tile_height,
            image_count=len(tile_infos),
            requested_object_count=generated.requested_object_count,
            actual_object_count=generated.actual_object_count,
            territory_count=generated.territory_count,
            warnings=list(generated.warnings),
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


def generate_markup_files(
    request: MarkupExportRequest,
    config: TrainingUIAPIConfig,
    output_root: Path,
    *,
    dataset: DatasetInfo | None = None,
) -> GeneratedMarkupFiles:
    dataset = dataset or find_dataset(
        config.mlmarkup_root,
        request.dataset_key,
        config.images_root,
    )
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError("Для экспорта разметки нужен существующий датасет MLMarkup.")
    if dataset.diagnostics:
        raise TrainingUIAPIError("; ".join(dataset.diagnostics))
    if not dataset.scenes_file or not dataset.annotation_file:
        raise TrainingUIAPIError("У датасета должны быть TXT со сценами и один positive GeoJSON.")

    images_root = Path(dataset.images_dir or config.images_root)
    source_paths = resolve_scenes_file_images(Path(dataset.scenes_file), images_root)
    if not source_paths:
        raise TrainingUIAPIError(
            "Для датасета не найдены снимки в MLSYSTEM2_IMAGES_ROOT."
        )
    annotations = _load_annotations(Path(dataset.annotation_file))
    candidates = _build_candidates(
        source_paths=source_paths,
        images_root=images_root,
        annotations=annotations,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
        max_grid_origins=max(32, request.image_count * 8),
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
            "с полностью валидными данными."
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

    tile_files = _write_selected_tiles(
        output_root=output_root,
        selected=selected,
        annotations=annotations,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
    )
    class_name = dataset.class_name or dataset.name.split("\\", maxsplit=1)[0]
    dataset_short_name = dataset.dataset_name or "main"
    return GeneratedMarkupFiles(
        dataset_key=dataset.key,
        dataset_name=dataset.name,
        dataset_short_name=dataset_short_name,
        dataset_version=dataset.version,
        class_key=dataset.class_key or class_name,
        class_name=class_name,
        tile_width=request.tile_width,
        tile_height=request.tile_height,
        requested_object_count=request.object_count,
        actual_object_count=actual_object_count,
        territory_count=len({item.territory for item in selected}),
        warnings=tuple(warnings),
        tiles=tuple(tile_files),
    )


def generate_markup_pool_files(
    *,
    dataset_key: str,
    tile_size: int,
    min_final_image_count: int,
    max_final_image_count: int,
    min_object_count: int,
    config: TrainingUIAPIConfig,
    output_root: Path,
    dataset: DatasetInfo | None = None,
) -> GeneratedMarkupFiles:
    """Создать максимально широкий пул, содержащий допустимую итоговую разметку."""

    dataset = dataset or find_dataset(config.mlmarkup_root, dataset_key, config.images_root)
    if dataset is None or dataset.is_custom:
        raise TrainingUIAPIError(
            "Для группового создания нужен существующий датасет MLMarkup."
        )
    if dataset.diagnostics:
        raise TrainingUIAPIError("; ".join(dataset.diagnostics))
    if not dataset.scenes_file or not dataset.annotation_file:
        raise TrainingUIAPIError(
            "У датасета должны быть TXT со сценами и один positive GeoJSON."
        )

    images_root = Path(dataset.images_dir or config.images_root)
    source_paths = resolve_scenes_file_images(Path(dataset.scenes_file), images_root)
    if not source_paths:
        raise TrainingUIAPIError(
            "Для датасета не найдены TIFF в MLSYSTEM2_IMAGES_ROOT."
        )
    annotations = _load_annotations(Path(dataset.annotation_file))
    if min_final_image_count > max_final_image_count:
        raise TrainingUIAPIError(
            "Минимальное число итоговых тайлов не может быть больше максимального."
        )
    requested_pool_count = max_final_image_count * 3
    requested_pool_objects = min_object_count * 3
    candidates = _build_candidates(
        source_paths=source_paths,
        images_root=images_root,
        annotations=annotations,
        tile_width=tile_size,
        tile_height=tile_size,
        max_grid_origins=max(32, requested_pool_count * 8),
    )
    if len(candidates) < min_final_image_count:
        raise TrainingUIAPIError(
            "Недостаточно полностью валидных тайлов с объектами: "
            f"найдено {len(candidates)}, для итоговой разметки требуется минимум "
            f"{min_final_image_count}."
        )

    selected_indices: list[int] | None = None
    selected_pool_count: int | None = None
    touching_allowed = False
    maximum_pool_count = min(requested_pool_count, len(candidates))
    for allow_touching in (False, True):
        for pool_count in range(maximum_pool_count, min_final_image_count - 1, -1):
            request = MarkupExportRequest(
                dataset_key=dataset_key,
                tile_width=tile_size,
                tile_height=tile_size,
                image_count=pool_count,
                object_count=requested_pool_objects,
            )
            selected_indices = _select_candidates(
                candidates,
                request,
                allow_touching=allow_touching,
                min_final_image_count=min_final_image_count,
                max_final_image_count=min(max_final_image_count, pool_count),
                min_final_object_count=min_object_count,
            )
            if selected_indices is not None:
                selected_pool_count = pool_count
                touching_allowed = allow_touching
                break
        if selected_indices is not None:
            break

    if selected_indices is None or selected_pool_count is None:
        maximum_objects = _maximum_achievable_object_count(
            candidates,
            max_image_count=max_final_image_count,
            allow_touching=True,
        )
        raise TrainingUIAPIError(
            "Невозможно сформировать итоговую разметку без перекрытий: "
            f"требуется от {min_final_image_count} до {max_final_image_count} тайлов "
            f"и минимум {min_object_count} объектов; "
            f"валидных кандидатов {len(candidates)}, достижимый максимум объектов "
            f"при не более чем {max_final_image_count} тайлах — {maximum_objects}."
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
    actual_object_count = sum(item.object_count for item in selected)
    warnings = list(annotations.warnings)
    if selected_pool_count < requested_pool_count:
        warnings.append(
            "Расширенный пул уменьшен: "
            f"запрошено до {requested_pool_count} тайлов, сформировано {selected_pool_count}."
        )
    if actual_object_count != requested_pool_objects:
        warnings.append(
            "Целевое число объектов расширенного пула недостижимо точно: "
            f"цель {requested_pool_objects}, сформировано {actual_object_count}."
        )
    if touching_allowed:
        warnings.append(
            "Для формирования пула разрешено касание границ тайлов; перекрытий между тайлами нет."
        )

    tile_files = _write_selected_tiles(
        output_root=output_root,
        selected=selected,
        annotations=annotations,
        tile_width=tile_size,
        tile_height=tile_size,
    )
    class_name = dataset.class_name or dataset.name.split("\\", maxsplit=1)[0]
    dataset_short_name = dataset.dataset_name or "main"
    return GeneratedMarkupFiles(
        dataset_key=dataset.key,
        dataset_name=dataset.name,
        dataset_short_name=dataset_short_name,
        dataset_version=dataset.version,
        class_key=dataset.class_key or class_name,
        class_name=class_name,
        tile_width=tile_size,
        tile_height=tile_size,
        requested_object_count=min_object_count,
        actual_object_count=actual_object_count,
        territory_count=len({item.territory for item in selected}),
        warnings=tuple(warnings),
        tiles=tuple(tile_files),
    )


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
    return _annotations_from_payload(payload)


def _load_uploaded_annotations(content: bytes) -> _AnnotationSet:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TrainingUIAPIError(
            f"Не удалось прочитать загруженный GeoJSON: {exc}"
        ) from exc
    return _annotations_from_payload(payload)


def _annotations_from_payload(payload: Any) -> _AnnotationSet:
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
    max_grid_origins: int,
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
                    for column, row in _candidate_origins(
                        dataset,
                        clipped_to_image,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        max_grid_origins=max_grid_origins,
                    ):
                        key = (Path(source_path).resolve(), column, row)
                        if key in candidates:
                            continue
                        window = Window(column, row, tile_width, tile_height)
                        raster_footprint = box(*window_bounds(window, dataset.transform))
                        if clipped_to_image.intersection(raster_footprint).area <= 0.0:
                            continue
                        if not _window_is_fully_valid(dataset, window):
                            continue
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
    transformed = geometry
    if source_crs != target_crs:
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        transformed = transform_geometry(transformer.transform, geometry)
    return _polygonal_geometry(transformed) or GeometryCollection()


def _clamped_origin(value: float, maximum: int) -> int:
    return max(0, min(maximum, int(round(value))))


def _candidate_origins(
    dataset: rasterio.io.DatasetReader,
    geometry: BaseGeometry,
    *,
    tile_width: int,
    tile_height: int,
    max_grid_origins: int,
) -> list[tuple[int, int]]:
    maximum_column = dataset.width - tile_width
    maximum_row = dataset.height - tile_height
    point = geometry.representative_point()
    pixel_column, pixel_row = (~dataset.transform) * (point.x, point.y)
    origins = {
        (
            _clamped_origin(
                pixel_column - tile_width * x_fraction,
                maximum_column,
            ),
            _clamped_origin(
                pixel_row - tile_height * y_fraction,
                maximum_row,
            ),
        )
        for x_fraction, y_fraction in _ORIGIN_FRACTIONS
    }

    min_column, min_row, max_column, max_row = _geometry_pixel_bounds(
        dataset,
        geometry,
    )
    phases = ((0, 0), (tile_width // 2, tile_height // 2))
    grid_origins: set[tuple[int, int]] = set()
    for column_phase, row_phase in phases:
        columns = _grid_axis_origins(
            minimum=min_column,
            maximum=max_column,
            tile_size=tile_width,
            image_size=dataset.width,
            phase=column_phase,
        )
        rows = _grid_axis_origins(
            minimum=min_row,
            maximum=max_row,
            tile_size=tile_height,
            image_size=dataset.height,
            phase=row_phase,
        )
        for row in rows:
            for column in columns:
                window = Window(column, row, tile_width, tile_height)
                footprint = box(*window_bounds(window, dataset.transform))
                if geometry.intersection(footprint).area > 0.0:
                    grid_origins.add((column, row))
    origins.update(_spread_origins(grid_origins, max_grid_origins))
    return sorted(origins, key=lambda item: (item[1], item[0]))


def _spread_origins(
    origins: set[tuple[int, int]],
    limit: int,
) -> list[tuple[int, int]]:
    ordered = sorted(origins, key=lambda item: (item[1], item[0]))
    if len(ordered) <= limit:
        return ordered
    coordinates = np.asarray(ordered, dtype=float)
    selected = [0]
    selected_mask = np.zeros(len(ordered), dtype=bool)
    selected_mask[0] = True
    minimum_distances = np.full(len(ordered), np.inf, dtype=float)
    while len(selected) < limit:
        delta = coordinates - coordinates[selected[-1]]
        distances = np.sum(delta * delta, axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected_mask] = -1.0
        next_index = int(np.argmax(minimum_distances))
        selected.append(next_index)
        selected_mask[next_index] = True
    return [ordered[index] for index in selected]


def _geometry_pixel_bounds(
    dataset: rasterio.io.DatasetReader,
    geometry: BaseGeometry,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = geometry.bounds
    pixel_corners = [
        (~dataset.transform) * (x, y)
        for x, y in ((left, bottom), (left, top), (right, bottom), (right, top))
    ]
    columns = [float(column) for column, _ in pixel_corners]
    rows = [float(row) for _, row in pixel_corners]
    return min(columns), min(rows), max(columns), max(rows)


def _grid_axis_origins(
    *,
    minimum: float,
    maximum: float,
    tile_size: int,
    image_size: int,
    phase: int,
) -> list[int]:
    maximum_origin = image_size - tile_size
    if maximum_origin < 0 or phase > maximum_origin:
        return []
    stride = tile_size + 1
    first_step = int(np.floor((minimum - tile_size - phase) / stride))
    last_step = int(np.ceil((maximum - phase) / stride))
    return [
        origin
        for step in range(first_step, last_step + 1)
        if 0 <= (origin := phase + step * stride) <= maximum_origin
    ]


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
    min_final_image_count: int | None = None,
    max_final_image_count: int | None = None,
    min_final_object_count: int | None = None,
) -> list[int] | None:
    candidate_count = len(candidates)
    territories = sorted({item.territory for item in candidates}, key=str.casefold)
    sources = sorted({item.source_name for item in candidates}, key=str.casefold)
    territory_index = {value: index for index, value in enumerate(territories)}
    source_index = {value: index for index, value in enumerate(sources)}
    territory_offset = candidate_count
    source_offset = territory_offset + len(territories)
    deviation_index = source_offset + len(sources)
    final_offset = deviation_index + 1
    has_final_subset = (
        min_final_image_count is not None
        and max_final_image_count is not None
        and min_final_object_count is not None
    )
    variable_count = final_offset + (candidate_count if has_final_subset else 0)

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
    if has_final_subset:
        assert min_final_image_count is not None
        assert max_final_image_count is not None
        assert min_final_object_count is not None
        add_constraint(
            [(final_offset + index, 1.0) for index in range(candidate_count)],
            minimum=min_final_image_count,
            maximum=max_final_image_count,
        )
        for index in range(candidate_count):
            add_constraint(
                [(final_offset + index, 1.0), (index, -1.0)],
                maximum=0.0,
            )
        add_constraint(
            [
                (final_offset + index, float(candidates[index].object_count))
                for index in range(candidate_count)
            ],
            minimum=min_final_object_count,
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
    try:
        source_result = _run_milp(
            source_objective,
            integrality=integrality,
            bounds=bounds,
            constraints=[*common_constraints, territory_constraint],
        )
    except _MilpTimeLimitError:
        if has_final_subset:
            return _selected_indices(
                territory_result,
                candidate_count,
                request.image_count,
            )
        raise
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
    try:
        deviation_result = _run_milp(
            deviation_objective,
            integrality=integrality,
            bounds=bounds,
            constraints=[*common_constraints, territory_constraint, source_constraint],
        )
    except _MilpTimeLimitError:
        if has_final_subset:
            return _selected_indices(source_result, candidate_count, request.image_count)
        raise
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
    try:
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
    except _MilpTimeLimitError:
        if has_final_subset:
            return _selected_indices(
                deviation_result,
                candidate_count,
                request.image_count,
            )
        raise
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
    try:
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
    except _MilpTimeLimitError:
        if has_final_subset:
            return _selected_indices(
                density_result,
                candidate_count,
                request.image_count,
            )
        raise
    if stable_result is None:
        return None
    return _selected_indices(stable_result, candidate_count, request.image_count)


def _selected_indices(
    result: np.ndarray,
    candidate_count: int,
    expected_count: int,
) -> list[int] | None:
    selected = [index for index in range(candidate_count) if result[index] > 0.5]
    return selected if len(selected) == expected_count else None


def _maximum_achievable_object_count(
    candidates: list[_Candidate],
    *,
    max_image_count: int,
    allow_touching: bool,
) -> int:
    candidate_count = len(candidates)
    if candidate_count == 0 or max_image_count <= 0:
        return 0
    object_counts = np.asarray(
        [item.object_count for item in candidates],
        dtype=float,
    )
    constraints = [
        _single_constraint(
            candidate_count,
            [(index, 1.0) for index in range(candidate_count)],
            minimum=0.0,
            maximum=float(max_image_count),
        )
    ]
    conflicts = _candidate_conflicts(candidates, allow_touching=allow_touching)
    if conflicts:
        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for row_index, (left, right) in enumerate(sorted(conflicts)):
            rows.extend((row_index, row_index))
            columns.extend((left, right))
            values.extend((1.0, 1.0))
        matrix = coo_matrix(
            (values, (rows, columns)),
            shape=(len(conflicts), candidate_count),
            dtype=float,
        ).tocsr()
        constraints.append(
            LinearConstraint(
                matrix,
                np.full(len(conflicts), -np.inf),
                np.ones(len(conflicts)),
            )
        )
    result = _run_milp(
        -object_counts,
        integrality=np.ones(candidate_count, dtype=int),
        bounds=Bounds(
            np.zeros(candidate_count, dtype=float),
            np.ones(candidate_count, dtype=float),
        ),
        constraints=constraints,
        time_limit=_MILP_MAXIMUM_TIME_LIMIT_SECONDS,
    )
    if result is None:
        return 0
    return int(
        round(
            sum(
                candidates[index].object_count
                for index in range(candidate_count)
                if result[index] > 0.5
            )
        )
    )


def _candidate_conflicts(
    candidates: list[_Candidate],
    *,
    allow_touching: bool,
) -> set[tuple[int, int]]:
    conflicts: set[tuple[int, int]] = set()
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
    time_limit: float = _MILP_TIME_LIMIT_SECONDS,
) -> np.ndarray | None:
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"presolve": True, "time_limit": time_limit},
    )
    if allow_infeasible and result.status == 2:
        return None
    if result.status == 1:
        raise _MilpTimeLimitError(
            "Не удалось оптимально подобрать тестовые тайлы за допустимое время."
        )
    if not result.success or result.x is None:
        raise TrainingUIAPIError(
            "Не удалось подобрать тестовые тайлы с заданными ограничениями."
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
    selected: list[_Candidate],
    annotations: _AnnotationSet,
    tile_width: int,
    tile_height: int,
) -> list[GeneratedMarkupTile]:
    tile_infos: list[GeneratedMarkupTile] = []
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
        tile_infos.append(
            GeneratedMarkupTile(
                index=index,
                source_name=candidate.source_name,
                territory=candidate.territory,
                object_count=candidate.object_count,
                preview_filename=preview_path.name,
            )
        )
    return tile_infos


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
    if geometry.is_empty:
        return None
    repaired = make_valid(geometry) if not geometry.is_valid else geometry
    if isinstance(repaired, Polygon):
        return repaired
    if isinstance(repaired, MultiPolygon):
        return repaired
    if not isinstance(repaired, GeometryCollection):
        return None
    polygons: list[Polygon] = []
    for part in repaired.geoms:
        normalized = _polygonal_geometry(part)
        if isinstance(normalized, Polygon):
            polygons.append(normalized)
        elif isinstance(normalized, MultiPolygon):
            polygons.extend(normalized.geoms)
    if not polygons:
        return None
    merged = unary_union(polygons)
    if not merged.is_valid:
        merged = make_valid(merged)
    return merged if isinstance(merged, (Polygon, MultiPolygon)) else None


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


def _scene_list_download_filename(uploaded_name: str) -> str:
    basename = PurePosixPath(str(uploaded_name).strip().replace("\\", "/")).name
    path = PurePosixPath(basename)
    if path.suffix.casefold() != ".geojson" or not path.stem.strip():
        raise TrainingUIAPIError("Нужен файл GeoJSON с расширением .geojson.")
    return f"{path.stem}.txt"


def _imagery_type_name(imagery_type: ImageryType) -> str:
    return "Канопус" if imagery_type == ImageryType.KANOPUS else "Ортофото"


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
    "SceneListExportArtifact",
    "build_markup_export",
    "build_scene_list_export",
    "cleanup_expired_markup_exports",
    "generate_markup_files",
    "generate_markup_pool_files",
    "load_markup_export",
]

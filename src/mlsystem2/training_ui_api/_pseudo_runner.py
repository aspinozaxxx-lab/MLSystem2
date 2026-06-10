"""Запуск псевдоразметки для training UI API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Transformer
from rasterio import features as rasterio_features
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid as shapely_make_valid
import yaml

from mlsystem2.mlflow_adapter.api import download_run_artifact
from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest


@dataclass(frozen=True)
class _PostprocessProfile:
    level: int
    name: str
    mask_min_object_pixels: int | None = None
    mask_min_hole_pixels: int | None = None
    binary_closing_radius: int | None = None
    min_area_m2: float | None = None
    min_hole_area_m2: float | None = None
    simplify_m: float | None = None


@dataclass(frozen=True)
class _SceneInput:
    image_path: Path
    scene_id: str
    request_scenes: tuple[str, ...]
    request_scene_count: int


_POSTPROCESS_NONE = _PostprocessProfile(level=1, name="none")
_POSTPROCESS_DETAIL_V2 = _PostprocessProfile(
    level=2,
    name="detail_v2",
    mask_min_object_pixels=32,
    mask_min_hole_pixels=32,
    min_area_m2=3000.0,
    min_hole_area_m2=3000.0,
    simplify_m=10.0,
)
_POSTPROCESS_STRONG = _PostprocessProfile(
    level=3,
    name="strong",
    mask_min_object_pixels=64,
    mask_min_hole_pixels=64,
    binary_closing_radius=2,
    min_area_m2=10000.0,
    min_hole_area_m2=10000.0,
    simplify_m=30.0,
)
_POSTPROCESS_MERGE_POLICY = "overlap_or_touch"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-training-ui-pseudo-runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = run_pseudo_markup(payload)
    report_path = Path(payload["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"ok", "partial"} else 1


def run_pseudo_markup(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_root = Path(config["run_root"])
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    output_geojson = Path(config["output_geojson"])

    scenes = _read_scenes(Path(config["scenes_file"]))
    image_index = _image_index(Path(config["images_root"]))
    scene_inputs, missing = _collect_scene_inputs(scenes, image_index)
    postprocess_profile = _select_postprocess_profile(len(scene_inputs))
    progress_path = run_root / "progress.json"
    all_features: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    _write_pseudo_progress(
        progress_path,
        total=len(scenes),
        started=started,
        scene_reports=scene_reports,
        failures=failures,
    )
    checkpoint_path = _resolve_checkpoint(config, run_root / "checkpoint")
    threshold = float(config.get("threshold") or 0.5)
    tile_size = int(config.get("tile_size") or 768)
    stride = int(config.get("stride") or tile_size)
    batch_size = int(config.get("batch_size") or 1)
    device = str(config.get("device") or "cpu")

    try:
        torch = _torch()
        loaded = load_checkpoint(
            LoadCheckpointRequest(checkpoint_uri=str(checkpoint_path), map_location=device)
        )
        model = loaded.model.model
        model.to(torch.device(device))
        model.eval()
    except Exception as exc:  # noqa: BLE001
        failures = [{"stage": "load_checkpoint", "error": repr(exc)}]
        _write_pseudo_progress(
            progress_path,
            total=len(scenes),
            started=started,
            scene_reports=scene_reports,
            failures=failures,
        )
        _write_feature_collection(output_geojson, [])
        return _summary(
            config,
            scenes=scenes,
            status="error",
            output_geojson=output_geojson,
            started=started,
            scene_reports=[],
            failures=failures,
            missing=missing,
            feature_count=0,
            feature_count_before_merge=0,
            unique_image_count=len(scene_inputs),
            postprocess_profile=postprocess_profile,
        )

    for scene in missing:
        scene_reports.append(
            {
                "scene_id": scene,
                "number": len(scene_reports) + 1,
                "status": "missing_image",
                "feature_count": 0,
            }
        )
    _write_pseudo_progress(
        progress_path,
        total=len(scenes),
        started=started,
        scene_reports=scene_reports,
        failures=failures,
    )

    for scene_input in scene_inputs:
        scene_started = time.time()
        try:
            scene_features = _infer_scene(
                torch=torch,
                model=model,
                image_path=scene_input.image_path,
                scene=scene_input.scene_id,
                config=config,
                tile_size=tile_size,
                stride=stride,
                batch_size=batch_size,
                threshold=threshold,
                device=device,
                postprocess_profile=postprocess_profile,
            )
            all_features.extend(scene_features)
            _write_feature_collection(
                run_root / "per_scene" / _safe_dir_name(scene_input.scene_id) / "pseudo_markup.geojson",
                scene_features,
            )
            scene_reports.append(
                {
                    "scene_id": scene_input.scene_id,
                    "request_scene": scene_input.request_scenes[0],
                    "request_scenes": list(scene_input.request_scenes),
                    "request_scene_count": scene_input.request_scene_count,
                    "number": len(scene_reports) + 1,
                    "status": "ok",
                    "image": str(scene_input.image_path),
                    "feature_count": len(scene_features),
                    "elapsed_sec": round(time.time() - scene_started, 3),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "scene_id": scene_input.scene_id,
                    "image": str(scene_input.image_path),
                    "error": repr(exc),
                }
            )
            scene_reports.append(
                {
                    "scene_id": scene_input.scene_id,
                    "request_scene": scene_input.request_scenes[0],
                    "request_scenes": list(scene_input.request_scenes),
                    "request_scene_count": scene_input.request_scene_count,
                    "number": len(scene_reports) + 1,
                    "status": "failed",
                    "image": str(scene_input.image_path),
                    "feature_count": 0,
                    "error": repr(exc),
                }
            )
        _write_pseudo_progress(
            progress_path,
            total=len(scenes),
            started=started,
            scene_reports=scene_reports,
            failures=failures,
        )

    feature_count_before_merge = len(all_features)
    merged_features = _merge_overlapping_features(all_features)
    _write_feature_collection(output_geojson, merged_features)
    status = _final_status(scene_reports, failures, missing)
    return _summary(
        config,
        scenes=scenes,
        status=status,
        output_geojson=output_geojson,
        started=started,
        scene_reports=scene_reports,
        failures=failures,
        missing=missing,
        feature_count=len(merged_features),
        feature_count_before_merge=feature_count_before_merge,
        unique_image_count=len(scene_inputs),
        postprocess_profile=postprocess_profile,
    )


def _infer_scene(
    *,
    torch,
    model,
    image_path: Path,
    scene: str,
    config: dict[str, Any],
    tile_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    device: str,
    postprocess_profile: _PostprocessProfile,
) -> list[dict[str, Any]]:
    del batch_size
    with rasterio.open(image_path) as dataset:
        nodata = _resolve_nodata(dataset)
        mask = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        for window in _windows(dataset.width, dataset.height, tile_size, stride):
            image = dataset.read(
                window=window,
                boundless=True,
                fill_value=nodata,
                out_shape=(dataset.count, tile_size, tile_size),
                masked=False,
            )
            if image.shape[0] > 4:
                image = image[:4]
            if image.shape[0] < 4:
                image = np.pad(image, ((0, 4 - image.shape[0]), (0, 0), (0, 0)))
            if np.all(_nodata_pixels(image, nodata)):
                continue
            tile_mask = _predict_tile(
                torch,
                model,
                image.astype(np.float32, copy=False),
                threshold=threshold,
                device=device,
            )
            crop_h = min(tile_size, dataset.height - int(window.row_off))
            crop_w = min(tile_size, dataset.width - int(window.col_off))
            y0 = int(window.row_off)
            x0 = int(window.col_off)
            mask[y0 : y0 + crop_h, x0 : x0 + crop_w] = np.maximum(
                mask[y0 : y0 + crop_h, x0 : x0 + crop_w],
                tile_mask[:crop_h, :crop_w],
            )
        mask = _postprocess_mask(mask, postprocess_profile)
        return _features_from_mask(
            mask,
            dataset.transform,
            dataset.crs,
            dataset.res,
            scene,
            config,
            postprocess_profile=postprocess_profile,
        )


def _predict_tile(torch, model, image: np.ndarray, *, threshold: float, device: str) -> np.ndarray:
    tensor = torch.as_tensor(image[None, :, :, :], dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        output = model(tensor)
        logits = output.logits if hasattr(output, "logits") else output
        if logits.shape[-2:] != tensor.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probs = torch.sigmoid(logits[:, :1, :, :])
        predicted = probs[0, 0].detach().cpu().numpy() >= threshold
    return predicted.astype(np.uint8)


def _features_from_mask(
    mask: np.ndarray,
    transform,
    crs,
    resolution: tuple[float, float],
    scene: str,
    config: dict[str, Any],
    postprocess_profile: _PostprocessProfile = _POSTPROCESS_NONE,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_crs = str(crs) if crs is not None else None
    for geometry, value in rasterio_features.shapes(mask, mask=mask > 0, transform=transform):
        if int(value) != 1:
            continue
        if _has_vector_postprocess(postprocess_profile):
            if crs is None:
                raise RuntimeError(
                    "Для профильной постобработки нужен CRS снимка, "
                    "иначе нельзя применить пороги площади в м²."
                )
            processed_geometry = _postprocess_geometry(shape(geometry), crs, postprocess_profile)
            if processed_geometry.is_empty:
                continue
            geometry = mapping(processed_geometry)
        if source_crs:
            geometry = transform_geom(source_crs, "EPSG:4326", geometry)
        output.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": {
                    "_x_res": abs(float(resolution[0])),
                    "_y_res": abs(float(resolution[1])),
                    "_crs": source_crs,
                    "scene_id": scene,
                    "class_key": config.get("class_key"),
                    "class_name": config.get("class_name"),
                    "source_model": config.get("source_model"),
                    "source_run_id": config.get("mlflow_run_id"),
                    "source_checkpoint": config.get("checkpoint_uri"),
                    "source_threshold": config.get("threshold"),
                    "source_f1_score": config.get("checkpoint_f1_score"),
                    "source_epoch": config.get("checkpoint_epoch"),
                    "postprocess_profile": postprocess_profile.name,
                    "postprocess_level": postprocess_profile.level,
                },
            }
        )
    return output


def _merge_overlapping_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_features: list[dict[str, Any]] = []
    geometries: list[BaseGeometry] = []
    for feature in features:
        geometry_data = feature.get("geometry")
        if geometry_data is None:
            continue
        geometry = _make_valid(shape(geometry_data))
        polygon_geometry = _polygons_to_geometry(
            [polygon for polygon in _iter_polygons(geometry) if not polygon.is_empty]
        )
        if polygon_geometry.is_empty:
            continue
        indexed_features.append(feature)
        geometries.append(polygon_geometry)

    if not geometries:
        return []

    merged_geometry = _make_valid(unary_union(geometries))
    merged_polygons = [polygon for polygon in _iter_polygons(merged_geometry) if not polygon.is_empty]
    if not merged_polygons:
        return []

    tree = STRtree(geometries)
    index_by_id = {id(geometry): index for index, geometry in enumerate(geometries)}
    output: list[dict[str, Any]] = []
    for polygon in merged_polygons:
        contributor_indexes = _intersecting_geometry_indexes(
            tree,
            geometries,
            index_by_id,
            polygon,
        )
        if not contributor_indexes:
            continue
        contributors = [indexed_features[index] for index in contributor_indexes]
        output.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": _merged_feature_properties(contributors),
            }
        )
    return output


def _intersecting_geometry_indexes(
    tree: STRtree,
    geometries: list[BaseGeometry],
    index_by_id: dict[int, int],
    geometry: BaseGeometry,
) -> list[int]:
    indexes: list[int] = []
    for candidate in tree.query(geometry):
        if isinstance(candidate, (int, np.integer)):
            index = int(candidate)
        else:
            index = index_by_id.get(id(candidate))
            if index is None:
                continue
        if geometries[index].intersects(geometry):
            indexes.append(index)
    return sorted(set(indexes))


def _merged_feature_properties(features: list[dict[str, Any]]) -> dict[str, Any]:
    properties = dict(features[0].get("properties") or {})
    source_scene_ids: list[str] = []
    for feature in features:
        source_properties = feature.get("properties") or {}
        existing_scene_ids = source_properties.get("source_scene_ids")
        if isinstance(existing_scene_ids, list):
            for scene_id in existing_scene_ids:
                _append_unique_string(source_scene_ids, scene_id)
        else:
            _append_unique_string(source_scene_ids, source_properties.get("scene_id"))

    if source_scene_ids:
        properties["scene_id"] = source_scene_ids[0]
    properties["source_scene_ids"] = source_scene_ids
    properties["merged_feature_count"] = len(features)
    return properties


def _append_unique_string(values: list[str], value: object) -> None:
    if value is None:
        return
    text = str(value)
    if text not in values:
        values.append(text)


def _collect_scene_inputs(
    scenes: list[str],
    image_index: dict[str, list[Path]],
) -> tuple[list[_SceneInput], list[str]]:
    by_path: dict[Path, dict[str, int]] = {}
    missing: list[str] = []
    for scene in scenes:
        image_paths = _find_images(scene, image_index)
        if not image_paths:
            missing.append(scene)
            continue
        for image_path in image_paths:
            request_scene_counts = by_path.setdefault(image_path, {})
            request_scene_counts[scene] = request_scene_counts.get(scene, 0) + 1
    return (
        [
            _SceneInput(
                image_path=image_path,
                scene_id=image_path.stem,
                request_scenes=tuple(request_scene_counts.keys()),
                request_scene_count=sum(request_scene_counts.values()),
            )
            for image_path, request_scene_counts in by_path.items()
        ],
        missing,
    )


def _select_postprocess_profile(unique_image_count: int) -> _PostprocessProfile:
    if unique_image_count <= 5:
        return _POSTPROCESS_NONE
    if unique_image_count <= 50:
        return _POSTPROCESS_DETAIL_V2
    return _POSTPROCESS_STRONG


def _postprocess_profile_params(profile: _PostprocessProfile) -> dict[str, float | int]:
    params: dict[str, float | int] = {}
    for field in (
        "mask_min_object_pixels",
        "mask_min_hole_pixels",
        "binary_closing_radius",
        "min_area_m2",
        "min_hole_area_m2",
        "simplify_m",
    ):
        value = getattr(profile, field)
        if value is not None:
            params[field] = value
    return params


def _postprocess_mask(mask: np.ndarray, profile: _PostprocessProfile) -> np.ndarray:
    processed = mask > 0
    if profile.mask_min_object_pixels is not None:
        processed = _remove_small_mask_objects(processed, profile.mask_min_object_pixels)
    if profile.mask_min_hole_pixels is not None:
        processed = _remove_small_mask_holes(processed, profile.mask_min_hole_pixels)
    if profile.binary_closing_radius is not None:
        processed = ndimage.binary_closing(
            processed,
            structure=_disk_structure(profile.binary_closing_radius),
        )
    return processed.astype(np.uint8)


def _remove_small_mask_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=_label_structure())
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _remove_small_mask_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    labels, count = ndimage.label(~mask, structure=_label_structure())
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    fill = sizes < area_threshold
    fill[0] = False
    fill[_border_labels(labels)] = False
    result = mask.copy()
    result[fill[labels]] = True
    return result


def _label_structure() -> np.ndarray:
    return ndimage.generate_binary_structure(2, 1)


def _border_labels(labels: np.ndarray) -> np.ndarray:
    return np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )


def _disk_structure(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _has_vector_postprocess(profile: _PostprocessProfile) -> bool:
    return any(
        value is not None
        for value in (profile.min_area_m2, profile.min_hole_area_m2, profile.simplify_m)
    )


def _postprocess_geometry(
    geometry: BaseGeometry,
    crs,
    profile: _PostprocessProfile,
) -> BaseGeometry:
    metric_geometry, metric_to_source = _geometry_to_metric(geometry, crs)
    metric_geometry = _make_valid(metric_geometry)
    if profile.min_area_m2 is not None:
        metric_geometry = _filter_small_polygons(metric_geometry, profile.min_area_m2)
    if metric_geometry.is_empty:
        return metric_geometry
    if profile.min_hole_area_m2 is not None:
        metric_geometry = _remove_small_geometry_holes(metric_geometry, profile.min_hole_area_m2)
        metric_geometry = _make_valid(metric_geometry)
    if profile.simplify_m is not None and profile.simplify_m > 0:
        metric_geometry = metric_geometry.simplify(profile.simplify_m, preserve_topology=True)
        metric_geometry = _make_valid(metric_geometry)
    if metric_to_source is not None and not metric_geometry.is_empty:
        return shapely_transform(metric_to_source, metric_geometry)
    return metric_geometry


def _geometry_to_metric(
    geometry: BaseGeometry,
    crs,
) -> tuple[BaseGeometry, Callable[..., Any] | None]:
    source_crs = PyprojCRS.from_user_input(str(crs))
    if source_crs.is_projected:
        return geometry, None
    if not source_crs.is_geographic:
        raise RuntimeError(f"CRS снимка не является ни метрической, ни географической: {source_crs}.")
    representative = geometry.representative_point()
    to_lonlat = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(representative.x, representative.y)
    metric_crs = _utm_crs_for_lonlat(float(lon), float(lat))
    to_metric = Transformer.from_crs(source_crs, metric_crs, always_xy=True)
    to_source = Transformer.from_crs(metric_crs, source_crs, always_xy=True)
    return shapely_transform(to_metric.transform, geometry), to_source.transform


def _utm_crs_for_lonlat(lon: float, lat: float) -> PyprojCRS:
    if not math.isfinite(lon) or not math.isfinite(lat):
        raise RuntimeError("Не удалось определить UTM-зону для геометрии снимка.")
    zone = min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return PyprojCRS.from_epsg(epsg)


def _make_valid(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.is_empty or geometry.is_valid:
        return geometry
    return shapely_make_valid(geometry)


def _filter_small_polygons(geometry: BaseGeometry, min_area_m2: float) -> BaseGeometry:
    return _polygons_to_geometry(
        [polygon for polygon in _iter_polygons(geometry) if polygon.area >= min_area_m2]
    )


def _remove_small_geometry_holes(geometry: BaseGeometry, min_hole_area_m2: float) -> BaseGeometry:
    return _polygons_to_geometry(
        [_remove_small_polygon_holes(polygon, min_hole_area_m2) for polygon in _iter_polygons(geometry)]
    )


def _remove_small_polygon_holes(polygon: Polygon, min_hole_area_m2: float) -> Polygon:
    holes = [ring for ring in polygon.interiors if Polygon(ring).area >= min_hole_area_m2]
    return Polygon(polygon.exterior, holes)


def _iter_polygons(geometry: BaseGeometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def _polygons_to_geometry(polygons: list[Polygon]) -> BaseGeometry:
    polygons = [polygon for polygon in polygons if not polygon.is_empty]
    if not polygons:
        return GeometryCollection()
    if len(polygons) == 1:
        return polygons[0]
    return MultiPolygon(polygons)


def _resolve_checkpoint(config: dict[str, Any], dst_dir: Path) -> Path:
    local = config.get("local_checkpoint_path")
    if local and Path(str(local)).is_file():
        return Path(str(local))
    run_id = config.get("mlflow_run_id")
    artifact_path = config.get("checkpoint_artifact_path") or "checkpoints/best.pt"
    if run_id:
        downloaded = download_run_artifact(
            tracking_uri=str(config["mlflow_tracking_uri"]),
            run_id=str(run_id),
            artifact_path=str(artifact_path),
            dst_dir=dst_dir,
        )
        return Path(downloaded.local_path)
    checkpoint_uri = str(config.get("checkpoint_uri") or "")
    if checkpoint_uri and Path(checkpoint_uri).is_file():
        return Path(checkpoint_uri)
    raise RuntimeError("Не задан локальный checkpoint или MLflow run id для скачивания best.pt.")


def _windows(width: int, height: int, tile_size: int, stride: int):
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            yield Window(x, y, tile_size, tile_size)


def _read_scenes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_index(images_root: Path) -> dict[str, list[Path]]:
    files = [*images_root.rglob("*.tif"), *images_root.rglob("*.tiff")]
    index: dict[str, list[Path]] = {}
    for path in sorted(files):
        for key in _scene_lookup_keys(path.name):
            _add_index_path(index, key, path)
        for key in _scene_lookup_keys(path.stem):
            _add_index_path(index, key, path)
        for parent in path.parents:
            if parent == images_root:
                break
            for key in _scene_lookup_keys(parent.name):
                _add_index_path(index, key, path)
            try:
                relative_parent = parent.relative_to(images_root).as_posix()
            except ValueError:
                continue
            for key in _scene_lookup_keys(relative_parent):
                _add_index_path(index, key, path)
    return index


def _add_index_path(index: dict[str, list[Path]], key: str, path: Path) -> None:
    paths = index.setdefault(key, [])
    if path not in paths:
        paths.append(path)


def _find_images(scene: str, index: dict[str, list[Path]]) -> list[Path]:
    found: list[Path] = []
    for key in _scene_lookup_keys(scene):
        for path in index.get(key, []):
            if path not in found:
                found.append(path)
    return sorted(found)


def _find_image(scene: str, index: dict[str, list[Path]]) -> Path | None:
    paths = _find_images(scene, index)
    if paths:
        return paths[0]
    return None


def _scene_lookup_keys(value: str) -> set[str]:
    raw = value.strip().replace("\\", "/")
    while raw.startswith("./"):
        raw = raw[2:]
    raw = raw.strip("/")
    if not raw:
        return set()

    path = PurePosixPath(raw)
    variants = {raw, path.name, path.stem, _strip_raster_suffix(raw), _strip_raster_suffix(path.name)}
    keys: set[str] = set()
    for variant in variants:
        if not variant:
            continue
        keys.add(variant.lower())
        if variant.endswith("_cog"):
            keys.add(variant[:-4].lower())
        else:
            keys.add(f"{variant}_cog".lower())
    return keys


def _strip_raster_suffix(value: str) -> str:
    lowered = value.lower()
    for suffix in (".tiff", ".tif"):
        if lowered.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _safe_dir_name(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_").replace(":", "_")


def _final_status(
    scene_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    missing: list[str],
) -> str:
    processed = sum(1 for item in scene_reports if item.get("status") == "ok")
    if processed == 0:
        return "error"
    if failures or missing:
        return "partial"
    return "ok"


def _resolve_nodata(dataset) -> object:
    if dataset.nodata is not None:
        return dataset.nodata
    for nodata in dataset.nodatavals:
        if nodata is not None:
            return nodata
    return 0


def _nodata_pixels(image: np.ndarray, nodata: object) -> np.ndarray:
    if _is_nan(nodata):
        return np.all(np.isnan(image), axis=0)
    return np.all(image == nodata, axis=0)


def _is_nan(value: object) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _write_feature_collection(path: Path, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_pseudo_progress(
    path: Path,
    *,
    total: int,
    started: float,
    scene_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    current = min(total, _completed_request_scene_count(scene_reports))
    payload = {
        "current": current,
        "total": total,
        "processed": sum(1 for item in scene_reports if item.get("status") == "ok"),
        "failed": len(failures),
        "missing": sum(1 for item in scene_reports if item.get("status") == "missing_image"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def _completed_request_scene_count(scene_reports: list[dict[str, Any]]) -> int:
    total = 0
    for report in scene_reports:
        count = report.get("request_scene_count")
        if count is not None:
            try:
                total += max(1, int(count))
                continue
            except (TypeError, ValueError):
                pass
        request_scenes = report.get("request_scenes")
        if isinstance(request_scenes, list):
            total += max(1, len(request_scenes))
        else:
            total += 1
    return total


def _summary(
    config: dict[str, Any],
    *,
    scenes: list[str],
    status: str,
    output_geojson: Path,
    started: float,
    scene_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    missing: list[str],
    feature_count: int,
    unique_image_count: int,
    postprocess_profile: _PostprocessProfile,
    feature_count_before_merge: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "class_key": config.get("class_key"),
        "class_name": config.get("class_name"),
        "input_scene_count": len(scenes),
        "unique_image_count": unique_image_count,
        "scene_count": len(scene_reports),
        "processed": sum(1 for item in scene_reports if item.get("status") == "ok"),
        "failed": len(failures),
        "missing_images": len(missing),
        "feature_count_before_merge": (
            feature_count if feature_count_before_merge is None else feature_count_before_merge
        ),
        "feature_count": feature_count,
        "output_geojson": str(output_geojson),
        "elapsed_sec": round(time.time() - started, 3),
        "postprocess_profile": postprocess_profile.name,
        "postprocess_level": postprocess_profile.level,
        "postprocess_params": _postprocess_profile_params(postprocess_profile),
        "postprocess_merge_overlaps": True,
        "postprocess_merge_policy": _POSTPROCESS_MERGE_POLICY,
        "source": {
            "run_id": config.get("mlflow_run_id"),
            "checkpoint": config.get("checkpoint_uri"),
            "threshold": config.get("threshold"),
            "f1_score": config.get("checkpoint_f1_score"),
            "epoch": config.get("checkpoint_epoch"),
        },
        "scenes": scene_reports,
        "failures": failures,
        "missing": missing,
    }


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для псевдоразметки нужен установленный PyTorch.") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())

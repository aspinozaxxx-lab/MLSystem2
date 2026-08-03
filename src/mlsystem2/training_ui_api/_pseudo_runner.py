"""Запуск псевдоразметки для training UI API."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import shutil
import time
import uuid
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from pyproj import Geod, Transformer
from rasterio import features as rasterio_features
from rasterio.errors import NotGeoreferencedWarning
from scipy.ndimage import label as label_components
from rasterio.warp import transform_geom
from rasterio.windows import Window, bounds as window_bounds
from scipy import ndimage
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.validation import make_valid as shapely_make_valid
import yaml

from mlsystem2.dataset_preparing.api import resolve_scene_images
from mlsystem2.dataset_preparing.contracts import SceneImageResolutionRequest
from mlsystem2.metrics.api import compute_object_f1
from mlsystem2.metrics.contracts import ObjectF1Request
from mlsystem2.mlflow_adapter.api import download_run_artifact
from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest

from ._markup_export import find_intersecting_images


PSEUDO_INFERENCE_BACKEND = "pytorch_one_off"


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
    filter_compact_min_isoperimetric_quotient: float | None = None
    filter_compact_max_bbox_ratio: float | None = None


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
    mask_min_object_pixels=48,
    mask_min_hole_pixels=48,
    min_area_m2=3000.0,
    min_hole_area_m2=5000.0,
    simplify_m=15.0,
)
_POSTPROCESS_MERGE_POLICY = "overlap_or_touch"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-training-ui-pseudo-runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    result = (
        run_test_sample_f1(payload)
        if payload.get("operation") == "test_sample_f1"
        else run_pseudo_markup(payload)
    )
    report_path = Path(payload["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] in {"ok", "partial"} else 1


def run_test_sample_f1(config: dict[str, Any]) -> dict[str, Any]:
    """Считает пиксельный и объектовый F1 на независимых TIFF-тайлах разметки."""

    started = time.time()
    run_root = Path(config["run_root"])
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    progress_path = run_root / "progress.json"
    tiles = list(config.get("tiles") or [])
    _write_test_f1_progress(progress_path, current=0, total=len(tiles), started=started)
    threshold = float(config.get("threshold"))
    inference_tile_size = int(config.get("tile_size") or 768)
    stride = int(config.get("stride") or inference_tile_size)
    device = str(config.get("device") or "cpu")
    profile = _postprocess_profile_from_config(
        _configured_postprocess_profile(config, len(tiles)),
        config.get("postprocess_config"),
    )
    counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    object_counts = {
        "object_true_positive": 0,
        "object_false_positive": 0,
        "object_false_negative": 0,
    }
    reports: list[dict[str, Any]] = []
    torch = None
    loaded = None
    model = None
    try:
        checkpoint_path = _resolve_checkpoint(config, run_root / "checkpoint")
        torch = _torch()
        loaded = load_checkpoint(
            LoadCheckpointRequest(checkpoint_uri=str(checkpoint_path), map_location=device)
        )
        model = loaded.model.model
        input_channels = _loaded_input_channels(loaded, config)
        _validate_configured_input_channels(config, input_channels)
        model.to(torch.device(device))
        model.eval()
        for number, tile in enumerate(tiles, start=1):
            tile_started = time.time()
            prediction = _infer_test_tile_mask(
                torch=torch,
                model=model,
                input_channels=input_channels,
                image_path=Path(str(tile["image_path"])),
                tile_size=inference_tile_size,
                stride=stride,
                threshold=threshold,
                device=device,
                postprocess_profile=profile,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
                with rasterio.open(Path(str(tile["mask_path"]))) as mask_dataset:
                    ground_truth = mask_dataset.read(1) > 0
            if ground_truth.shape != prediction.shape:
                raise RuntimeError(
                    f"Размер эталонной маски тайла {tile.get('index')} не совпадает с TIFF."
                )
            predicted = prediction > 0
            geojson_path = tile.get("geojson_path")
            ground_truth_instances = (
                _test_tile_instance_mask(
                    Path(str(geojson_path)),
                    Path(str(tile["image_path"])),
                    predicted.shape,
                )
                if geojson_path
                else label_components(ground_truth, structure=np.ones((3, 3), dtype=np.uint8))[0]
            )
            objects = compute_object_f1(
                ObjectF1Request(
                    y_true_instances=ground_truth_instances,
                    y_pred_mask=predicted,
                )
            )
            true_positive = int(np.count_nonzero(ground_truth & predicted))
            false_positive = int(np.count_nonzero(~ground_truth & predicted))
            false_negative = int(np.count_nonzero(ground_truth & ~predicted))
            counts["true_positive"] += true_positive
            counts["false_positive"] += false_positive
            counts["false_negative"] += false_negative
            object_counts["object_true_positive"] += objects.true_positive
            object_counts["object_false_positive"] += objects.false_positive
            object_counts["object_false_negative"] += objects.false_negative
            reports.append(
                {
                    "index": int(tile["index"]),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "object_true_positive": objects.true_positive,
                    "object_false_positive": objects.false_positive,
                    "object_false_negative": objects.false_negative,
                    "elapsed_sec": round(time.time() - tile_started, 3),
                }
            )
            _write_test_f1_progress(
                progress_path,
                current=number,
                total=len(tiles),
                started=started,
            )
    except Exception as exc:  # noqa: BLE001
        _write_test_f1_progress(
            progress_path,
            current=len(reports),
            total=len(tiles),
            started=started,
        )
        return {
            "status": "error",
            "operation": "test_sample_f1",
            "processed": len(reports),
            "total": len(tiles),
            **counts,
            **object_counts,
            "threshold": threshold,
            "error": repr(exc),
            "tiles": reports,
            "elapsed_sec": round(time.time() - started, 3),
        }
    finally:
        try:
            del model
            del loaded
        except UnboundLocalError:
            pass
        _release_cuda_cache(torch, device)

    return {
        "status": "ok" if reports else "error",
        "operation": "test_sample_f1",
        "processed": len(reports),
        "total": len(tiles),
        **counts,
        **object_counts,
        "threshold": threshold,
        "test_sample_id": config.get("test_sample_id"),
        "test_sample_revision": config.get("test_sample_revision"),
        "training_result_id": config.get("training_result_id"),
        "test_f1_evaluator_version": config.get("test_f1_evaluator_version"),
        "postprocess_profile": profile.name,
        "postprocess_level": profile.level,
        "postprocess_params": _postprocess_profile_params(profile),
        "preserve_boundary_components": True,
        "tiles": reports,
        "elapsed_sec": round(time.time() - started, 3),
    }


def _test_tile_instance_mask(
    geojson_path: Path,
    image_path: Path,
    out_shape: tuple[int, int],
) -> np.ndarray:
    try:
        payload = json.loads(geojson_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Не удалось прочитать GeoJSON тестового тайла: {geojson_path}") from exc
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise RuntimeError("GeoJSON тестового тайла должен быть FeatureCollection.")
    source_crs = _payload_crs(payload)
    with rasterio.open(image_path) as dataset:
        target_crs = PyprojCRS.from_user_input(dataset.crs) if dataset.crs is not None else source_crs
        transform = dataset.transform
    transformer = (
        Transformer.from_crs(source_crs, target_crs, always_xy=True)
        if source_crs != target_crs
        else None
    )
    geometries: list[tuple[BaseGeometry, int]] = []
    for index, feature in enumerate(features, start=1):
        geometry_payload = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry_payload, dict):
            continue
        try:
            geometry = shape(geometry_payload)
        except Exception:
            continue
        if transformer is not None:
            geometry = shapely_transform(transformer.transform, geometry)
        if not geometry.is_empty:
            geometries.append((geometry, index))
    return rasterio_features.rasterize(
        geometries,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    ).astype(np.int64, copy=False)


def _payload_crs(payload: dict[str, Any]) -> PyprojCRS:
    raw = payload.get("crs")
    if isinstance(raw, dict):
        properties = raw.get("properties")
        if isinstance(properties, dict):
            raw = properties.get("name") or properties.get("href")
        else:
            raw = raw.get("name")
    return PyprojCRS.from_user_input(raw or "EPSG:4326")


def run_pseudo_markup(config: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    run_root = Path(config["run_root"])
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    output_geojson = Path(config["output_geojson"])
    progress_path = run_root / "progress.json"
    is_aoi = config.get("operation") == "pseudolabel_aoi"
    aoi_wgs84: BaseGeometry | None = None
    source_image_ids: list[str] = []
    coverage_percent: float | None = None
    api_warnings: list[str] = []
    if is_aoi:
        _write_pseudo_progress(
            progress_path,
            total=0,
            started=started,
            scene_reports=[],
            failures=[],
            stage="selecting_images",
        )
        aoi_wgs84 = _aoi_geometry(config)
        selected = find_intersecting_images(aoi_wgs84, Path(str(config["images_root"])))
        source_image_ids = [item.source_id for item in selected.images]
        coverage_percent = selected.coverage_percent
        api_warnings = list(selected.warnings)
        scene_inputs = [
            _SceneInput(
                image_path=item.path,
                scene_id=item.source_id,
                request_scenes=(item.source_id,),
                request_scene_count=1,
            )
            for item in selected.images
        ]
        missing: list[str] = []
        input_scene_count = len(scene_inputs)
    else:
        scene_inputs, missing, input_scene_count = _resolve_scene_inputs(config)
    progress_total = len(scene_inputs) + len(missing)
    postprocess_profile = _postprocess_profile_from_config(
        _select_postprocess_profile(len(scene_inputs)),
        config.get("postprocess_config"),
    )
    all_features: list[dict[str, Any]] = []
    scene_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    progress_context = {
        "source_image_ids": source_image_ids,
        "coverage_percent": coverage_percent,
        "warnings": api_warnings,
    }
    _write_pseudo_progress(
        progress_path,
        total=progress_total,
        started=started,
        scene_reports=scene_reports,
        failures=failures,
        stage="loading_model" if scene_inputs else "selecting_images",
        **progress_context,
    )
    if is_aoi and not scene_inputs:
        metadata = _aoi_metadata(config, source_image_ids, coverage_percent, api_warnings)
        _write_feature_collection(output_geojson, [], metadata=metadata)
        summary = _summary(
            config,
            input_scene_count=0,
            status="error",
            output_geojson=output_geojson,
            started=started,
            scene_reports=[],
            failures=[],
            missing=[],
            feature_count=0,
            feature_count_before_merge=0,
            unique_image_count=0,
            postprocess_profile=postprocess_profile,
        )
        return _with_aoi_report(
            summary,
            source_image_ids,
            coverage_percent,
            api_warnings,
            error={
                "code": "SOURCE_IMAGES_NOT_FOUND",
                "message": "Для зоны интереса не найдены пересекающиеся исходные снимки.",
                "details": {},
            },
        )
    threshold = float(config.get("threshold") or 0.5)
    tile_size = int(config.get("tile_size") or 768)
    stride = int(config.get("stride") or tile_size)
    batch_size = int(config.get("batch_size") or 1)
    device = str(config.get("device") or "cpu")

    torch = None
    loaded = None
    model = None
    try:
        checkpoint_path = _resolve_checkpoint(config, run_root / "checkpoint")
        torch = _torch()
        loaded = load_checkpoint(
            LoadCheckpointRequest(checkpoint_uri=str(checkpoint_path), map_location=device)
        )
        model = loaded.model.model
        input_channels = _loaded_input_channels(loaded, config)
        _validate_configured_input_channels(config, input_channels)
        model.to(torch.device(device))
        model.eval()
    except Exception as exc:  # noqa: BLE001
        try:
            del model
            del loaded
        except UnboundLocalError:
            pass
        _release_cuda_cache(torch, device)
        failures = [{"stage": "load_checkpoint", "error": repr(exc)}]
        _write_pseudo_progress(
            progress_path,
            total=progress_total,
            started=started,
            scene_reports=scene_reports,
            failures=failures,
            stage="loading_model",
            **progress_context,
        )
        metadata = (
            _aoi_metadata(config, source_image_ids, coverage_percent, api_warnings)
            if is_aoi
            else None
        )
        _write_feature_collection(output_geojson, [], metadata=metadata)
        summary = _summary(
            config,
            input_scene_count=input_scene_count,
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
        if not is_aoi:
            return summary
        return _with_aoi_report(
            summary,
            source_image_ids,
            coverage_percent,
            api_warnings,
            error={
                "code": "MODEL_LOAD_FAILED",
                "message": "Не удалось загрузить зафиксированную модель распознавания.",
                "details": {},
            },
        )

    try:
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
            total=progress_total,
            started=started,
            scene_reports=scene_reports,
            failures=failures,
            stage="inference",
            **progress_context,
        )

        for scene_input in scene_inputs:
            scene_started = time.time()
            try:
                scene_features = _infer_scene(
                    torch=torch,
                    model=model,
                    input_channels=input_channels,
                    image_path=scene_input.image_path,
                    scene=scene_input.scene_id,
                    config=config,
                    tile_size=tile_size,
                    stride=stride,
                    batch_size=batch_size,
                    threshold=threshold,
                    device=device,
                    postprocess_profile=postprocess_profile,
                    aoi_wgs84=aoi_wgs84,
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
                total=progress_total,
                started=started,
                scene_reports=scene_reports,
                failures=failures,
                stage="inference",
                **progress_context,
            )

        feature_count_before_merge = len(all_features)
        _write_pseudo_progress(
            progress_path,
            total=progress_total,
            started=started,
            scene_reports=scene_reports,
            failures=failures,
            stage="vectorization",
            **progress_context,
        )
        merged_features = (
            _merge_connected_features(all_features)
            if is_aoi
            else _merge_overlapping_features(all_features)
        )
        merged_features = _filter_compact_features(merged_features, postprocess_profile)
        if is_aoi:
            assert aoi_wgs84 is not None
            merged_features = _finalize_aoi_features(
                merged_features,
                aoi_wgs84,
                config,
                source_image_ids,
            )
        metadata = (
            _aoi_metadata(
                config,
                source_image_ids,
                coverage_percent,
                api_warnings,
                object_count=len(merged_features),
            )
            if is_aoi
            else None
        )
        _write_feature_collection(output_geojson, merged_features, metadata=metadata)
        status = _final_status(scene_reports, failures, missing)
        summary = _summary(
            config,
            input_scene_count=input_scene_count,
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
        if not is_aoi:
            return summary
        error = None
        if status == "error":
            error = {
                "code": "INFERENCE_FAILED",
                "message": "Не удалось обработать ни одного исходного снимка.",
                "details": {},
            }
        return _with_aoi_report(
            summary,
            source_image_ids,
            coverage_percent,
            api_warnings,
            error=error,
        )
    finally:
        try:
            del model
            del loaded
        except UnboundLocalError:
            pass
        _release_cuda_cache(torch, device)


def _validate_configured_input_channels(
    config: dict[str, Any],
    checkpoint_input_channels: int,
) -> None:
    configured = config.get("input_channels")
    if configured is None:
        return
    try:
        expected = int(configured)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("input_channels в конфигурации должен быть целым числом.") from exc
    if expected != checkpoint_input_channels:
        raise RuntimeError(
            "Число входных каналов конфигурации не совпадает с checkpoint: "
            f"{expected} != {checkpoint_input_channels}."
        )


def _loaded_input_channels(loaded: object, config: dict[str, Any]) -> int:
    model_handle = getattr(loaded, "model", None)
    spec = getattr(model_handle, "spec", None)
    value = getattr(spec, "input_channels", None)
    if value is None:
        value = config.get("input_channels", 4)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Не удалось определить число входных каналов checkpoint.") from exc
    if parsed <= 0:
        raise RuntimeError("Число входных каналов checkpoint должно быть положительным.")
    return parsed


def _validate_raster_input_channels(
    dataset: rasterio.io.DatasetReader,
    image_path: Path,
    expected: int,
) -> tuple[int, ...]:
    if dataset.count == expected:
        return tuple(range(1, expected + 1))
    if expected == 3 and dataset.count == 4:
        return (1, 2, 3)
    raise RuntimeError(
        f"Снимок должен содержать {expected} каналов, "
        f"получено {dataset.count}: {image_path}"
    )


def _infer_scene(
    *,
    torch,
    model,
    input_channels: int = 4,
    image_path: Path,
    scene: str,
    config: dict[str, Any],
    tile_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    device: str,
    postprocess_profile: _PostprocessProfile,
    aoi_wgs84: BaseGeometry | None = None,
) -> list[dict[str, Any]]:
    del batch_size
    with rasterio.open(image_path) as dataset:
        input_indexes = _validate_raster_input_channels(dataset, image_path, input_channels)
        nodata = _resolve_nodata(dataset)
        raster_aoi = None
        selected_windows = _windows(dataset.width, dataset.height, tile_size, stride)
        mask_window = Window(0, 0, dataset.width, dataset.height)
        if aoi_wgs84 is not None:
            if dataset.crs is None:
                raise RuntimeError("У исходного снимка отсутствует CRS.")
            to_raster = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
            raster_aoi = shapely_transform(to_raster.transform, aoi_wgs84)
            selected_windows = [
                window
                for window in _windows(dataset.width, dataset.height, tile_size, stride)
                if box(*window_bounds(window, dataset.transform)).intersects(raster_aoi)
            ]
            if not selected_windows:
                return []
            min_col = max(0, min(int(window.col_off) for window in selected_windows))
            min_row = max(0, min(int(window.row_off) for window in selected_windows))
            max_col = min(
                dataset.width,
                max(int(window.col_off + window.width) for window in selected_windows),
            )
            max_row = min(
                dataset.height,
                max(int(window.row_off + window.height) for window in selected_windows),
            )
            mask_window = Window(min_col, min_row, max_col - min_col, max_row - min_row)
        mask = np.zeros((int(mask_window.height), int(mask_window.width)), dtype=np.uint8)
        for window in selected_windows:
            image = dataset.read(
                indexes=input_indexes,
                window=window,
                boundless=True,
                fill_value=nodata,
                out_shape=(len(input_indexes), tile_size, tile_size),
                masked=False,
            )
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
            y0 = int(window.row_off - mask_window.row_off)
            x0 = int(window.col_off - mask_window.col_off)
            mask[y0 : y0 + crop_h, x0 : x0 + crop_w] = np.maximum(
                mask[y0 : y0 + crop_h, x0 : x0 + crop_w],
                tile_mask[:crop_h, :crop_w],
            )
        mask = _postprocess_mask(mask, postprocess_profile)
        return _features_from_mask(
            mask,
            dataset.window_transform(mask_window),
            dataset.crs,
            dataset.res,
            scene,
            config,
            postprocess_profile=postprocess_profile,
        )


def _infer_test_tile_mask(
    *,
    torch,
    model,
    input_channels: int = 4,
    image_path: Path,
    tile_size: int,
    stride: int,
    threshold: float,
    device: str,
    postprocess_profile: _PostprocessProfile,
) -> np.ndarray:
    with rasterio.open(image_path) as dataset:
        input_indexes = _validate_raster_input_channels(dataset, image_path, input_channels)
        nodata = _resolve_nodata(dataset)
        mask = np.zeros((dataset.height, dataset.width), dtype=np.uint8)
        for window in _windows(dataset.width, dataset.height, tile_size, stride):
            image = dataset.read(
                indexes=input_indexes,
                window=window,
                boundless=True,
                fill_value=nodata,
                out_shape=(len(input_indexes), tile_size, tile_size),
                masked=False,
            )
            if np.all(_nodata_pixels(image, nodata)):
                continue
            predicted = _predict_tile(
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
                predicted[:crop_h, :crop_w],
            )
        mask = _postprocess_mask(
            mask,
            postprocess_profile,
            preserve_border_objects=True,
        )
        if not _has_vector_postprocess(postprocess_profile) or not np.any(mask):
            return mask
        if dataset.crs is None:
            raise RuntimeError(
                "Для профильной постобработки тестового тайла нужен CRS снимка."
            )
        geometries: list[BaseGeometry] = []
        raster_boundary = box(*dataset.bounds).boundary
        for geometry, value in rasterio_features.shapes(
            mask,
            mask=mask > 0,
            transform=dataset.transform,
        ):
            if int(value) != 1:
                continue
            source_geometry = shape(geometry)
            processed = _postprocess_geometry(
                source_geometry,
                dataset.crs,
                postprocess_profile,
                preserve_boundary_fragment=source_geometry.intersects(raster_boundary),
            )
            if not processed.is_empty:
                geometries.append(processed)
        if not geometries:
            return np.zeros_like(mask)
        return rasterio_features.rasterize(
            [(geometry, 1) for geometry in geometries],
            out_shape=mask.shape,
            transform=dataset.transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
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


def _merge_connected_features(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Obedinit obekty tolko vnutri svyaznyh grupp."""

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

    tree = STRtree(geometries)
    index_by_id = {id(geometry): index for index, geometry in enumerate(geometries)}
    unseen = set(range(len(geometries)))
    output: list[dict[str, Any]] = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        component = {first}
        pending = [first]
        while pending:
            current = pending.pop()
            neighbours = _intersecting_geometry_indexes(
                tree,
                geometries,
                index_by_id,
                geometries[current],
            )
            for neighbour in neighbours:
                if neighbour not in unseen:
                    continue
                unseen.remove(neighbour)
                component.add(neighbour)
                pending.append(neighbour)
        merged = _make_valid(unary_union([geometries[index] for index in sorted(component)]))
        properties = _merged_feature_properties(
            [indexed_features[index] for index in sorted(component)]
        )
        for polygon in _iter_polygons(merged):
            if not polygon.is_empty:
                output.append(
                    {
                        "type": "Feature",
                        "geometry": mapping(polygon),
                        "properties": dict(properties),
                    }
                )
    return output


def _finalize_aoi_features(
    features: list[dict[str, Any]],
    aoi_wgs84: BaseGeometry,
    config: dict[str, Any],
    source_image_ids: list[str],
) -> list[dict[str, Any]]:
    """Obrezat kandidaty, razdelit Polygon i prisvoit stabilnye ID."""

    clipped: list[tuple[Polygon, list[str]]] = []
    for feature in features:
        geometry_data = feature.get("geometry")
        if geometry_data is None:
            continue
        geometry = _make_valid(shape(geometry_data).intersection(aoi_wgs84))
        properties = feature.get("properties") or {}
        feature_sources = properties.get("source_scene_ids")
        if not isinstance(feature_sources, list):
            feature_sources = [properties.get("scene_id")]
        normalized_sources = [
            str(source_id)
            for source_id in feature_sources
            if source_id is not None and str(source_id) in source_image_ids
        ]
        for polygon in _iter_polygons(geometry):
            if not polygon.is_empty and polygon.area > 0:
                clipped.append((polygon, normalized_sources))
    clipped.sort(key=lambda item: item[0].wkb_hex)
    output: list[dict[str, Any]] = []
    job_id = str(config.get("job_id") or "")
    for polygon, feature_sources in clipped:
        candidate_id = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"mlsystem2:pseudolabel:{job_id}:{polygon.wkb_hex}",
        )
        output.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    "candidate_id": str(candidate_id),
                    "class_id": str(config.get("class_key") or ""),
                    "model_id": str(config.get("model_id") or ""),
                    "model_version": str(config.get("model_version") or ""),
                    "job_id": job_id,
                    "source_image_ids": feature_sources,
                    "area_m2": round(_geodesic_area_m2(polygon), 3),
                },
            }
        )
    return output


def _aoi_geometry(config: dict[str, Any]) -> BaseGeometry:
    """Prochitat zafiksirovannuyu WGS84-geometriyu runner."""

    geometry_data = config.get("aoi")
    if not isinstance(geometry_data, dict):
        raise RuntimeError("В задании отсутствует GeoJSON зоны интереса.")
    geometry = shape(geometry_data)
    if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty:
        raise RuntimeError("Зона интереса должна быть Polygon или MultiPolygon.")
    if not geometry.is_valid:
        raise RuntimeError("Геометрия зоны интереса невалидна.")
    return geometry


def _aoi_metadata(
    config: dict[str, Any],
    source_image_ids: list[str],
    coverage_percent: float | None,
    api_warnings: list[str],
    *,
    object_count: int = 0,
) -> dict[str, Any]:
    """Sobrat publichnye metadannye FeatureCollection."""

    return {
        "job_id": str(config.get("job_id") or ""),
        "class_id": str(config.get("class_key") or ""),
        "model_id": str(config.get("model_id") or ""),
        "model_version": str(config.get("model_version") or ""),
        "source_image_ids": source_image_ids,
        "coverage_percent": coverage_percent,
        "warnings": api_warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_crs": "EPSG:4326",
        "object_count": object_count,
        "aoi_area_m2": config.get("aoi_area_m2"),
    }


def _with_aoi_report(
    summary: dict[str, Any],
    source_image_ids: list[str],
    coverage_percent: float | None,
    api_warnings: list[str],
    *,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    """Dobavit publichnoe pokrytie i oshibku v otchet worker."""

    return {
        **summary,
        "source_image_ids": source_image_ids,
        "coverage_percent": coverage_percent,
        "warnings": api_warnings,
        "error": error,
    }


def _geodesic_area_m2(geometry: BaseGeometry) -> float:
    """Poschitat geodezicheskuyu ploshchad kandidata."""

    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(geometry)
    return abs(float(area))


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


def _resolve_scene_inputs(
    config: dict[str, Any],
) -> tuple[list[_SceneInput], list[str], int]:
    raw_annotation_files = config.get("annotation_files") or []
    if isinstance(raw_annotation_files, str):
        raw_annotation_files = [raw_annotation_files]
    resolution = resolve_scene_images(
        SceneImageResolutionRequest(
            images_dir=str(config["images_root"]),
            scenes_file=str(config["scenes_file"]),
            annotation_files=[str(value) for value in raw_annotation_files if value],
        )
    )
    if resolution.ambiguous_scenes:
        details = "; ".join(
            f"{scene}: {', '.join(paths)}"
            for scene, paths in resolution.ambiguous_scenes.items()
        )
        raise RuntimeError(
            "Сцены неоднозначно сопоставлены со снимками. "
            f"Укажите относительные пути или разметку для выбора: {details}"
        )
    return (
        [
            _SceneInput(
                image_path=Path(item.image_path),
                scene_id=item.scene_id,
                request_scenes=tuple(dict.fromkeys(item.request_scenes)),
                request_scene_count=len(item.request_scenes),
            )
            for item in resolution.images
        ],
        resolution.missing_scenes,
        resolution.input_scene_count,
    )


def _select_postprocess_profile(unique_image_count: int) -> _PostprocessProfile:
    if unique_image_count <= 5:
        return _POSTPROCESS_NONE
    if unique_image_count <= 50:
        return _POSTPROCESS_DETAIL_V2
    return _POSTPROCESS_STRONG


def postprocess_profile_name(unique_image_count: int) -> str:
    """Возвращает имя автоматического профиля для заданного числа снимков."""

    return _select_postprocess_profile(unique_image_count).name


def _configured_postprocess_profile(
    config: dict[str, Any],
    fallback_image_count: int,
) -> _PostprocessProfile:
    name = config.get("postprocess_profile")
    if name is None:
        return _select_postprocess_profile(fallback_image_count)
    profiles = {
        _POSTPROCESS_NONE.name: _POSTPROCESS_NONE,
        _POSTPROCESS_DETAIL_V2.name: _POSTPROCESS_DETAIL_V2,
        _POSTPROCESS_STRONG.name: _POSTPROCESS_STRONG,
    }
    try:
        return profiles[str(name)]
    except KeyError as exc:
        raise RuntimeError(f"Неизвестный профиль постобработки: {name}") from exc


def _postprocess_profile_from_config(
    base: _PostprocessProfile,
    config: object,
) -> _PostprocessProfile:
    if not isinstance(config, dict):
        return base
    updates: dict[str, object] = {}
    field_map = {
        "postprocess.mask_min_object_pixels": "mask_min_object_pixels",
        "postprocess.mask_min_hole_pixels": "mask_min_hole_pixels",
        "postprocess.binary_closing_radius": "binary_closing_radius",
        "postprocess.min_area_m2": "min_area_m2",
        "postprocess.min_hole_area_m2": "min_hole_area_m2",
        "postprocess.simplify_m": "simplify_m",
    }
    for config_key, profile_field in field_map.items():
        value = config.get(config_key)
        if value is not None:
            updates[profile_field] = value
    if bool(config.get("postprocess.filter_compact_objects.enabled")):
        min_iso = config.get("postprocess.filter_compact_objects.min_isoperimetric_quotient")
        max_ratio = config.get("postprocess.filter_compact_objects.max_bbox_ratio")
        if min_iso is not None and max_ratio is not None:
            updates["filter_compact_min_isoperimetric_quotient"] = float(min_iso)
            updates["filter_compact_max_bbox_ratio"] = float(max_ratio)
    else:
        updates["filter_compact_min_isoperimetric_quotient"] = None
        updates["filter_compact_max_bbox_ratio"] = None
    if not updates:
        return base
    return replace(base, **updates)


def _postprocess_profile_params(profile: _PostprocessProfile) -> dict[str, float | int | bool]:
    params: dict[str, float | int | bool] = {}
    for field in (
        "mask_min_object_pixels",
        "mask_min_hole_pixels",
        "binary_closing_radius",
        "min_area_m2",
        "min_hole_area_m2",
        "simplify_m",
        "filter_compact_min_isoperimetric_quotient",
        "filter_compact_max_bbox_ratio",
    ):
        value = getattr(profile, field)
        if value is not None:
            params[field] = value
    return params


def _postprocess_mask(
    mask: np.ndarray,
    profile: _PostprocessProfile,
    *,
    preserve_border_objects: bool = False,
) -> np.ndarray:
    processed = mask > 0
    if profile.mask_min_object_pixels is not None:
        processed = _remove_small_mask_objects(
            processed,
            profile.mask_min_object_pixels,
            preserve_border_objects=preserve_border_objects,
        )
    if profile.mask_min_hole_pixels is not None:
        processed = _remove_small_mask_holes(processed, profile.mask_min_hole_pixels)
    if profile.binary_closing_radius is not None:
        processed = ndimage.binary_closing(
            processed,
            structure=_disk_structure(profile.binary_closing_radius),
        )
    return processed.astype(np.uint8)


def _remove_small_mask_objects(
    mask: np.ndarray,
    min_size: int,
    *,
    preserve_border_objects: bool = False,
) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=_label_structure())
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    if preserve_border_objects:
        border_labels = _border_labels(labels)
        keep[border_labels[border_labels != 0]] = True
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
        for value in (
            profile.min_area_m2,
            profile.min_hole_area_m2,
            profile.simplify_m,
            profile.filter_compact_min_isoperimetric_quotient,
            profile.filter_compact_max_bbox_ratio,
        )
    )


def _postprocess_geometry(
    geometry: BaseGeometry,
    crs,
    profile: _PostprocessProfile,
    *,
    preserve_boundary_fragment: bool = False,
) -> BaseGeometry:
    metric_geometry, metric_to_source = _geometry_to_metric(geometry, crs)
    metric_geometry = _make_valid(metric_geometry)
    if profile.min_area_m2 is not None and not preserve_boundary_fragment:
        metric_geometry = _filter_small_polygons(metric_geometry, profile.min_area_m2)
    if metric_geometry.is_empty:
        return metric_geometry
    if profile.min_hole_area_m2 is not None:
        metric_geometry = _remove_small_geometry_holes(metric_geometry, profile.min_hole_area_m2)
        metric_geometry = _make_valid(metric_geometry)
    if profile.simplify_m is not None and profile.simplify_m > 0:
        metric_geometry = metric_geometry.simplify(profile.simplify_m, preserve_topology=True)
        metric_geometry = _make_valid(metric_geometry)
    if not preserve_boundary_fragment:
        metric_geometry = _filter_compact_geometry(metric_geometry, profile)
    if metric_to_source is not None and not metric_geometry.is_empty:
        return shapely_transform(metric_to_source, metric_geometry)
    return metric_geometry


def _filter_compact_features(
    features: list[dict[str, Any]],
    profile: _PostprocessProfile,
) -> list[dict[str, Any]]:
    if (
        profile.filter_compact_min_isoperimetric_quotient is None
        or profile.filter_compact_max_bbox_ratio is None
    ):
        return features
    output: list[dict[str, Any]] = []
    for feature in features:
        geometry_data = feature.get("geometry")
        if geometry_data is None:
            continue
        try:
            metric_geometry, metric_to_source = _geometry_to_metric(shape(geometry_data), "EPSG:4326")
            metric_geometry = _filter_compact_geometry(_make_valid(metric_geometry), profile)
        except Exception:  # noqa: BLE001
            output.append(feature)
            continue
        if metric_geometry.is_empty:
            continue
        geometry = shapely_transform(metric_to_source, metric_geometry) if metric_to_source is not None else metric_geometry
        updated = dict(feature)
        updated["geometry"] = mapping(geometry)
        output.append(updated)
    return output


def _filter_compact_geometry(
    geometry: BaseGeometry,
    profile: _PostprocessProfile,
) -> BaseGeometry:
    if (
        profile.filter_compact_min_isoperimetric_quotient is None
        or profile.filter_compact_max_bbox_ratio is None
        or geometry.is_empty
    ):
        return geometry
    return _polygons_to_geometry(
        [
            polygon
            for polygon in _iter_polygons(geometry)
            if not _is_compact_polygon(
                polygon,
                min_isoperimetric_quotient=profile.filter_compact_min_isoperimetric_quotient,
                max_bbox_ratio=profile.filter_compact_max_bbox_ratio,
            )
        ]
    )


def _is_compact_polygon(
    polygon: Polygon,
    *,
    min_isoperimetric_quotient: float,
    max_bbox_ratio: float,
) -> bool:
    return (
        _isoperimetric_quotient(polygon) >= min_isoperimetric_quotient
        and _minimum_rectangle_ratio(polygon) < max_bbox_ratio
    )


def _isoperimetric_quotient(geometry: BaseGeometry) -> float:
    if geometry.length <= 0:
        return 0.0
    return float(4.0 * math.pi * geometry.area / (geometry.length * geometry.length))


def _minimum_rectangle_ratio(geometry: BaseGeometry) -> float:
    rectangle = geometry.minimum_rotated_rectangle
    exterior = getattr(rectangle, "exterior", None)
    if exterior is None:
        return _bounds_ratio(geometry)
    coords = list(exterior.coords)
    if len(coords) < 4:
        return _bounds_ratio(geometry)
    lengths = [
        math.hypot(coords[index + 1][0] - coords[index][0], coords[index + 1][1] - coords[index][1])
        for index in range(min(4, len(coords) - 1))
    ]
    positive = [value for value in lengths if value > 0]
    if not positive:
        return 0.0
    shortest = min(positive)
    if shortest <= 0:
        return math.inf
    return max(positive) / shortest


def _bounds_ratio(geometry: BaseGeometry) -> float:
    min_x, min_y, max_x, max_y = geometry.bounds
    width = abs(max_x - min_x)
    height = abs(max_y - min_y)
    shortest = min(width, height)
    if shortest <= 0:
        return math.inf if max(width, height) > 0 else 0.0
    return max(width, height) / shortest


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


def _write_feature_collection(
    path: Path,
    features: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_pseudo_progress(
    path: Path,
    *,
    total: int,
    started: float,
    scene_reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    stage: str = "inference",
    source_image_ids: list[str] | None = None,
    coverage_percent: float | None = None,
    warnings: list[str] | None = None,
) -> None:
    current = min(total, _completed_image_count(scene_reports))
    payload = {
        "current": current,
        "total": total,
        "processed": sum(1 for item in scene_reports if item.get("status") == "ok"),
        "failed": len(failures),
        "missing": sum(1 for item in scene_reports if item.get("status") == "missing_image"),
        "elapsed_sec": round(time.time() - started, 3),
        "stage": stage,
        "source_image_ids": source_image_ids or [],
        "coverage_percent": coverage_percent,
        "warnings": warnings or [],
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def _write_test_f1_progress(
    path: Path,
    *,
    current: int,
    total: int,
    started: float,
) -> None:
    payload = {
        "current": min(max(0, current), max(0, total)),
        "total": max(0, total),
        "elapsed_sec": round(time.time() - started, 3),
    }
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)


def _completed_image_count(scene_reports: list[dict[str, Any]]) -> int:
    completed_statuses = {"ok", "failed", "missing_image"}
    return sum(1 for report in scene_reports if report.get("status") in completed_statuses)


def _summary(
    config: dict[str, Any],
    *,
    input_scene_count: int,
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
        "inference_backend": PSEUDO_INFERENCE_BACKEND,
        "triton_model": None,
        "class_key": config.get("class_key"),
        "class_name": config.get("class_name"),
        "input_scene_count": input_scene_count,
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


def _release_cuda_cache(torch: Any | None, device: str) -> None:
    if torch is None or not str(device).startswith("cuda"):
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
    except Exception:  # noqa: BLE001
        return


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Для псевдоразметки нужен установленный PyTorch.") from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())

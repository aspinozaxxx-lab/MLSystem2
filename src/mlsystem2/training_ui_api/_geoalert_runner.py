"""Оркестрация ортофото-инференса через Geoalert Workflow Engine и Triton."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
import uuid
import zipfile

import numpy as np
import rasterio
from pyproj import CRS as PyprojCRS
from rasterio import features as rasterio_features
from rasterio.warp import transform_geom
import yaml

from ._external_imagery import ExternalImageryError, prepare_external_imagery
from ._external_models import external_model_manifest
from ._inference_backend import GEOALERT_INFERENCE_BACKEND, configured_inference_backend
from ._markup_export import find_intersecting_images
from ._model_export import (
    ModelExportArchive,
    _export_class_schema_override,
    build_external_triton_model_export_zip,
    build_geoalert_pipeline_yaml,
    build_triton_model_export_zip,
)
from ._pseudo_runner import (
    _SceneInput,
    _aoi_geometry,
    _aoi_metadata,
    _configured_postprocess_profile,
    _finalize_aoi_features,
    _model_schema_metadata,
    _postprocess_profile_from_config,
    _postprocess_profile_params,
    _resolve_checkpoint,
    _resolve_scene_inputs,
    _with_aoi_report,
    _write_feature_collection,
    _write_pseudo_progress,
    run_test_sample_f1,
)


@dataclass(frozen=True)
class _PreparedScenes:
    scenes: list[_SceneInput]
    missing: list[str]
    input_scene_count: int
    is_aoi: bool = False
    aoi_wgs84: object | None = None
    source_image_ids: list[str] | None = None
    coverage_percent: float | None = None
    warnings: list[str] | None = None
    performance: dict[str, Any] | None = None


@dataclass(frozen=True)
class _RuntimeExport:
    model_name: str
    model_dir: Path
    pipeline_path: Path
    bricks: list[str]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsystem2-training-ui-geoalert-runner")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    report_path = Path(config["report_path"])
    output_path = Path(config.get("output_geojson") or Path(config["run_root"]) / "pseudo_markup.geojson")
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop(_signum, _frame):
        raise KeyboardInterrupt("Geoalert-инференс остановлен диспетчером.")

    signal.signal(signal.SIGTERM, stop)
    try:
        result = run_geoalert(config)
    except BaseException as exc:  # noqa: BLE001
        _write_feature_collection(output_path, [])
        result = {
            "status": "error",
            "operation": config.get("operation"),
            "inference_backend": GEOALERT_INFERENCE_BACKEND,
            "processed": 0,
            "feature_count": 0,
            "error": repr(exc),
        }
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "ok" else 1


def run_geoalert(config: dict[str, Any]) -> dict[str, Any]:
    """Выполнить одно задание, не подменяя Geoalert локальным predictor."""

    if configured_inference_backend(config) != GEOALERT_INFERENCE_BACKEND:
        raise RuntimeError("Geoalert runner получил задание другого backend.")
    if str(config.get("model_imagery_type") or config.get("imagery_type") or "") != "ortho":
        raise RuntimeError("Geoalert runner Training UI разрешён только для моделей ортофото.")
    run_root = Path(config["run_root"])
    _reset_run_root(run_root)
    if config.get("operation") == "test_sample_f1":
        return _run_test_sample_f1(config, run_root)
    return _run_pseudo_markup(config, run_root)


def _run_pseudo_markup(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    started = time.time()
    output_geojson = Path(config["output_geojson"])
    progress_path = run_root / "progress.json"
    prepared = _prepare_scenes(config, run_root, progress_path, started)
    source_ids = list(prepared.source_image_ids or [])
    warnings = list(prepared.warnings or [])
    initial_reports = [
        {
            "scene_id": scene,
            "number": number,
            "status": "missing_image",
            "feature_count": 0,
        }
        for number, scene in enumerate(prepared.missing, start=1)
    ]
    _write_pseudo_progress(
        progress_path,
        total=len(prepared.scenes) + len(prepared.missing),
        started=started,
        scene_reports=initial_reports,
        failures=[],
        stage="loading_model" if prepared.scenes else "selecting_images",
        source_image_ids=source_ids,
        coverage_percent=prepared.coverage_percent,
        warnings=warnings,
    )
    if not prepared.scenes:
        metadata = (
            _aoi_metadata(config, source_ids, prepared.coverage_percent, warnings)
            if prepared.is_aoi
            else _model_schema_metadata(config)
        )
        _write_feature_collection(output_geojson, [], metadata=metadata)
        result = _pseudo_summary(
            config,
            started=started,
            output_geojson=output_geojson,
            prepared=prepared,
            reports=initial_reports,
            failures=[],
            feature_count=0,
            runtime_export=None,
            postprocess_params={},
        )
        if prepared.is_aoi:
            return _with_aoi_report(
                result,
                source_ids,
                prepared.coverage_percent,
                warnings,
                error={
                    "code": "SOURCE_IMAGES_NOT_FOUND",
                    "message": "Для зоны интереса не найдены пересекающиеся исходные снимки.",
                    "details": {},
                },
                performance=prepared.performance,
            )
        return result

    external_manifest = external_model_manifest(config)
    postprocess_config = _effective_postprocess_config(
        config,
        len(prepared.scenes),
        external=external_manifest is not None,
    )
    checkpoint = _resolve_checkpoint(config, run_root / "checkpoint")
    runtime_export = _ensure_runtime_export(
        config,
        checkpoint,
        external_manifest=external_manifest,
        postprocess_config=postprocess_config,
    )
    child_result: dict[str, Any]
    try:
        _load_triton_model(config, runtime_export.model_name)
        child_result = _run_compose(
            config,
            run_root,
            runtime_export,
            prepared.scenes,
            initial_reports=initial_reports,
            source_image_ids=source_ids,
            coverage_percent=prepared.coverage_percent,
            warnings=warnings,
        )
    finally:
        _unload_triton_model(config, runtime_export.model_name)

    reports = initial_reports + list(child_result.get("reports") or [])
    failures = list(child_result.get("failures") or [])
    features = _collect_features(config, child_result.get("reports") or [])
    feature_count_before_merge = len(features)
    if prepared.is_aoi:
        features = _finalize_aoi_features(
            features,
            prepared.aoi_wgs84,
            config,
            source_ids,
        )
    metadata = (
        _aoi_metadata(
            config,
            source_ids,
            prepared.coverage_percent,
            warnings,
            object_count=len(features),
        )
        if prepared.is_aoi
        else _model_schema_metadata(config)
    )
    _write_feature_collection(output_geojson, features, metadata=metadata)
    result = _pseudo_summary(
        config,
        started=started,
        output_geojson=output_geojson,
        prepared=prepared,
        reports=reports,
        failures=failures,
        feature_count=len(features),
        feature_count_before_merge=feature_count_before_merge,
        runtime_export=runtime_export,
        postprocess_params=postprocess_config,
    )
    if not prepared.is_aoi:
        return result
    error = None
    if failures:
        error = {
            "code": "SOURCE_IMAGE_PROCESSING_FAILED",
            "message": "Не удалось обработать все снимки зоны интереса.",
            "details": {
                "failed_source_image_ids": [str(item.get("scene_id")) for item in failures],
                "selected_image_count": len(prepared.scenes),
                "processed_image_count": sum(item.get("status") == "ok" for item in reports),
            },
        }
    return _with_aoi_report(
        result,
        source_ids,
        prepared.coverage_percent,
        warnings,
        error=error,
        performance={
            **dict(prepared.performance or {}),
            "geoalert_sec": float(child_result.get("elapsed_sec") or 0.0),
        },
    )


def _run_test_sample_f1(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    tiles = list(config.get("tiles") or [])
    scenes = [
        _SceneInput(
            image_path=Path(str(tile["image_path"])),
            scene_id=f"tile_{int(tile['index']):03d}",
            request_scenes=(f"tile_{int(tile['index']):03d}",),
            request_scene_count=1,
        )
        for tile in tiles
    ]
    external_manifest = external_model_manifest(config)
    postprocess_config = _effective_postprocess_config(
        config,
        len(scenes),
        external=external_manifest is not None,
    )
    checkpoint = _resolve_checkpoint(config, run_root / "checkpoint")
    runtime_export = _ensure_runtime_export(
        config,
        checkpoint,
        external_manifest=external_manifest,
        postprocess_config=postprocess_config,
    )
    try:
        _load_triton_model(config, runtime_export.model_name)
        child_result = _run_compose(
            config,
            run_root,
            runtime_export,
            scenes,
            initial_reports=[],
            source_image_ids=[],
            coverage_percent=None,
            warnings=[],
        )
    finally:
        _unload_triton_model(config, runtime_export.model_name)
    reports = list(child_result.get("reports") or [])
    if len(reports) != len(tiles) or any(item.get("status") != "ok" for item in reports):
        return {
            "status": "error",
            "operation": "test_sample_f1",
            "inference_backend": GEOALERT_INFERENCE_BACKEND,
            "triton_model": runtime_export.model_name,
            "geoalert_bricks": runtime_export.bricks,
            "processed": sum(item.get("status") == "ok" for item in reports),
            "total": len(tiles),
            "error": "Geoalert не обработал все тестовые тайлы.",
            "failures": list(child_result.get("failures") or []),
        }

    prediction_root = run_root / "test_predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    prepared_tiles: list[dict[str, Any]] = []
    for tile, report in zip(tiles, reports, strict=True):
        prediction, instances = _rasterize_scene_prediction(config, tile, report)
        prediction_path = prediction_root / f"tile_{int(tile['index']):03d}.npy"
        instance_path = prediction_root / f"tile_{int(tile['index']):03d}_instances.npy"
        np.save(prediction_path, prediction)
        np.save(instance_path, instances)
        prepared_tiles.append(
            {
                **tile,
                "precomputed_prediction_path": str(prediction_path),
                "precomputed_instances_path": str(instance_path),
            }
        )
    metrics_config = {
        **config,
        "run_root": str(run_root / "metrics"),
        "tiles": prepared_tiles,
        "external_model": None,
        "postprocess_config": {},
        "postprocess_profile": "none",
    }
    result = run_test_sample_f1(metrics_config)
    result.update(
        {
            "inference_backend": GEOALERT_INFERENCE_BACKEND,
            "triton_model": runtime_export.model_name,
            "geoalert_bricks": runtime_export.bricks,
            "postprocess_profile": "geoalert_pipeline",
            "postprocess_params": postprocess_config,
        }
    )
    metrics_progress = Path(metrics_config["run_root"]) / "progress.json"
    if metrics_progress.is_file():
        shutil.copy2(metrics_progress, run_root / "progress.json")
    return result


def _prepare_scenes(
    config: dict[str, Any],
    run_root: Path,
    progress_path: Path,
    started: float,
) -> _PreparedScenes:
    if config.get("operation") != "pseudolabel_aoi":
        scenes, missing, input_count = _resolve_scene_inputs(config)
        return _PreparedScenes(scenes=scenes, missing=missing, input_scene_count=input_count)

    aoi_wgs84 = _aoi_geometry(config)
    performance: dict[str, Any] = {"image_search_sec": 0.0, "download_sec": 0.0}
    warnings: list[str] = []
    coverage_from_provider: float | None = None
    if str(config.get("source_kind") or "local") != "local":
        def external_progress(stage: str) -> None:
            _write_pseudo_progress(
                progress_path,
                total=0,
                started=started,
                scene_reports=[],
                failures=[],
                stage=stage,
            )

        try:
            external = prepare_external_imagery(
                config,
                aoi_wgs84,
                run_root,
                progress=external_progress,
            )
        except ExternalImageryError:
            raise
        config["images_root"] = str(external.images_root)
        config["source_attribution"] = external.attribution
        config["source_license_url"] = external.license_url
        performance["download_sec"] = external.download_sec
        coverage_from_provider = external.coverage_percent
        warnings.extend(external.warnings)
    search_started = time.perf_counter()
    selected = find_intersecting_images(
        aoi_wgs84,
        Path(str(config["images_root"])),
        index_path=(Path(str(config["raster_index_path"])) if config.get("raster_index_path") else None),
        index_workers=int(config.get("image_scan_workers") or 8),
    )
    performance["image_search_sec"] = time.perf_counter() - search_started
    coverage = (
        min(selected.coverage_percent, coverage_from_provider)
        if coverage_from_provider is not None
        else selected.coverage_percent
    )
    warnings.extend(selected.warnings)
    scenes = [
        _SceneInput(
            image_path=item.path,
            scene_id=item.source_id,
            request_scenes=(item.source_id,),
            request_scene_count=1,
        )
        for item in selected.images
    ]
    return _PreparedScenes(
        scenes=scenes,
        missing=[],
        input_scene_count=len(scenes),
        is_aoi=True,
        aoi_wgs84=aoi_wgs84,
        source_image_ids=[item.source_id for item in selected.images],
        coverage_percent=coverage,
        warnings=warnings,
        performance=performance,
    )


def _effective_postprocess_config(
    config: dict[str, Any],
    image_count: int,
    *,
    external: bool,
) -> dict[str, object]:
    if external:
        return {}
    profile = _postprocess_profile_from_config(
        _configured_postprocess_profile(config, image_count),
        config.get("postprocess_config"),
    )
    params = _postprocess_profile_params(profile)
    return {
        "postprocess.mask_min_object_pixels": params.get("mask_min_object_pixels"),
        "postprocess.mask_min_hole_pixels": params.get("mask_min_hole_pixels"),
        "postprocess.binary_closing_radius": params.get("binary_closing_radius"),
        "postprocess.min_area_m2": params.get("min_area_m2"),
        "postprocess.min_hole_area_m2": params.get("min_hole_area_m2"),
        "postprocess.smooth.enabled": params.get("smooth_iterations") is not None,
        "postprocess.smooth.iterations": params.get("smooth_iterations"),
        "postprocess.smooth.offset": params.get("smooth_offset"),
        "postprocess.simplify_m": params.get("simplify_m"),
        "postprocess.filter_compact_objects.enabled": (
            params.get("filter_compact_min_isoperimetric_quotient") is not None
        ),
        "postprocess.filter_compact_objects.min_isoperimetric_quotient": params.get(
            "filter_compact_min_isoperimetric_quotient"
        ),
        "postprocess.filter_compact_objects.max_bbox_ratio": params.get(
            "filter_compact_max_bbox_ratio"
        ),
    }


def _ensure_runtime_export(
    config: dict[str, Any],
    checkpoint: Path,
    *,
    external_manifest: Any,
    postprocess_config: dict[str, object],
) -> _RuntimeExport:
    checkpoint_sha = _sha256_file(checkpoint)
    model_identity = json.dumps(
        {
            "runtime_contract": 2,
            "checkpoint_sha256": checkpoint_sha,
            "tile_size": int(config.get("tile_size") or 768),
            "threshold": config.get("threshold"),
            "external": config.get("external_model"),
            "python_site_packages": config.get("geoalert_triton_python_site_packages"),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    model_digest = hashlib.sha256(model_identity).hexdigest()
    pipeline_identity = json.dumps(
        {
            "pipeline_contract": 3,
            "model_sha256": model_digest,
            "context": int(config.get("context") or 0),
            "resolution_m": config.get("resample_to_resolution_m"),
            "postprocess": postprocess_config,
            "class_schema": list(
                config.get("class_schema") or config.get("object_types") or []
            ),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    pipeline_digest = hashlib.sha256(pipeline_identity).hexdigest()
    model_name = f"mls_ortho_{model_digest[:24]}"
    repository = Path(str(config["geoalert_model_repository"]))
    pipeline_root = Path(str(config["geoalert_pipeline_root"]))
    model_dir = repository / model_name
    pipeline_path = pipeline_root / f"{model_name}_{pipeline_digest[:12]}.yaml"
    marker_path = pipeline_root / f"{model_name}_{pipeline_digest[:12]}.json"
    model_marker_path = pipeline_root / f"{model_name}.model.json"
    if (
        model_dir.joinpath("config.pbtxt").is_file()
        and model_marker_path.is_file()
    ):
        model_marker = json.loads(model_marker_path.read_text(encoding="utf-8"))
        if model_marker.get("identity_sha256") != model_digest:
            raise RuntimeError("Кеш Geoalert содержит модель с несовпадающей идентичностью.")
        if not pipeline_path.is_file() or not marker_path.is_file():
            configured_schema = config.get("class_schema") or config.get("object_types")
            class_schema = _export_class_schema_override(
                str(model_marker.get("task") or "binary"),
                list(model_marker.get("class_schema") or []),
                list(configured_schema) if configured_schema is not None else None,
            )
            pipeline = build_geoalert_pipeline_yaml(
                model_name=model_name,
                sample_size=int(model_marker.get("sample_size") or config.get("tile_size") or 768),
                input_channels=int(model_marker["input_channels"]),
                class_schema=class_schema,
                context=int(config.get("context") or 0),
                postprocess_config=postprocess_config,
                resolution_m=(
                    float(config["resample_to_resolution_m"])
                    if config.get("resample_to_resolution_m") is not None
                    else None
                ),
                external_manifest=external_manifest,
            )
            _install_pipeline_cache(
                pipeline_path,
                marker_path,
                pipeline.encode("utf-8"),
                pipeline_digest,
                checkpoint_sha,
                model_name,
            )
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("identity_sha256") == pipeline_digest:
            return _RuntimeExport(
                model_name=model_name,
                model_dir=model_dir,
                pipeline_path=pipeline_path,
                bricks=list(marker.get("bricks") or []),
            )

    repository.mkdir(parents=True, exist_ok=True)
    pipeline_root.mkdir(parents=True, exist_ok=True)
    archive: ModelExportArchive
    if external_manifest is not None:
        archive = build_external_triton_model_export_zip(
            model_name=model_name,
            source_archive=checkpoint,
            manifest=external_manifest,
            python_site_packages=str(
                config.get("geoalert_triton_python_site_packages")
                or "/mlsystem2-venv/lib/python3.12/site-packages"
            ),
        )
    else:
        archive = build_triton_model_export_zip(
            model_name=model_name,
            checkpoint_filename=checkpoint.name if checkpoint.suffix.casefold() == ".pt" else "checkpoint.pt",
            checkpoint_bytes=checkpoint.read_bytes(),
            sample_size=int(config.get("tile_size") or 768),
            context=int(config.get("context") or 0),
            threshold=float(config["threshold"]) if config.get("threshold") is not None else None,
            instance_kind="KIND_GPU",
            postprocess_config=postprocess_config,
            resolution_m=(
                float(config["resample_to_resolution_m"])
                if config.get("resample_to_resolution_m") is not None
                else None
            ),
            class_schema_override=list(
                config.get("class_schema") or config.get("object_types") or []
            ),
        )
    try:
        with zipfile.ZipFile(archive.zip_path) as outer:
            metadata = json.loads(outer.read("export_metadata.json").decode("utf-8"))
            pipeline_bytes = outer.read(str(metadata["pipeline"]))
            nested_path = pipeline_root / f".{model_name}.{uuid.uuid4().hex}.zip"
            with outer.open(str(metadata["model_archive"])) as source, nested_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        staging = repository / f".{model_name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir(parents=True)
        try:
            _extract_model_archive(nested_path, staging, model_name)
            if model_dir.exists():
                _assert_owned_child(repository, model_dir)
                shutil.rmtree(model_dir)
            staging.replace(model_dir)
        finally:
            nested_path.unlink(missing_ok=True)
            if staging.exists():
                shutil.rmtree(staging)
        _write_json_atomic(
            model_marker_path,
            {
                "identity_sha256": model_digest,
                "checkpoint_sha256": checkpoint_sha,
                "model_name": model_name,
                "input_channels": int(metadata["input_channels"]),
                "sample_size": int(metadata.get("sample_size") or config.get("tile_size") or 768),
                "class_schema": list(metadata.get("class_schema") or []),
                "task": str(metadata.get("task") or "binary"),
            },
        )
        bricks = _pipeline_bricks(yaml.safe_load(pipeline_bytes.decode("utf-8")))
        _install_pipeline_cache(
            pipeline_path,
            marker_path,
            pipeline_bytes,
            pipeline_digest,
            checkpoint_sha,
            model_name,
        )
        return _RuntimeExport(model_name, model_dir, pipeline_path, bricks)
    finally:
        archive.cleanup()


def _run_compose(
    config: dict[str, Any],
    run_root: Path,
    runtime_export: _RuntimeExport,
    scenes: list[_SceneInput],
    *,
    initial_reports: list[dict[str, Any]],
    source_image_ids: list[str],
    coverage_percent: float | None,
    warnings: list[str],
) -> dict[str, Any]:
    result_path = run_root / "compose_result.json"
    spec_path = run_root / "compose_spec.json"
    spec = {
        "pipeline_path": str(runtime_export.pipeline_path),
        "compose_root": str(run_root / "compose"),
        "progress_path": str(run_root / "progress.json"),
        "result_path": str(result_path),
        "outputs": _pipeline_outputs(config),
        "initial_reports": initial_reports,
        "source_image_ids": source_image_ids,
        "coverage_percent": coverage_percent,
        "warnings": warnings,
        "control_dir": config.get("control_dir"),
        "triton_http_url": config.get("geoalert_triton_http_url"),
        "triton_model_name": runtime_export.model_name,
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "image_path": str(scene.image_path),
                "request_scene": scene.request_scenes[0],
                "request_scenes": list(scene.request_scenes),
                "request_scene_count": scene.request_scene_count,
            }
            for scene in scenes
        ],
    }
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    python_path = Path(str(config["geoalert_python_path"]))
    inference_root = Path(str(config["geoalert_inference_root"]))
    if not python_path.is_file() or not inference_root.is_dir():
        raise RuntimeError("Окружение Geoalert Workflow Engine не найдено на сервере.")
    env = os.environ.copy()
    python_paths = [
        inference_root / "shims",
        inference_root / "modules",
        inference_root / "modules" / "urban",
        inference_root / "modules" / "aeronet_raster",
        inference_root / "modules" / "gpdadapter",
        Path("/usr/lib/python3/dist-packages"),
    ]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(path) for path in python_paths] + ([existing] if existing else [])
    )
    command = [
        str(python_path),
        str(Path(__file__).with_name("_geoalert_compose_runner.py")),
        "--spec",
        str(spec_path),
    ]
    completed = subprocess.run(command, cwd=inference_root, env=env, check=False)
    if not result_path.is_file():
        raise RuntimeError(
            f"Geoalert Compose завершился с кодом {completed.returncode} и не создал отчёт."
        )
    return json.loads(result_path.read_text(encoding="utf-8"))


def _pipeline_outputs(config: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(config.get("external_model"), dict) or str(config.get("task") or "binary") != "multiclass":
        return [{"key": "output", "filename": "output.geojson"}]
    return [
        {"key": str(item["slug"]), "filename": f"{item['slug']}.geojson"}
        for item in config.get("object_types") or []
    ]


def _collect_features(
    config: dict[str, Any],
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    object_types = {str(item["slug"]): item for item in config.get("object_types") or []}
    output: list[dict[str, Any]] = []
    for report in reports:
        if report.get("status") != "ok":
            continue
        image_path = Path(str(report["image"]))
        with rasterio.open(image_path) as dataset:
            source_crs = str(dataset.crs) if dataset.crs is not None else None
            x_res, y_res = (abs(float(value)) for value in dataset.res)
        for key, raw_path in (report.get("outputs") or {}).items():
            payload = json.loads(Path(str(raw_path)).read_text(encoding="utf-8-sig"))
            payload_crs = _payload_crs(payload)
            object_type = object_types.get(str(key))
            for feature in payload.get("features") or []:
                geometry = feature.get("geometry")
                if not isinstance(geometry, dict):
                    continue
                if payload_crs != PyprojCRS.from_epsg(4326):
                    geometry = transform_geom(payload_crs.to_string(), "EPSG:4326", geometry)
                properties = dict(feature.get("properties") or {})
                if properties.get("confidence") is None:
                    for score_key in ("score", "_score", "confidence_score"):
                        if properties.get(score_key) is not None:
                            properties["confidence"] = properties[score_key]
                            break
                properties.update(
                    {
                        "_x_res": x_res,
                        "_y_res": y_res,
                        "_crs": source_crs,
                        "scene_id": str(report["scene_id"]),
                        "class_id": str(config.get("class_id") or config.get("class_key") or ""),
                        "class_key": config.get("class_key"),
                        "class_name": config.get("class_name"),
                        "source_model": config.get("source_model"),
                        "source_run_id": config.get("mlflow_run_id"),
                        "source_checkpoint": config.get("checkpoint_uri"),
                        "source_threshold": config.get("threshold"),
                        "source_f1_score": config.get("checkpoint_f1_score"),
                        "source_epoch": config.get("checkpoint_epoch"),
                        "postprocess_profile": "geoalert_pipeline",
                    }
                )
                if object_type is not None:
                    properties.update(
                        {
                            "object_type_id": int(object_type["id"]),
                            "object_type_slug": str(object_type["slug"]),
                            "object_type_name": str(object_type["name"]),
                            "object_type_color": str(object_type["color"]),
                        }
                    )
                output.append({"type": "Feature", "geometry": geometry, "properties": properties})
    return output


def _rasterize_scene_prediction(
    config: dict[str, Any],
    tile: dict[str, Any],
    report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    image_path = Path(str(tile["image_path"]))
    with rasterio.open(image_path) as dataset:
        shape_hw = (dataset.height, dataset.width)
        target_crs = PyprojCRS.from_user_input(dataset.crs or "EPSG:4326")
        transform = dataset.transform
    prediction = np.zeros(shape_hw, dtype=np.uint8)
    instances = np.zeros(shape_hw, dtype=np.int32)
    object_types = list(config.get("object_types") or [])
    class_by_slug = {str(item["slug"]): int(item["id"]) for item in object_types}
    ordered_keys = (
        [str(item["slug"]) for item in sorted(object_types, key=lambda value: (int(value.get("priority") or 0), -int(value["id"])))]
        if str(config.get("task") or "binary") == "multiclass"
        else ["output"]
    )
    instance_id = 0
    for key in ordered_keys:
        path = (report.get("outputs") or {}).get(key)
        if not path:
            continue
        payload = json.loads(Path(str(path)).read_text(encoding="utf-8-sig"))
        source_crs = _payload_crs(payload)
        geometries: list[dict[str, Any]] = []
        instance_shapes: list[tuple[dict[str, Any], int]] = []
        for feature in payload.get("features") or []:
            geometry = feature.get("geometry")
            if not isinstance(geometry, dict):
                continue
            if source_crs != target_crs:
                geometry = transform_geom(source_crs.to_string(), target_crs.to_string(), geometry)
            geometries.append(geometry)
            instance_id += 1
            instance_shapes.append((geometry, instance_id))
        if not geometries:
            continue
        class_id = class_by_slug.get(key, 1)
        class_mask = rasterio_features.rasterize(
            [(geometry, class_id) for geometry in geometries],
            out_shape=shape_hw,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        prediction[class_mask > 0] = class_mask[class_mask > 0]
        instance_mask = rasterio_features.rasterize(
            instance_shapes,
            out_shape=shape_hw,
            transform=transform,
            fill=0,
            dtype=np.int32,
        )
        instances[instance_mask > 0] = instance_mask[instance_mask > 0]
    return prediction, instances


def _pseudo_summary(
    config: dict[str, Any],
    *,
    started: float,
    output_geojson: Path,
    prepared: _PreparedScenes,
    reports: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    feature_count: int,
    runtime_export: _RuntimeExport | None,
    postprocess_params: dict[str, object],
    feature_count_before_merge: int | None = None,
) -> dict[str, Any]:
    processed = sum(item.get("status") == "ok" for item in reports)
    status = "error" if processed == 0 else "partial" if failures or prepared.missing else "ok"
    return {
        "status": status,
        "inference_backend": GEOALERT_INFERENCE_BACKEND,
        "triton_model": runtime_export.model_name if runtime_export is not None else None,
        "geoalert_bricks": runtime_export.bricks if runtime_export is not None else [],
        "class_key": config.get("class_key"),
        "class_name": config.get("class_name"),
        **_model_schema_metadata(config),
        "input_scene_count": prepared.input_scene_count,
        "unique_image_count": len(prepared.scenes),
        "scene_count": len(reports),
        "processed": processed,
        "failed": len(failures),
        "missing_images": len(prepared.missing),
        "feature_count_before_merge": (
            feature_count if feature_count_before_merge is None else feature_count_before_merge
        ),
        "feature_count": feature_count,
        "output_geojson": str(output_geojson),
        "elapsed_sec": round(time.time() - started, 3),
        "postprocess_profile": "geoalert_pipeline",
        "postprocess_level": None,
        "postprocess_params": postprocess_params,
        "postprocess_merge_overlaps": False,
        "postprocess_merge_policy": "geoalert_per_scene",
        "source_id": str(config.get("source_id") or ""),
        "source_name": str(config.get("source_name") or ""),
        "source_imagery_type": str(config.get("source_imagery_type") or ""),
        "model_imagery_type": str(config.get("model_imagery_type") or ""),
        "channel_mapping": str(config.get("channel_mapping") or ""),
        "target_resolution_m": config.get("target_resolution_m"),
        "source_attributions": [str(config.get("source_attribution"))] if config.get("source_attribution") else [],
        "source_license_url": config.get("source_license_url"),
        "source": {
            "run_id": config.get("mlflow_run_id"),
            "checkpoint": config.get("checkpoint_uri"),
            "threshold": config.get("threshold"),
            "f1_score": config.get("checkpoint_f1_score"),
            "epoch": config.get("checkpoint_epoch"),
        },
        "scenes": reports,
        "failures": failures,
        "missing": prepared.missing,
    }


def _load_triton_model(config: dict[str, Any], model_name: str) -> None:
    _repository_action(config, model_name, "load")
    ready_url = f"{str(config['geoalert_triton_http_url']).rstrip('/')}/v2/models/{urllib_parse.quote(model_name)}/ready"
    deadline = time.monotonic() + 120.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib_request.urlopen(ready_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib_error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Triton не перевёл модель {model_name} в READY: {last_error!r}")


def _unload_triton_model(config: dict[str, Any], model_name: str) -> None:
    try:
        _repository_action(config, model_name, "unload")
    except Exception:  # noqa: BLE001
        pass


def _repository_action(config: dict[str, Any], model_name: str, action: str) -> None:
    url = (
        f"{str(config['geoalert_triton_http_url']).rstrip('/')}/v2/repository/models/"
        f"{urllib_parse.quote(model_name)}/{action}"
    )
    request = urllib_request.Request(
        url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=120) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Triton вернул HTTP {response.status} для {action}.")
    except urllib_error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Triton не выполнил {action} модели {model_name}: {details}") from exc


def _extract_model_archive(archive_path: Path, target: Path, model_name: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            name = item.filename.replace("\\", "/").strip("/")
            if not name or item.is_dir():
                continue
            prefix = f"{model_name}/"
            if not name.startswith(prefix):
                raise RuntimeError("Архив Triton содержит файл вне корня модели.")
            relative = Path(name[len(prefix):])
            destination = (target / relative).resolve()
            if target.resolve() not in destination.parents:
                raise RuntimeError("Архив Triton содержит небезопасный путь.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)


def _pipeline_bricks(payload: object) -> list[str]:
    output: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            brick = value.get("_class")
            if isinstance(brick, str) and brick != "Compose" and brick not in output:
                output.append(brick)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return output


def _install_pipeline_cache(
    pipeline_path: Path,
    marker_path: Path,
    pipeline_bytes: bytes,
    identity_sha256: str,
    checkpoint_sha256: str,
    model_name: str,
) -> None:
    temporary_pipeline = pipeline_path.with_name(
        f".{pipeline_path.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary_pipeline.write_bytes(pipeline_bytes)
    temporary_pipeline.replace(pipeline_path)
    bricks = _pipeline_bricks(yaml.safe_load(pipeline_bytes.decode("utf-8")))
    _write_json_atomic(
        marker_path,
        {
            "identity_sha256": identity_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "model_name": model_name,
            "bricks": bricks,
        },
    )


def _payload_crs(payload: dict[str, Any]) -> PyprojCRS:
    raw = payload.get("crs")
    if isinstance(raw, dict):
        properties = raw.get("properties")
        raw = properties.get("name") if isinstance(properties, dict) else raw.get("name")
    return PyprojCRS.from_user_input(raw or "EPSG:4326")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reset_run_root(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 4:
        raise RuntimeError("Небезопасный путь рабочего каталога Geoalert.")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _assert_owned_child(parent: Path, child: Path) -> None:
    resolved_parent = parent.resolve()
    resolved_child = child.resolve()
    if resolved_child == resolved_parent or resolved_parent not in resolved_child.parents:
        raise RuntimeError("Путь модели вышел за пределы Triton repository.")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

"""Оркестрация конвейера обучения."""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata as importlib_metadata
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import (
    DatasetClassRequest,
    DatasetPreparationRequest,
    DatasetPreparationResult,
    PreparedDataset,
)
from mlsystem2.mlflow_adapter.api import (
    end_run,
    log_dataset_artifacts,
    log_dataset_preparation,
    log_pipeline_report,
    log_run_config,
    log_tile_preparation,
    log_timing_report,
    log_training_artifacts,
    log_training_epoch,
    log_training_metrics,
    start_run,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef, MLflowRunStatus, MLflowStartRunRequest
from mlsystem2.models.api import create_model, load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec
from mlsystem2.settings.api import get_settings, get_settings_path
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import (
    TileClassAnnotation,
    TileClassDefinition,
    TileDataloaderRequest,
    TileSceneSource,
    TileSplitRequest,
)
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import (
    TrainClassDefinition,
    TrainConfig,
    TrainProgressEvent,
    TrainRequest,
    TrainResult,
)

from ._timing import elapsed_since, now, timed_call
from ._next_gen import preprocessing_parameters
from .contracts import (
    ModuleTiming,
    PipelineReport,
    PipelineStatus,
    TimingReport,
    TrainPipelineError,
    TrainPipelineRequest,
    TrainPipelineResult,
)


@dataclass(frozen=True)
class _PipelineDependencies:
    get_settings: object
    get_settings_path: object
    start_run: object
    prepare_dataset: object
    create_tile_dataloader: object
    create_model: object
    load_checkpoint: object
    train_model: object
    log_dataset_preparation: object
    log_tile_preparation: object
    log_run_config: object
    log_training_epoch: object
    log_training_metrics: object
    log_training_artifacts: object
    log_timing_report: object
    log_pipeline_report: object
    end_run: object
    log_dataset_artifacts: object | None = None


def _default_dependencies() -> _PipelineDependencies:
    return _PipelineDependencies(
        get_settings=get_settings,
        get_settings_path=get_settings_path,
        start_run=start_run,
        prepare_dataset=prepare_dataset,
        create_tile_dataloader=create_tile_dataloader,
        create_model=create_model,
        load_checkpoint=load_checkpoint,
        train_model=train_model,
        log_dataset_preparation=log_dataset_preparation,
        log_tile_preparation=log_tile_preparation,
        log_run_config=log_run_config,
        log_training_epoch=log_training_epoch,
        log_training_metrics=log_training_metrics,
        log_training_artifacts=log_training_artifacts,
        log_timing_report=log_timing_report,
        log_pipeline_report=log_pipeline_report,
        end_run=end_run,
        log_dataset_artifacts=log_dataset_artifacts,
    )


def run_train_pipeline(
    request: TrainPipelineRequest,
    dependencies: _PipelineDependencies | None = None,
) -> TrainPipelineResult:
    deps = dependencies or _default_dependencies()
    total_started = now()
    timings: list[ModuleTiming] = []
    mlflow_elapsed = 0.0
    run: MLflowRunRef | None = None

    def measure_mlflow(action) -> None:
        nonlocal mlflow_elapsed
        started = now()
        action()
        mlflow_elapsed += elapsed_since(started)

    try:
        settings, timing = timed_call(
            "settings",
            lambda: deps.get_settings(),
        )
        timings.append(timing)
        settings = _expect_settings(settings)

        run = deps.start_run(_mlflow_start_request(settings, request))
        measure_mlflow(lambda: deps.log_run_config(run, deps.get_settings_path()))

        dataset_result, timing = timed_call(
            "dataset_preparing",
            lambda: deps.prepare_dataset(_dataset_request(settings)),
        )
        timings.append(timing)
        dataset_result = _expect_dataset_result(dataset_result)

        measure_mlflow(lambda: deps.log_dataset_preparation(run, dataset_result.report))

        if dataset_result.report.status == "error" or dataset_result.dataset is None:
            report = PipelineReport(
                status=PipelineStatus.FAILED,
                message="Подготовка датасета завершилась ошибкой.",
                dataset_status=dataset_result.report.status,
                errors=dataset_result.report.errors,
                warnings=[],
                artifacts={},
            )
            timing_report = _timing_report(total_started, timings, mlflow_elapsed)
            measure_mlflow(lambda: deps.log_timing_report(run, timing_report))
            measure_mlflow(lambda: deps.log_pipeline_report(run, report))
            measure_mlflow(lambda: deps.end_run(run, MLflowRunStatus.FAILED))
            return TrainPipelineResult(
                status=PipelineStatus.FAILED,
                mlflow_run=run,
                timings=_timing_report(total_started, timings, mlflow_elapsed),
                report=report,
            )

        if deps.log_dataset_artifacts is not None:
            measure_mlflow(
                lambda: deps.log_dataset_artifacts(run, _dataset_artifact_files(settings))
            )

        _validate_prepared_dataset_train_consistency(dataset_result.dataset, settings)

        loaders, timing = timed_call(
            "tile_preparation",
            lambda: (
                deps.create_tile_dataloader(
                    _tile_request(
                        dataset_result.dataset,
                        settings.train.batch_size,
                        "train",
                        _tile_split_request(settings),
                        max_batches_per_epoch=settings.train.max_train_batches_per_epoch,
                        include_object_instances=False,
                        collect_band_histogram=(
                            settings.train.pipeline_variant == "next_gen"
                            and settings.next_gen.normalization == "robust_percentile"
                        ),
                    )
                ),
                deps.create_tile_dataloader(
                    _tile_request(
                        dataset_result.dataset,
                        settings.train.batch_size,
                        "val",
                        _tile_split_request(settings),
                        max_batches_per_epoch=settings.train.max_val_batches_per_epoch,
                        include_object_instances=settings.train.task == "binary",
                    )
                ),
            ),
        )
        timings.append(timing)
        train_loader, val_loader = loaders
        train_loader = _CountingLoader(
            train_loader,
            "train",
            sampling_mode=_sampling_mode(settings, train_loader),
            positive_factor_used=(
                _weighted_factor_used(
                    train_loader,
                    "positive_factor_used",
                    settings.tile_preparation.positive_factor,
                )
                if _uses_weighted_sampler(train_loader)
                else None
            ),
            hard_negative_factor_used=(
                _weighted_factor_used(
                    train_loader,
                    "hard_negative_factor_used",
                    settings.tile_preparation.hard_negative_factor,
                )
                if _uses_weighted_sampler(train_loader)
                else None
            ),
            background_factor_used=(
                _weighted_factor_used(
                    train_loader,
                    "background_factor_used",
                    settings.tile_preparation.background_factor,
                )
                if _uses_weighted_sampler(train_loader)
                else None
            ),
        )
        val_sampling_mode = _sampling_mode(settings, val_loader)
        val_loader = _CountingLoader(
            val_loader,
            "val",
            sampling_mode=val_sampling_mode,
        )

        _seed_training(settings.tile_preparation.seed)
        model = _load_or_create_model(settings, deps, train_loader)

        def progress_sink(event: TrainProgressEvent) -> None:
            if event.metrics is not None:
                measure_mlflow(lambda: deps.log_training_epoch(run, event.metrics))

        train_result, timing = timed_call(
            "train",
            lambda: deps.train_model(
                _train_request(
                    settings,
                    model,
                    train_loader,
                    val_loader,
                    dataset=dataset_result.dataset,
                ),
                progress_sink=progress_sink,
            ),
        )
        timings.append(timing)
        train_result = _expect_train_result(train_result)

        tile_report = _tile_preparation_report(settings, train_loader, val_loader)
        if settings.train.pipeline_variant == "next_gen":
            _attach_next_gen_diagnostics(
                train_result,
                settings,
                model,
                dataset_result,
                train_loader,
                val_loader,
                tile_report,
            )
        measure_mlflow(lambda: deps.log_training_metrics(run, train_result))
        measure_mlflow(lambda: deps.log_training_artifacts(run, train_result))
        measure_mlflow(lambda: deps.log_tile_preparation(run, tile_report))

        report = PipelineReport(
            status=PipelineStatus.SUCCEEDED,
            message=(
                "Обучение остановлено пользователем; сохранён чекпойнт с лучшей F1."
                if train_result.stopped_early
                else "Конвейер обучения завершен."
            ),
            dataset_status=dataset_result.report.status,
            errors=[],
            warnings=(
                list(dataset_result.report.warnings)
                + (["Незавершённая эпоха отброшена; файл final.pt не создавался."]
                if train_result.stopped_early
                else [])
            ),
            artifacts={
                "best_checkpoint_path": train_result.best_checkpoint_path,
                "final_checkpoint_path": train_result.final_checkpoint_path,
            },
        )
        timing_report = _timing_report(total_started, timings, mlflow_elapsed)
        measure_mlflow(lambda: deps.log_timing_report(run, timing_report))
        measure_mlflow(lambda: deps.log_pipeline_report(run, report))
        measure_mlflow(lambda: deps.end_run(run, MLflowRunStatus.FINISHED))
        return TrainPipelineResult(
            status=PipelineStatus.SUCCEEDED,
            mlflow_run=run,
            timings=_timing_report(total_started, timings, mlflow_elapsed),
            report=report,
        )
    except (KeyboardInterrupt, InterruptedError) as exc:
        if run is not None:
            report = PipelineReport(
                status=PipelineStatus.FAILED,
                message="Конвейер обучения был прерван.",
                errors=[str(exc)],
            )
            try:
                timing_report = _timing_report(total_started, timings, mlflow_elapsed)
                measure_mlflow(lambda: deps.log_timing_report(run, timing_report))
                measure_mlflow(lambda: deps.log_pipeline_report(run, report))
                measure_mlflow(lambda: deps.end_run(run, MLflowRunStatus.KILLED))
            except Exception:
                pass
        raise TrainPipelineError("Конвейер обучения был прерван") from exc
    except Exception as exc:
        if run is not None:
            report = PipelineReport(
                status=PipelineStatus.FAILED,
                message="Конвейер обучения завершился невосстановимой ошибкой.",
                errors=[str(exc)],
            )
            try:
                timing_report = _timing_report(total_started, timings, mlflow_elapsed)
                measure_mlflow(lambda: deps.log_timing_report(run, timing_report))
                measure_mlflow(lambda: deps.log_pipeline_report(run, report))
                measure_mlflow(lambda: deps.end_run(run, MLflowRunStatus.FAILED))
            except Exception:
                pass
        raise TrainPipelineError("Конвейер обучения завершился ошибкой") from exc


def _mlflow_start_request(
    settings: SystemSettings,
    request: TrainPipelineRequest,
) -> MLflowStartRunRequest:
    return MLflowStartRunRequest(
        enabled=settings.mlflow.enabled,
        tracking_uri=settings.mlflow.tracking_uri,
        experiment_name=settings.mlflow.experiment_name,
        dataset=_mlflow_dataset_name(settings),
        run_name=request.run_name,
        tags={
            "pipeline": "train",
            "pipeline_variant": settings.train.pipeline_variant,
            "class": _mlflow_class_tag(settings),
            "task": settings.train.task,
            "seed": str(settings.tile_preparation.seed),
        },
    )


def _dataset_request(settings: SystemSettings) -> DatasetPreparationRequest:
    expected_band_names = (
        ["RED", "GRN", "BLU", "NIR"]
        if settings.train.pipeline_variant == "next_gen"
        else []
    )
    if settings.dataset.classes:
        return DatasetPreparationRequest(
            images_dir=settings.dataset.images_dir,
            classes=[
                DatasetClassRequest(
                    slug=item.slug,
                    name=item.name,
                    scenes_file=item.scenes_file,
                    annotation_file=item.annotation_file,
                    hard_negative_annotation_file=item.hard_negative_annotation_file,
                    priority=item.priority,
                )
                for item in settings.dataset.classes
            ],
            val_fraction=settings.dataset.val_fraction,
            expected_band_count=settings.train.input_channels,
            expected_dtype="uint8",
            expected_band_names=expected_band_names,
        )
    return DatasetPreparationRequest(
        images_dir=settings.dataset.images_dir,
        scenes_file=settings.dataset.scenes_file,
        annotation_file=settings.dataset.annotation_file,
        hard_negative_annotation_file=settings.dataset.hard_negative_annotation_file,
        annotations_dir=settings.dataset.annotations_dir,
        val_fraction=settings.dataset.val_fraction,
        expected_band_count=settings.train.input_channels,
        expected_dtype="uint8",
        expected_band_names=expected_band_names,
    )


def _dataset_artifact_files(settings: SystemSettings) -> dict[str, str]:
    if settings.dataset.classes:
        files: dict[str, str] = {}
        for item in settings.dataset.classes:
            files[f"{item.slug}_scenes{Path(item.scenes_file).suffix}"] = item.scenes_file
            files[f"{item.slug}_annotation{Path(item.annotation_file).suffix}"] = (
                item.annotation_file
            )
            if item.hard_negative_annotation_file is not None:
                files[
                    f"{item.slug}_hard_negative{Path(item.hard_negative_annotation_file).suffix}"
                ] = item.hard_negative_annotation_file
        return files
    if settings.dataset.annotations_dir is not None:
        annotations_dir = Path(settings.dataset.annotations_dir)
        files = {
            f"per_image/{path.name}": path.as_posix()
            for path in sorted(
                annotations_dir.glob("*.geojson"),
                key=lambda item: item.name.casefold(),
            )
            if path.is_file()
        }
        manifest = annotations_dir / ".mlsystem2-dataset.json"
        if manifest.is_file():
            files[f"per_image/{manifest.name}"] = manifest.as_posix()
        return files
    files = {}
    if settings.dataset.scenes_file is not None:
        files[Path(settings.dataset.scenes_file).name] = settings.dataset.scenes_file
    if settings.dataset.annotation_file is not None:
        files[Path(settings.dataset.annotation_file).name] = settings.dataset.annotation_file
    if settings.dataset.hard_negative_annotation_file is not None:
        files[Path(settings.dataset.hard_negative_annotation_file).name] = (
            settings.dataset.hard_negative_annotation_file
        )
    return files


def _tile_split_request(settings: SystemSettings) -> TileSplitRequest:
    return TileSplitRequest(
        val_fraction=settings.dataset.val_fraction,
        seed=settings.tile_preparation.seed,
        strategy=(
            "scene_fold"
            if settings.train.pipeline_variant == "next_gen"
            else "window_random"
        ),
        validation_fold=settings.next_gen.validation_fold,
        spatial_purge=settings.train.pipeline_variant == "next_gen",
    )


def _tile_request(
    dataset: PreparedDataset,
    batch_size: int,
    mode: str,
    tile_split: TileSplitRequest | None = None,
    max_batches_per_epoch: int | None = None,
    include_object_instances: bool = False,
    collect_band_histogram: bool = False,
) -> TileDataloaderRequest:
    scenes = [
        TileSceneSource(
            scene_id=item.scene_id,
            image_path=item.image_path,
            annotation_file=item.annotation_file,
            footprint_file=item.footprint_file,
        )
        for item in dataset.scenes
    ]
    if dataset.class_annotations:
        return TileDataloaderRequest(
            scenes=scenes,
            class_annotations=[
                TileClassAnnotation(
                    class_id=item.class_id,
                    slug=item.slug,
                    name=item.name,
                    annotation_file=item.annotation_file,
                    hard_negative_annotation_file=item.hard_negative_annotation_file,
                    priority=item.priority,
                )
                for item in dataset.class_annotations
            ],
            batch_size=batch_size,
            mode=mode,
            tile_split=tile_split,
            max_batches_per_epoch=max_batches_per_epoch,
            include_object_instances=include_object_instances,
            pipeline_variant=(
                "next_gen" if tile_split and tile_split.strategy == "scene_fold" else "legacy"
            ),
            collect_band_histogram=collect_band_histogram,
        )
    if dataset.classes:
        return TileDataloaderRequest(
            scenes=scenes,
            classes=[
                TileClassDefinition(
                    class_id=item.id,
                    slug=item.slug,
                    name=item.name,
                    color=item.color,
                    priority=item.priority,
                )
                for item in dataset.classes
            ],
            batch_size=batch_size,
            mode=mode,
            tile_split=tile_split,
            max_batches_per_epoch=max_batches_per_epoch,
            include_object_instances=include_object_instances,
            pipeline_variant=(
                "next_gen" if tile_split and tile_split.strategy == "scene_fold" else "legacy"
            ),
            collect_band_histogram=collect_band_histogram,
        )
    if dataset.annotation_file is None and not all(
        item.annotation_file is not None for item in dataset.scenes
    ):
        raise TrainPipelineError("PreparedDataset не содержит binary-разметку")
    return TileDataloaderRequest(
        scenes=scenes,
        annotation_file=dataset.annotation_file,
        hard_negative_annotation_file=dataset.hard_negative_annotation_file,
        batch_size=batch_size,
        mode=mode,
        tile_split=tile_split,
        max_batches_per_epoch=max_batches_per_epoch,
        include_object_instances=include_object_instances,
        pipeline_variant=(
            "next_gen" if tile_split and tile_split.strategy == "scene_fold" else "legacy"
        ),
        collect_band_histogram=collect_band_histogram,
    )


def _model_spec(settings: SystemSettings, train_loader: object | None = None) -> ModelSpec:
    parameters: dict[str, object] = {}
    if settings.train.pipeline_variant == "next_gen":
        dataset = getattr(train_loader, "dataset", None)
        histogram = getattr(dataset, "band_histogram", None)
        preprocessing = preprocessing_parameters(settings.next_gen.normalization, histogram)
        split_manifest = getattr(dataset, "tile_split_manifest", {})
        parameters = {
            "pipeline_variant": "next_gen",
            "preprocessing": preprocessing,
            "band_contract": ["RED", "GRN", "BLU", "NIR"],
            "split": split_manifest,
            "scheduler": {
                "name": "reduce_lr_on_plateau",
                "mode": "max",
                "factor": 0.5,
                "patience": 3,
                "min_lr": 1e-7,
            },
            "threshold_mode": settings.next_gen.threshold_mode,
            "threshold_policy": {
                "mode": settings.next_gen.threshold_mode,
                "configured_threshold": settings.train.threshold,
                "pixel_histogram_bins": 4096,
            },
            "pretrained_provenance": {
                "enabled": settings.train.pretrained,
                "source": (
                    "nvidia/segformer-b0-finetuned-ade-512-512"
                    if settings.train.pretrained
                    else None
                ),
                "revision": (
                    "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
                    if settings.train.pretrained
                    else None
                ),
            },
        }
    return ModelSpec(
        name=settings.train.model_name,
        input_channels=settings.train.input_channels,
        output_channels=settings.train.output_channels,
        pretrained=settings.train.pretrained,
        parameters=parameters,
    )


def _load_or_create_model(
    settings: SystemSettings,
    deps: _PipelineDependencies,
    train_loader: object | None = None,
):
    spec = _model_spec(settings, train_loader)
    if settings.train.initial_checkpoint_uri:
        loaded = deps.load_checkpoint(
            LoadCheckpointRequest(
                checkpoint_uri=settings.train.initial_checkpoint_uri,
                model_spec=(spec if settings.train.pipeline_variant == "legacy" else None),
                map_location=settings.train.device,
            )
        )
        if settings.train.pipeline_variant == "next_gen":
            loaded_spec = loaded.model.spec
            if (
                loaded_spec.name != spec.name
                or loaded_spec.input_channels != spec.input_channels
                or loaded_spec.output_channels != spec.output_channels
                or loaded_spec.parameters.get("preprocessing")
                != spec.parameters.get("preprocessing")
            ):
                raise TrainPipelineError(
                    "Параметры next_gen checkpoint не совпадают с текущей архитектурой/preprocessing"
                )
        return loaded.model
    return deps.create_model(spec)


def _attach_next_gen_diagnostics(
    result: TrainResult,
    settings: SystemSettings,
    model: object,
    dataset_result: DatasetPreparationResult,
    train_loader: object,
    val_loader: object,
    tile_report: dict[str, object],
) -> None:
    model_spec = getattr(model, "spec", None)
    model_parameters = dict(getattr(model_spec, "parameters", {}) or {})
    train_dataset = getattr(train_loader, "dataset", None)
    val_dataset = getattr(val_loader, "dataset", None)
    split_manifest = {
        "train": getattr(train_dataset, "tile_split_manifest", {}),
        "validation": getattr(val_dataset, "tile_split_manifest", {}),
    }
    validation_events = [
        item for item in result.history if item.validation_performed
    ]
    validation_by_scene = {
        "validation_fold": settings.next_gen.validation_fold,
        "events": [
            {
                "epoch": item.epoch,
                "threshold": item.val_best_threshold,
                "macro_pixel_f1": item.val_macro_pixel_f1,
                "micro_pixel_f1": item.val_micro_pixel_f1,
                "fixed_0_5_pixel_f1": item.val_fixed_0_5_pixel_f1,
                "fixed_0_5_pixel_precision": item.val_fixed_0_5_pixel_precision,
                "fixed_0_5_pixel_recall": item.val_fixed_0_5_pixel_recall,
                "object_f1": item.val_best_threshold_object_f1,
                "object_precision": item.val_best_threshold_object_precision,
                "object_recall": item.val_best_threshold_object_recall,
                "scenes": item.val_per_scene_metrics,
            }
            for item in validation_events
        ],
    }
    dataset_revision = _dataset_revision(dataset_result.dataset)
    runtime_environment = _runtime_environment(
        settings.runtime.project_root,
        dataset_revision,
    )
    runtime_environment["peak_vram_bytes"] = int(
        result.diagnostics.get("peak_vram_bytes") or 0
    )
    parameter_source = {
        "train": settings.train.model_dump(mode="json"),
        "next_gen": settings.next_gen.model_dump(mode="json"),
        "tile_preparation": settings.tile_preparation.model_dump(mode="json"),
        "dataset_revision": dataset_revision,
        "model": {
            key: value
            for key, value in model_parameters.items()
            if key != "hf_config"
        },
    }
    result.diagnostics.update(
        {
            "pipeline_variant": "next_gen",
            "validation_fold": settings.next_gen.validation_fold,
            "resolved_train_config": settings.model_dump(mode="json"),
            "split_manifest": split_manifest,
            "preprocessing": model_parameters.get("preprocessing", {}),
            "runtime_environment": runtime_environment,
            "validation_by_scene": validation_by_scene,
            "flattened_params": _flatten_mlflow_params(parameter_source),
            "tile_preparation": tile_report,
        }
    )


def _flatten_mlflow_params(
    source: dict[str, object],
    *,
    prefix: str = "",
    limit: int = 200,
) -> dict[str, object]:
    import json

    flattened: dict[str, object] = {}

    def visit(value: object, path: str) -> None:
        if len(flattened) >= limit:
            return
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            flattened[path] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            return
        flattened[path] = value if value is not None else "null"

    visit(source, prefix)
    return flattened


def _dataset_revision(dataset: PreparedDataset | None) -> dict[str, object]:
    if dataset is None:
        return {"sha256": None, "scenes": []}
    digest = sha256()
    scenes: list[dict[str, object]] = []
    for scene in sorted(dataset.scenes, key=lambda item: item.scene_id):
        image_path = Path(scene.image_path)
        try:
            image_stat = image_path.stat()
            image_identity = {
                "size_bytes": image_stat.st_size,
                "mtime_ns": image_stat.st_mtime_ns,
            }
        except OSError:
            image_identity = {"size_bytes": None, "mtime_ns": None}
        annotation_hash = _optional_file_hash(scene.annotation_file)
        footprint_hash = _optional_file_hash(scene.footprint_file)
        payload = {
            "scene_id": scene.scene_id,
            "image_path": str(image_path),
            **image_identity,
            "annotation_sha256": annotation_hash,
            "footprint_sha256": footprint_hash,
        }
        digest.update(repr(sorted(payload.items())).encode("utf-8"))
        scenes.append(payload)
    for path in (
        dataset.annotation_file,
        dataset.hard_negative_annotation_file,
        dataset.manifest_file,
    ):
        digest.update(str(path).encode("utf-8"))
        digest.update(str(_optional_file_hash(path)).encode("ascii"))
    return {"sha256": digest.hexdigest(), "scenes": scenes}


def _optional_file_hash(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        return None
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def _runtime_environment(
    project_root: str,
    dataset_revision: dict[str, object],
) -> dict[str, object]:
    commit = _git_output(project_root, "rev-parse", "HEAD")
    dirty = bool(_git_output(project_root, "status", "--porcelain"))
    package_names = (
        "torch",
        "transformers",
        "segmentation-models-pytorch",
        "numpy",
        "rasterio",
        "onnx",
        "onnxruntime",
        "mlflow",
    )
    packages: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            packages[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            packages[package_name] = None
    cuda: dict[str, object] = {"available": False, "version": None, "devices": []}
    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "version": torch.version.cuda,
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception:
        pass
    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "dataset_revision": dataset_revision,
    }


def _git_output(project_root: str, *arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _seed_training(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        return

    torch.set_num_threads(_positive_env_int("MLSYSTEM2_TORCH_NUM_THREADS", 4))
    try:
        torch.set_num_interop_threads(_positive_env_int("MLSYSTEM2_TORCH_NUM_INTEROP_THREADS", 2))
    except RuntimeError:
        # PyTorch разрешает задать interop pool только до первого параллельного вызова.
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(1, value)


class _CountingLoader:
    def __init__(
        self,
        loader: object,
        split: str,
        *,
        sampling_mode: str = "sequential",
        positive_factor_used: float | None = None,
        hard_negative_factor_used: float | None = None,
        background_factor_used: float | None = None,
    ) -> None:
        self.loader = loader
        self.split = split
        self.sampling_mode = sampling_mode
        self.positive_factor_used = positive_factor_used
        self.hard_negative_factor_used = hard_negative_factor_used
        self.background_factor_used = background_factor_used
        self.observed_batches = 0
        self.observed_tiles = 0
        self.observed_augmented_tiles = 0
        self.observed_augmented_positive_tiles = 0
        self.observed_augmented_hard_negative_tiles = 0
        self.observed_positive_tiles = 0
        self.observed_hard_negative_tiles = 0
        self.observed_background_tiles = 0

    def __iter__(self):
        for batch in self.loader:
            images = batch[0]
            tile_count = int(images.shape[0])
            meta = batch[2] if len(batch) > 2 else {}
            aug_count = int(meta.get("augmented_tile_count", 0)) if isinstance(meta, dict) else 0
            positive_count = (
                int(meta.get("positive_tile_count", 0)) if isinstance(meta, dict) else 0
            )
            hard_negative_count = (
                int(meta.get("hard_negative_tile_count", 0)) if isinstance(meta, dict) else 0
            )
            background_count = (
                int(meta.get("background_tile_count", 0)) if isinstance(meta, dict) else 0
            )
            augmented_positive_count = (
                int(meta.get("augmented_positive_tile_count", 0)) if isinstance(meta, dict) else 0
            )
            augmented_hard_negative_count = (
                int(meta.get("augmented_hard_negative_tile_count", 0))
                if isinstance(meta, dict)
                else 0
            )
            self.observed_batches += 1
            self.observed_tiles += tile_count
            self.observed_augmented_tiles += aug_count
            self.observed_augmented_positive_tiles += augmented_positive_count
            self.observed_augmented_hard_negative_tiles += augmented_hard_negative_count
            self.observed_positive_tiles += positive_count
            self.observed_hard_negative_tiles += hard_negative_count
            self.observed_background_tiles += background_count
            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def dataset(self):
        return getattr(self.loader, "dataset", None)

    def snapshot(self) -> dict[str, object]:
        warnings = []
        class_balance_warnings = _dataset_attr(self.dataset, "class_balance_warnings")
        if isinstance(class_balance_warnings, list):
            warnings.extend(str(item) for item in class_balance_warnings)
        tile_split_warnings = _dataset_attr(self.dataset, "tile_split_warnings")
        if isinstance(tile_split_warnings, list):
            warnings.extend(str(item) for item in tile_split_warnings)
        sampling_warnings = _dataset_attr(self.dataset, "sampling_warnings")
        if isinstance(sampling_warnings, list):
            warnings.extend(str(item) for item in sampling_warnings)
        loader_warnings = _loader_attr(self.loader, "warnings")
        if isinstance(loader_warnings, list):
            warnings.extend(str(item) for item in loader_warnings)
        observed_positive_ratio = _safe_ratio(self.observed_positive_tiles, self.observed_tiles)
        observed_hard_negative_ratio = _safe_ratio(
            self.observed_hard_negative_tiles,
            self.observed_tiles,
        )
        observed_background_ratio = _safe_ratio(
            self.observed_background_tiles,
            self.observed_tiles,
        )
        positive_ratio_abs_error = _ratio_abs_error(
            observed_positive_ratio,
            self.positive_factor_used,
        )
        hard_negative_ratio_abs_error = _ratio_abs_error(
            observed_hard_negative_ratio,
            self.hard_negative_factor_used,
        )
        background_ratio_abs_error = _ratio_abs_error(
            observed_background_ratio,
            self.background_factor_used,
        )
        _add_ratio_warning(
            warnings,
            "positive",
            positive_ratio_abs_error,
        )
        _add_ratio_warning(
            warnings,
            "hard_negative",
            hard_negative_ratio_abs_error,
        )
        _add_ratio_warning(
            warnings,
            "background",
            background_ratio_abs_error,
        )
        scene_tile_diagnostics = _dataset_attr(self.dataset, "scene_tile_diagnostics")
        tile_split_manifest = _dataset_attr(self.dataset, "tile_split_manifest")
        return {
            "tile_count": _safe_len(self.dataset),
            "batch_count": _safe_len(self),
            "scene_count": _dataset_attr(self.dataset, "scene_count"),
            **(
                {"scene_tile_diagnostics": scene_tile_diagnostics}
                if scene_tile_diagnostics is not None
                else {}
            ),
            "candidate_window_count": _dataset_attr(self.dataset, "candidate_window_count"),
            "candidate_window_count_before_valid_filter": _dataset_attr(
                self.dataset,
                "candidate_window_count_before_valid_filter",
            ),
            "black_filtered_window_count": _dataset_attr(
                self.dataset,
                "black_filtered_window_count",
            ),
            "valid_footprint_stride": _dataset_attr(self.dataset, "valid_footprint_stride"),
            "valid_footprint_valid_cells": _dataset_attr(
                self.dataset,
                "valid_footprint_valid_cells",
            ),
            "valid_footprint_total_cells": _dataset_attr(
                self.dataset,
                "valid_footprint_total_cells",
            ),
            "pool_window_count": _dataset_attr(self.dataset, "pool_window_count"),
            "split_window_count": _dataset_attr(self.dataset, "split_window_count"),
            **(
                {"tile_split_manifest": tile_split_manifest}
                if isinstance(tile_split_manifest, dict)
                and tile_split_manifest.get("strategy") == "scene_fold"
                else {}
            ),
            "estimated_positive_tiles": _dataset_attr(self.dataset, "estimated_positive_tiles"),
            "estimated_hard_negative_tiles": _dataset_attr(
                self.dataset,
                "estimated_hard_negative_tiles",
            ),
            "estimated_background_tiles": _dataset_attr(self.dataset, "estimated_background_tiles"),
            "sampling_mode": self.sampling_mode,
            "positive_factor_used": self.positive_factor_used,
            "hard_negative_factor_used": self.hard_negative_factor_used,
            "background_factor_used": self.background_factor_used,
            "cache_mode": _loader_attr(self.loader, "cache_mode"),
            "cached_batches": _loader_attr(self.loader, "cached_batches"),
            "cached_tiles": _loader_attr(self.loader, "cached_tiles"),
            "selected_batches": _loader_attr(self.loader, "selected_batches"),
            "selected_tiles": _loader_attr(self.loader, "selected_tiles"),
            "cache_estimated_bytes": _loader_attr(self.loader, "cache_estimated_bytes"),
            "cache_limit_bytes": _loader_attr(self.loader, "cache_limit_bytes"),
            "cache_fallback_reason": _loader_attr(self.loader, "cache_fallback_reason"),
            "class_balance_enabled": _dataset_attr(self.dataset, "class_balance_enabled"),
            "observed_batches": self.observed_batches,
            "observed_tiles": self.observed_tiles,
            "observed_positive_tiles": self.observed_positive_tiles,
            "observed_hard_negative_tiles": self.observed_hard_negative_tiles,
            "observed_background_tiles": self.observed_background_tiles,
            "observed_positive_ratio": observed_positive_ratio,
            "observed_hard_negative_ratio": observed_hard_negative_ratio,
            "observed_background_ratio": observed_background_ratio,
            "positive_ratio_abs_error": positive_ratio_abs_error,
            "hard_negative_ratio_abs_error": hard_negative_ratio_abs_error,
            "background_ratio_abs_error": background_ratio_abs_error,
            "observed_augmented_tiles": self.observed_augmented_tiles,
            "observed_augmented_positive_tiles": self.observed_augmented_positive_tiles,
            "observed_augmented_hard_negative_tiles": (self.observed_augmented_hard_negative_tiles),
            "observed_real_tiles": self.observed_tiles - self.observed_augmented_tiles,
            "warnings": warnings,
        }


def _tile_preparation_report(
    settings: SystemSettings,
    train_loader: _CountingLoader,
    val_loader: _CountingLoader,
) -> dict[str, object]:
    return {
        "tile_size": settings.tile_preparation.tile_size,
        "context": settings.tile_preparation.context,
        "core_size": (settings.tile_preparation.tile_size - 2 * settings.tile_preparation.context),
        "stride": settings.tile_preparation.stride,
        "seed": settings.tile_preparation.seed,
        "batch_size": settings.train.batch_size,
        "input_channels": settings.train.input_channels,
        "input_dtype": "uint8",
        "num_workers": settings.tile_preparation.num_workers,
        "prefetch_epochs": settings.tile_preparation.prefetch_epochs,
        "augmentation_level": settings.tile_preparation.augmentation_level,
        "positive_factor": settings.tile_preparation.positive_factor,
        "hard_negative_factor": settings.tile_preparation.hard_negative_factor,
        "background_factor": settings.tile_preparation.background_factor,
        "val_positive_factor": settings.tile_preparation.val_positive_factor,
        "class_balance": settings.tile_preparation.class_balance,
        "splits": {
            "train": train_loader.snapshot(),
            "val": val_loader.snapshot(),
        },
    }


def _sampling_mode(settings: SystemSettings, loader: object) -> str:
    if settings.train.pipeline_variant == "next_gen" and _loader_attr(loader, "cache_mode"):
        return "full_natural_validation"
    cache_mode = _loader_attr(loader, "cache_mode")
    if cache_mode == "memory":
        return "cached_balanced"
    if cache_mode == "lazy":
        return "lazy_balanced"
    uses_weighted_sampler = _uses_weighted_sampler(loader)
    if not uses_weighted_sampler:
        return "sequential"
    if bool(_dataset_attr(getattr(loader, "dataset", None), "class_balance_enabled")):
        return "weighted_class_balance"
    return "weighted_category_factor"


def _uses_weighted_sampler(loader: object) -> bool:
    sampler = getattr(loader, "sampler", None)
    return sampler is not None and sampler.__class__.__name__ in {
        "WeightedRandomSampler",
        "_EpochDrawSampler",
    }


def _weighted_factor_used(loader: object, attr: str, default: float) -> float:
    value = _dataset_attr(getattr(loader, "dataset", None), attr)
    if value is None:
        return default
    return float(value)


def _safe_len(value: object) -> int | None:
    try:
        return int(len(value))
    except TypeError:
        return None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _ratio_abs_error(observed: float | None, target: float | None) -> float | None:
    if observed is None or target is None:
        return None
    return abs(observed - target)


def _add_ratio_warning(
    warnings: list[str],
    category: str,
    ratio_abs_error: float | None,
) -> None:
    if ratio_abs_error is not None and ratio_abs_error > 0.1:
        warnings.append(
            f"observed_{category}_ratio отклонился от заданного factor больше чем на 0.1."
        )


def _dataset_attr(dataset: object, name: str) -> object:
    if dataset is None:
        return None
    return getattr(dataset, name, None)


def _loader_attr(loader: object, name: str) -> object:
    if loader is None:
        return None
    return getattr(loader, name, None)


def _train_request(
    settings: SystemSettings,
    model,
    train_loader: object,
    val_loader: object,
    dataset: PreparedDataset | None = None,
) -> TrainRequest:
    prepared_classes = list(dataset.classes) if dataset is not None else []
    prepared_annotations = list(dataset.class_annotations) if dataset is not None else []
    return TrainRequest(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainConfig(
            epochs=settings.train.epochs,
            task=settings.train.task,
            quality_metric=settings.train.quality_metric,
            pipeline_variant=settings.train.pipeline_variant,
            validation_interval_epochs=(
                settings.next_gen.validation_interval_epochs
                if settings.train.pipeline_variant == "next_gen"
                else 1
            ),
            threshold_mode=(
                settings.next_gen.threshold_mode
                if settings.train.pipeline_variant == "next_gen"
                else "optimize"
            ),
            evaluate_gaussian_blend=(
                settings.next_gen.evaluate_gaussian_blend
                if settings.train.pipeline_variant == "next_gen"
                else False
            ),
            batch_size=settings.train.batch_size,
            seed=settings.tile_preparation.seed,
            inference_context=settings.tile_preparation.context,
            device=settings.train.device,
            learning_rate=settings.train.learning_rate,
            weight_decay=settings.train.weight_decay,
            loss=settings.train.loss,
            focal_alpha=settings.train.focal_alpha,
            pos_weight=settings.train.pos_weight,
            background_weight=settings.train.background_weight,
            hard_negative_weight=settings.train.hard_negative_weight,
            tversky_alpha=settings.train.tversky_alpha,
            tversky_beta=settings.train.tversky_beta,
            threshold=settings.train.threshold,
            early_stopping_patience=settings.train.early_stopping_patience,
            max_train_batches_per_epoch=settings.train.max_train_batches_per_epoch,
            max_val_batches_per_epoch=settings.train.max_val_batches_per_epoch,
            max_training_time_sec=settings.train.max_training_time_sec,
            class_schema=[
                TrainClassDefinition(
                    id=item.id,
                    slug=item.slug,
                    name=item.name,
                    color=item.color,
                    priority=item.priority,
                )
                for item in prepared_classes
            ]
            or [
                TrainClassDefinition(
                    id=item.class_id,
                    slug=item.slug,
                    name=item.name,
                    color="#808080",
                    priority=item.priority,
                )
                for item in prepared_annotations
            ]
            or [
                TrainClassDefinition(
                    id=index,
                    slug=item.slug,
                    name=item.name,
                    color="#808080",
                    priority=item.priority,
                )
                for index, item in enumerate(settings.dataset.classes, start=1)
            ],
        ),
        checkpoint_dir=f"{settings.runtime.scratch_root.rstrip('/')}/checkpoints",
        sample_size=settings.tile_preparation.tile_size,
        run_metadata=(
            {
                "dataset_revision": _dataset_revision(dataset),
                "code_revision": _git_output(
                    settings.runtime.project_root, "rev-parse", "HEAD"
                ),
            }
            if settings.train.pipeline_variant == "next_gen"
            else {}
        ),
    )


def _validate_prepared_dataset_train_consistency(
    dataset: PreparedDataset,
    settings: SystemSettings,
) -> None:
    dataset_is_multiclass = bool(dataset.classes or dataset.class_annotations)
    expected_task = "multiclass" if dataset_is_multiclass else "binary"
    if settings.train.task != expected_task:
        raise TrainPipelineError(
            f"Подготовленный датасет имеет task={expected_task}, "
            f"но настройки обучения задают task={settings.train.task}."
        )
    expected_channels = (
        len(dataset.classes or dataset.class_annotations) + 1 if dataset_is_multiclass else 1
    )
    if settings.train.output_channels != expected_channels:
        raise TrainPipelineError(
            "Число выходных каналов не соответствует схеме датасета: "
            f"ожидается {expected_channels}, получено {settings.train.output_channels}."
        )


def _timing_report(
    total_started: float,
    timings: list[ModuleTiming],
    mlflow_elapsed: float,
) -> TimingReport:
    all_timings = [
        *timings,
        ModuleTiming(module="mlflow_logging", elapsed_sec=mlflow_elapsed),
    ]
    return TimingReport(total_pipeline_time_sec=elapsed_since(total_started), modules=all_timings)


def _expect_settings(value: object) -> SystemSettings:
    if not isinstance(value, SystemSettings):
        raise TrainPipelineError("settings.get_settings вернул неожиданное значение")
    return value


def _expect_dataset_result(value: object) -> DatasetPreparationResult:
    if not isinstance(value, DatasetPreparationResult):
        raise TrainPipelineError("dataset_preparing.prepare_dataset вернул неожиданное значение")
    return value


def _expect_train_result(value: object) -> TrainResult:
    if not isinstance(value, TrainResult):
        raise TrainPipelineError("train.train_model вернул неожиданное значение")
    return value


def _mlflow_class_tag(settings: SystemSettings) -> str:
    if settings.dataset.classes:
        return "multiclass"
    if settings.dataset.annotation_file is None and settings.dataset.annotations_dir is None:
        return "unknown"
    return _binary_dataset_class_slug(settings)


def _binary_dataset_class_slug(settings: SystemSettings) -> str:
    if settings.dataset.annotations_dir:
        annotations_path = Path(settings.dataset.annotations_dir)
        class_slug = _slug_part(annotations_path.parent.name)
        dataset_slug = _slug_part(annotations_path.name)
        if class_slug and dataset_slug:
            return f"{class_slug}_{dataset_slug}"
    annotation_path = Path(settings.dataset.annotation_file or "")
    if settings.dataset.scenes_file:
        scenes_path = Path(settings.dataset.scenes_file)
        if (
            annotation_path.parent == scenes_path.parent
            and annotation_path.parent.name
            and annotation_path.parent.parent.name
            and annotation_path.parent.parent.parent.name.lower() == "mlmarkup"
        ):
            class_slug = _slug_part(annotation_path.parent.parent.name)
            dataset_slug = _slug_part(annotation_path.parent.name)
            if class_slug and dataset_slug:
                return f"{class_slug}_{dataset_slug}"
    return _slug_part(annotation_path.stem) or "unknown"


def _slug_part(value: str) -> str:
    normalized = re.sub(r"\s+", "_", value.strip().lower())
    return re.sub(r"[^\w-]+", "", normalized, flags=re.UNICODE)


def _mlflow_dataset_name(settings: SystemSettings) -> str | None:
    if settings.dataset.classes:
        names = [Path(item.annotation_file).stem for item in settings.dataset.classes]
        return "+".join(names)
    if settings.dataset.annotations_dir is not None:
        return Path(settings.dataset.annotations_dir).name
    if settings.dataset.annotation_file is None:
        return None
    return Path(settings.dataset.annotation_file).stem

"""Оркестрация конвейера обучения."""

from __future__ import annotations

from dataclasses import dataclass

from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest, DatasetPreparationResult
from mlsystem2.mlflow_adapter.api import (
    end_run,
    log_dataset_preparation,
    log_pipeline_report,
    log_timing_report,
    log_training_artifacts,
    log_training_metrics,
    start_run,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef, MLflowRunStatus, MLflowStartRunRequest
from mlsystem2.models.api import create_model
from mlsystem2.models.contracts import ModelSpec
from mlsystem2.settings.api import load_settings
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.tile_preparation.api import build_tile_sources
from mlsystem2.tile_preparation.contracts import (
    TilePreparationConfig,
    TileSourceBundle,
    TileSourceRequest,
)
from mlsystem2.train.api import train_model
from mlsystem2.train.contracts import TrainConfig, TrainRequest, TrainResult

from ._timing import elapsed_since, now, timed_call
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
    load_settings: object
    start_run: object
    prepare_dataset: object
    build_tile_sources: object
    create_model: object
    train_model: object
    log_dataset_preparation: object
    log_training_metrics: object
    log_training_artifacts: object
    log_timing_report: object
    log_pipeline_report: object
    end_run: object


def _default_dependencies() -> _PipelineDependencies:
    return _PipelineDependencies(
        load_settings=load_settings,
        start_run=start_run,
        prepare_dataset=prepare_dataset,
        build_tile_sources=build_tile_sources,
        create_model=create_model,
        train_model=train_model,
        log_dataset_preparation=log_dataset_preparation,
        log_training_metrics=log_training_metrics,
        log_training_artifacts=log_training_artifacts,
        log_timing_report=log_timing_report,
        log_pipeline_report=log_pipeline_report,
        end_run=end_run,
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
            lambda: deps.load_settings(request.config_path),
        )
        timings.append(timing)
        settings = _expect_settings(settings)

        run = deps.start_run(_mlflow_start_request(settings, request))

        dataset_result, timing = timed_call(
            "dataset_preparing",
            lambda: deps.prepare_dataset(_dataset_request(settings)),
        )
        timings.append(timing)
        dataset_result = _expect_dataset_result(dataset_result)

        measure_mlflow(lambda: deps.log_dataset_preparation(run, dataset_result.report))

        if dataset_result.status == "failed" or dataset_result.manifest is None:
            report = PipelineReport(
                status=PipelineStatus.FAILED,
                message="Подготовка датасета завершилась ошибкой.",
                dataset_status=dataset_result.status,
                errors=dataset_result.errors,
                warnings=dataset_result.report.warnings,
                artifacts={"footprints": dataset_result.report.footprints_artifact_uri},
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

        tile_bundle, timing = timed_call(
            "tile_preparation",
            lambda: deps.build_tile_sources(_tile_request(settings, dataset_result.manifest)),
        )
        timings.append(timing)
        tile_bundle = _expect_tile_bundle(tile_bundle)

        model = deps.create_model(_model_spec(settings))
        train_result, timing = timed_call(
            "train",
            lambda: deps.train_model(_train_request(settings, model, tile_bundle)),
        )
        timings.append(timing)
        train_result = _expect_train_result(train_result)

        measure_mlflow(lambda: deps.log_training_metrics(run, train_result))
        measure_mlflow(lambda: deps.log_training_artifacts(run, train_result))

        report = PipelineReport(
            status=PipelineStatus.SUCCEEDED,
            message="Конвейер обучения завершен.",
            dataset_status=dataset_result.status,
            errors=[],
            warnings=dataset_result.report.warnings + tile_bundle.report.warnings,
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
        run_name=request.run_name,
        tags={"pipeline": "train"},
    )


def _dataset_request(settings: SystemSettings) -> DatasetPreparationRequest:
    return DatasetPreparationRequest(
        images_uri=settings.storage.images_uri,
        scenes_file=settings.dataset.scenes_file,
        annotation_file=settings.dataset.annotation_file,
        default_class_dir=settings.dataset.default_class_dir,
        val_fraction=settings.dataset.val_fraction,
        split_strategy=settings.dataset.split_strategy,
        output_uri=settings.runtime.scratch_root,
    )


def _tile_request(settings: SystemSettings, manifest) -> TileSourceRequest:
    return TileSourceRequest(
        manifest=manifest,
        config=TilePreparationConfig(
            tile_size=settings.tile_preparation.tile_size,
            stride=settings.tile_preparation.stride,
            prefetch_workers=settings.tile_preparation.prefetch_workers,
            prefetch_batches=settings.tile_preparation.prefetch_batches,
            use_neighbor_footprints=settings.tile_preparation.use_neighbor_footprints,
        ),
        scratch_uri=settings.runtime.scratch_root,
    )


def _model_spec(settings: SystemSettings) -> ModelSpec:
    return ModelSpec(
        name=settings.train.model_name,
        input_channels=settings.train.input_channels,
        output_channels=settings.train.output_channels,
    )


def _train_request(settings: SystemSettings, model, tile_bundle: TileSourceBundle) -> TrainRequest:
    return TrainRequest(
        model=model,
        train_source=tile_bundle.train,
        val_source=tile_bundle.val,
        config=TrainConfig(
            epochs=settings.train.epochs,
            batch_size=settings.train.batch_size,
            device=settings.train.device,
            num_workers=settings.train.num_workers,
        ),
        checkpoint_dir=f"{settings.runtime.scratch_root.rstrip('/')}/checkpoints",
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
        raise TrainPipelineError("settings.load_settings вернул неожиданное значение")
    return value


def _expect_dataset_result(value: object) -> DatasetPreparationResult:
    if not isinstance(value, DatasetPreparationResult):
        raise TrainPipelineError("dataset_preparing.prepare_dataset вернул неожиданное значение")
    return value


def _expect_tile_bundle(value: object) -> TileSourceBundle:
    if not isinstance(value, TileSourceBundle):
        raise TrainPipelineError("tile_preparation.build_tile_sources вернул неожиданное значение")
    return value


def _expect_train_result(value: object) -> TrainResult:
    if not isinstance(value, TrainResult):
        raise TrainPipelineError("train.train_model вернул неожиданное значение")
    return value

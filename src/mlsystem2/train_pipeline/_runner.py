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
from mlsystem2.models.api import create_model, load_checkpoint
from mlsystem2.models.contracts import LoadCheckpointRequest, ModelSpec
from mlsystem2.settings.api import get_settings
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.tile_preparation.api import create_tile_dataloader
from mlsystem2.tile_preparation.contracts import TileDataloaderRequest
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
    get_settings: object
    start_run: object
    prepare_dataset: object
    create_tile_dataloader: object
    create_model: object
    load_checkpoint: object
    train_model: object
    log_dataset_preparation: object
    log_training_metrics: object
    log_training_artifacts: object
    log_timing_report: object
    log_pipeline_report: object
    end_run: object


def _default_dependencies() -> _PipelineDependencies:
    return _PipelineDependencies(
        get_settings=get_settings,
        start_run=start_run,
        prepare_dataset=prepare_dataset,
        create_tile_dataloader=create_tile_dataloader,
        create_model=create_model,
        load_checkpoint=load_checkpoint,
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
            lambda: deps.get_settings(),
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

        loaders, timing = timed_call(
            "tile_preparation",
            lambda: (
                deps.create_tile_dataloader(
                    _tile_request(
                        dataset_result.dataset.train_vrt_xml,
                        dataset_result.dataset.annotation_file,
                        settings.train.batch_size,
                        "train",
                    )
                ),
                deps.create_tile_dataloader(
                    _tile_request(
                        dataset_result.dataset.val_vrt_xml,
                        dataset_result.dataset.annotation_file,
                        settings.train.batch_size,
                        "val",
                    )
                ),
            ),
        )
        timings.append(timing)
        train_loader, val_loader = loaders

        model = _load_or_create_model(settings, deps)
        train_result, timing = timed_call(
            "train",
            lambda: deps.train_model(_train_request(settings, model, train_loader, val_loader)),
        )
        timings.append(timing)
        train_result = _expect_train_result(train_result)

        measure_mlflow(lambda: deps.log_training_metrics(run, train_result))
        measure_mlflow(lambda: deps.log_training_artifacts(run, train_result))

        report = PipelineReport(
            status=PipelineStatus.SUCCEEDED,
            message="Конвейер обучения завершен.",
            dataset_status=dataset_result.report.status,
            errors=[],
            warnings=[],
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
        images_dir=settings.dataset.images_dir,
        scenes_file=settings.dataset.scenes_file,
        annotation_file=settings.dataset.annotation_file,
        val_fraction=settings.dataset.val_fraction,
    )


def _tile_request(
    vrt_xml: str,
    annotation_file: str,
    batch_size: int,
    mode: str,
) -> TileDataloaderRequest:
    return TileDataloaderRequest(
        vrt_xml=vrt_xml,
        annotation_file=annotation_file,
        batch_size=batch_size,
        mode=mode,
    )


def _model_spec(settings: SystemSettings) -> ModelSpec:
    return ModelSpec(
        name=settings.train.model_name,
        input_channels=settings.train.input_channels,
        output_channels=settings.train.output_channels,
        pretrained=settings.train.pretrained,
    )


def _load_or_create_model(settings: SystemSettings, deps: _PipelineDependencies):
    spec = _model_spec(settings)
    if settings.train.initial_checkpoint_uri:
        loaded = deps.load_checkpoint(
            LoadCheckpointRequest(
                checkpoint_uri=settings.train.initial_checkpoint_uri,
                model_spec=spec,
                map_location=settings.train.device,
            )
        )
        return loaded.model
    return deps.create_model(spec)


def _train_request(
    settings: SystemSettings,
    model,
    train_loader: object,
    val_loader: object,
) -> TrainRequest:
    return TrainRequest(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=TrainConfig(
            epochs=settings.train.epochs,
            batch_size=settings.train.batch_size,
            device=settings.train.device,
            learning_rate=settings.train.learning_rate,
            weight_decay=settings.train.weight_decay,
            loss=settings.train.loss,
            focal_alpha=settings.train.focal_alpha,
            pos_weight=settings.train.pos_weight,
            tversky_alpha=settings.train.tversky_alpha,
            tversky_beta=settings.train.tversky_beta,
            threshold=settings.train.threshold,
            early_stopping_patience=settings.train.early_stopping_patience,
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

"""Оркестрация конвейера инференса."""

from __future__ import annotations

from mlsystem2.inference.api import run_inference
from mlsystem2.inference.contracts import InferenceConfig, InferenceRequest, InferenceResult
from mlsystem2.mlflow_adapter.api import end_run, log_pipeline_report, log_timing_report, start_run
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef, MLflowRunStatus, MLflowStartRunRequest
from mlsystem2.settings.api import load_settings
from mlsystem2.settings.contracts import SystemSettings
from mlsystem2.train_pipeline.contracts import (
    ModuleTiming,
    PipelineReport,
    PipelineStatus,
    TimingReport,
)

from ._timing import elapsed_since, now, timed_call
from .contracts import InferencePipelineError, InferencePipelineRequest, InferencePipelineResult


def run_inference_pipeline(request: InferencePipelineRequest) -> InferencePipelineResult:
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
        settings, timing = timed_call("settings", lambda: load_settings(request.config_path))
        timings.append(timing)
        settings = _expect_settings(settings)

        run = start_run(_mlflow_start_request(settings, request))
        result, timing = timed_call("inference", lambda: run_inference(_inference_request(settings)))
        timings.append(timing)
        result = _expect_inference_result(result)

        report = PipelineReport(
            status=PipelineStatus.SUCCEEDED,
            message="Конвейер инференса завершен.",
            artifacts={"inference_artifacts": [item.uri for item in result.artifacts]},
        )
        timing_report = _timing_report(total_started, timings, mlflow_elapsed)
        measure_mlflow(lambda: log_timing_report(run, timing_report))
        measure_mlflow(lambda: log_pipeline_report(run, report))
        measure_mlflow(lambda: end_run(run, MLflowRunStatus.FINISHED))
        return InferencePipelineResult(
            status=PipelineStatus.SUCCEEDED,
            mlflow_run=run,
            timings=_timing_report(total_started, timings, mlflow_elapsed),
            report=report,
        )
    except Exception as exc:
        if run is not None:
            report = PipelineReport(
                status=PipelineStatus.FAILED,
                message="Конвейер инференса завершился невосстановимой ошибкой.",
                errors=[str(exc)],
            )
            try:
                timing_report = _timing_report(total_started, timings, mlflow_elapsed)
                measure_mlflow(lambda: log_timing_report(run, timing_report))
                measure_mlflow(lambda: log_pipeline_report(run, report))
                measure_mlflow(lambda: end_run(run, MLflowRunStatus.FAILED))
            except Exception:
                pass
        raise InferencePipelineError("Конвейер инференса завершился ошибкой") from exc


def _mlflow_start_request(
    settings: SystemSettings,
    request: InferencePipelineRequest,
) -> MLflowStartRunRequest:
    return MLflowStartRunRequest(
        enabled=settings.mlflow.enabled,
        tracking_uri=settings.mlflow.tracking_uri,
        experiment_name=settings.mlflow.experiment_name,
        run_name=request.run_name,
        tags={"pipeline": "inference"},
    )


def _inference_request(settings: SystemSettings) -> InferenceRequest:
    return InferenceRequest(
        config=InferenceConfig(
            checkpoint_uri=settings.inference.checkpoint_uri,
            threshold=settings.inference.threshold,
            batch_size=settings.inference.batch_size,
            device=settings.inference.device,
        ),
        images_uri=settings.storage.images_uri,
        output_uri=settings.runtime.scratch_root,
        model_spec=None,
    )


def _timing_report(
    total_started: float,
    timings: list[ModuleTiming],
    mlflow_elapsed: float,
) -> TimingReport:
    return TimingReport(
        total_pipeline_time_sec=elapsed_since(total_started),
        modules=[*timings, ModuleTiming(module="mlflow_logging", elapsed_sec=mlflow_elapsed)],
    )


def _expect_settings(value: object) -> SystemSettings:
    if not isinstance(value, SystemSettings):
        raise InferencePipelineError("settings.load_settings вернул неожиданное значение")
    return value


def _expect_inference_result(value: object) -> InferenceResult:
    if not isinstance(value, InferenceResult):
        raise InferencePipelineError("inference.run_inference вернул неожиданное значение")
    return value

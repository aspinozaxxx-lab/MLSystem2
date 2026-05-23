"""Обертка клиента MLflow."""

from __future__ import annotations

from pathlib import Path

from mlsystem2.dataset_preparing.contracts import DatasetPreparationReport
from mlsystem2.train.contracts import TrainResult
from mlsystem2.train_pipeline.contracts import PipelineReport, TimingReport

from .contracts import MLflowAdapterError, MLflowRunRef, MLflowRunStatus, MLflowStartRunRequest


def start_run(request: MLflowStartRunRequest) -> MLflowRunRef:
    if not request.enabled:
        return MLflowRunRef(
            run_id="disabled",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=False,
        )
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(request.tracking_uri)
        mlflow.set_experiment(request.experiment_name)
        run = mlflow.start_run(run_name=request.run_name, tags=request.tags)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось начать запуск MLflow") from exc
    return MLflowRunRef(
        run_id=run.info.run_id,
        experiment_name=request.experiment_name,
        tracking_uri=request.tracking_uri,
        active=True,
    )


def log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None:
    if not run.active:
        return
    _log_dict(_model_dump(report), "reports/dataset_preparation.json")


def log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None:
    if not run.active:
        return
    mlflow = _mlflow()
    try:
        for item in result.history:
            mlflow.log_metric("train/loss", item.train_loss, step=item.epoch)
            mlflow.log_metric("val/loss", item.val_loss, step=item.epoch)
            mlflow.log_metric("val/pixel_precision", item.val_pixel_precision, step=item.epoch)
            mlflow.log_metric("val/pixel_recall", item.val_pixel_recall, step=item.epoch)
            mlflow.log_metric("val/pixel_f1", item.val_pixel_f1, step=item.epoch)
            mlflow.log_metric("train/epoch_time_sec", item.epoch_time_sec, step=item.epoch)
        mlflow.log_metric("train/epochs_total", result.epochs_total)
        mlflow.log_metric("train/training_time_sec", result.training_time_sec)
        if result.history:
            mlflow.log_metric("val/best_pixel_f1", max(item.val_pixel_f1 for item in result.history))
            mlflow.log_metric("val/final_pixel_f1", result.history[-1].val_pixel_f1)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось записать метрики обучения в MLflow") from exc


def log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None:
    if not run.active:
        return
    _log_dict(
        {"history": [_model_dump(item) for item in result.history]},
        "reports/training_history_full.json",
    )
    for path in (result.best_checkpoint_path, result.final_checkpoint_path):
        if path is not None and Path(path).exists():
            _log_artifact(path, "checkpoints")


def log_timing_report(run: MLflowRunRef, report: TimingReport) -> None:
    if not run.active:
        return
    _log_dict(_model_dump(report), "reports/pipeline_timings.json")


def log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None:
    if not run.active:
        return
    _log_dict(_model_dump(report), "reports/pipeline_summary.json")


def end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None:
    if not run.active:
        return
    mlflow = _mlflow()
    try:
        mlflow.end_run(status=status.value)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось завершить запуск MLflow") from exc


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise MLflowAdapterError("MLflow обязателен, когда логирование MLflow включено") from exc
    return mlflow


def _log_dict(payload: dict[str, object], artifact_file: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_dict(payload, artifact_file)
    except Exception as exc:
        raise MLflowAdapterError(f"Не удалось записать артефакт MLflow: {artifact_file}") from exc


def _log_artifact(path: str, artifact_path: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:
        raise MLflowAdapterError(f"Не удалось записать файл артефакта MLflow: {path}") from exc


def _model_dump(value: object) -> dict[str, object]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise MLflowAdapterError("Для сериализации в MLflow ожидалась модель Pydantic")

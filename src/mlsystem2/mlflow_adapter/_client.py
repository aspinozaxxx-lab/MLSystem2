"""Обертка клиента MLflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from mlsystem2.dataset_preparing.contracts import DatasetPreparationReport
from mlsystem2.train.contracts import EpochMetrics, TrainResult
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
        run_name = request.run_name or _auto_run_name(mlflow, request.experiment_name, request.tags)
        run = mlflow.start_run(run_name=run_name, tags=request.tags)
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


def log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None:
    if not run.active:
        return
    _log_dict(report, "reports/tile_preparation.json")


def log_run_config(run: MLflowRunRef, config_path: str | Path) -> None:
    if not run.active:
        return
    path = Path(config_path)
    if not path.is_file():
        raise MLflowAdapterError(f"Файл настроек для MLflow не найден: {path}")
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MLflowAdapterError(f"Не удалось прочитать файл настроек для MLflow: {path}") from exc
    _log_text(content, "config/train_config.yaml")


def log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None:
    if not run.active:
        return
    mlflow = _mlflow()
    try:
        mlflow.log_metric("train/loss", metrics.train_loss, step=metrics.epoch)
        mlflow.log_metric("train/optimizer_steps", metrics.train_optimizer_steps, step=metrics.epoch)
        mlflow.log_metric(
            "train/skipped_optimizer_steps",
            metrics.train_skipped_optimizer_steps,
            step=metrics.epoch,
        )
        mlflow.log_metric("val/loss", metrics.val_loss, step=metrics.epoch)
        mlflow.log_metric("val/pixel_precision", metrics.val_pixel_precision, step=metrics.epoch)
        mlflow.log_metric("val/pixel_recall", metrics.val_pixel_recall, step=metrics.epoch)
        mlflow.log_metric("val/pixel_f1", metrics.val_pixel_f1, step=metrics.epoch)
        mlflow.log_metric("val/positive_pixels", metrics.val_positive_pixels, step=metrics.epoch)
        mlflow.log_metric(
            "val/pred_positive_pixels",
            metrics.val_pred_positive_pixels,
            step=metrics.epoch,
        )
        mlflow.log_metric("val/true_positive", metrics.val_true_positive, step=metrics.epoch)
        mlflow.log_metric("val/false_positive", metrics.val_false_positive, step=metrics.epoch)
        mlflow.log_metric("val/false_negative", metrics.val_false_negative, step=metrics.epoch)
        mlflow.log_metric("val/best_threshold", metrics.val_best_threshold, step=metrics.epoch)
        mlflow.log_metric(
            "val/best_threshold_pixel_f1",
            metrics.val_best_threshold_pixel_f1,
            step=metrics.epoch,
        )
        mlflow.log_metric(
            "val/best_threshold_precision",
            metrics.val_best_threshold_precision,
            step=metrics.epoch,
        )
        mlflow.log_metric(
            "val/best_threshold_recall",
            metrics.val_best_threshold_recall,
            step=metrics.epoch,
        )
        mlflow.log_metric("val/prob_mean", metrics.val_prob_mean, step=metrics.epoch)
        mlflow.log_metric("val/prob_min", metrics.val_prob_min, step=metrics.epoch)
        mlflow.log_metric("val/prob_max", metrics.val_prob_max, step=metrics.epoch)
        mlflow.log_metric("val/prob_p50", metrics.val_prob_p50, step=metrics.epoch)
        mlflow.log_metric("val/prob_p90", metrics.val_prob_p90, step=metrics.epoch)
        mlflow.log_metric("val/prob_p99", metrics.val_prob_p99, step=metrics.epoch)
        mlflow.log_metric("train/epoch_time_sec", metrics.epoch_time_sec, step=metrics.epoch)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось записать метрики эпохи в MLflow") from exc


def log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None:
    if not run.active:
        return
    mlflow = _mlflow()
    try:
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
        raise MLflowAdapterError(
            "MLflow обязателен, когда логирование MLflow включено"
        ) from exc
    return mlflow


def _auto_run_name(mlflow, experiment_name: str, tags: dict[str, str]) -> str | None:
    class_slug = tags.get("class")
    if not class_slug:
        return None

    date = datetime.now().strftime("%d%m")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return _next_run_name([], class_slug, date)

    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs([experiment.experiment_id], max_results=1000)
        existing_names = [run.data.tags.get("mlflow.runName", "") for run in runs]
    except Exception:
        existing_names = []
    return _next_run_name(existing_names, class_slug, date)


def _next_run_name(existing_names: list[str], class_slug: str, date: str) -> str:
    prefix = f"{class_slug}_{date}_"
    max_number = 0
    for name in existing_names:
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        if suffix.isdigit():
            max_number = max(max_number, int(suffix))
    return f"{prefix}{max_number + 1}"


def _log_dict(payload: dict[str, object], artifact_file: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_dict(payload, artifact_file)
    except Exception as exc:
        raise MLflowAdapterError(
            f"Не удалось записать артефакт MLflow: {artifact_file}"
        ) from exc


def _log_text(content: str, artifact_file: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_text(content, artifact_file)
    except Exception as exc:
        raise MLflowAdapterError(
            f"Не удалось записать текстовый артефакт MLflow: {artifact_file}"
        ) from exc


def _log_artifact(path: str, artifact_path: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:
        raise MLflowAdapterError(
            f"Не удалось записать файл артефакта MLflow: {path}"
        ) from exc


def _model_dump(value: object) -> dict[str, object]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise MLflowAdapterError("Для сериализации в MLflow ожидалась модель Pydantic")

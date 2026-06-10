"""Обертка клиента MLflow."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha1
import os
from pathlib import Path
import shutil
import tempfile

from mlsystem2.dataset_preparing.contracts import DatasetPreparationReport
from mlsystem2.train.contracts import EpochMetrics, TrainResult
from mlsystem2.train_pipeline.contracts import PipelineReport, TimingReport

from .contracts import (
    MLflowAdapterError,
    MLflowBestCheckpoint,
    MLflowDownloadedArtifact,
    MLflowExperiment,
    MLflowExperimentRequest,
    MLflowRunRef,
    MLflowRunStatus,
    MLflowStartRunRequest,
    MLflowTrainingProgress,
)


BEST_CHECKPOINT_METRIC = "val/best_threshold_pixel_f1"
BEST_THRESHOLD_METRIC = "val/best_threshold"
TRAIN_EPOCH_TIME_METRIC = "train/epoch_time_sec"
BEST_CHECKPOINT_ARTIFACT_PATH = "checkpoints/best.pt"


def list_experiments(tracking_uri: str) -> list[MLflowExperiment]:
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
    except Exception as exc:
        raise MLflowAdapterError("Не удалось получить список экспериментов MLflow") from exc
    return [
        MLflowExperiment(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            lifecycle_stage=getattr(experiment, "lifecycle_stage", None),
        )
        for experiment in experiments
    ]


def create_experiment(request: MLflowExperimentRequest) -> MLflowExperiment:
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(request.tracking_uri)
        existing = mlflow.get_experiment_by_name(request.name)
        if existing is not None:
            return MLflowExperiment(
                experiment_id=existing.experiment_id,
                name=existing.name,
                lifecycle_stage=getattr(existing, "lifecycle_stage", None),
            )
        experiment_id = mlflow.create_experiment(request.name)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось создать эксперимент MLflow") from exc
    return MLflowExperiment(experiment_id=experiment_id, name=request.name, lifecycle_stage="active")


def get_best_training_checkpoint(
    tracking_uri: str,
    run_id: str,
) -> MLflowBestCheckpoint | None:
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        history = client.get_metric_history(run_id, BEST_CHECKPOINT_METRIC)
        thresholds = client.get_metric_history(run_id, BEST_THRESHOLD_METRIC)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось прочитать лучшую метрику training run из MLflow") from exc
    if not history:
        return None
    best = max(history, key=lambda item: (float(item.value), -int(item.step)))
    threshold = _metric_value_at_step(thresholds, int(best.step))
    artifact_root = getattr(run.info, "artifact_uri", None)
    artifact_uri = (
        f"{artifact_root.rstrip('/')}/{BEST_CHECKPOINT_ARTIFACT_PATH}"
        if isinstance(artifact_root, str) and artifact_root.strip()
        else None
    )
    return MLflowBestCheckpoint(
        tracking_uri=tracking_uri,
        run_id=run_id,
        metric_name=BEST_CHECKPOINT_METRIC,
        f1_score=float(best.value),
        epoch=int(best.step),
        threshold=threshold,
        artifact_path=BEST_CHECKPOINT_ARTIFACT_PATH,
        artifact_uri=artifact_uri,
    )


def get_training_epoch_progress(
    tracking_uri: str,
    run_id: str,
) -> MLflowTrainingProgress:
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        history = client.get_metric_history(run_id, TRAIN_EPOCH_TIME_METRIC)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось прочитать прогресс training run из MLflow") from exc
    completed_epochs = max((int(item.step) for item in history), default=0)
    return MLflowTrainingProgress(completed_epochs=max(0, completed_epochs))


def download_run_artifact(
    *,
    tracking_uri: str,
    run_id: str,
    artifact_path: str,
    dst_dir: str | Path,
) -> MLflowDownloadedArtifact:
    mlflow = _mlflow()
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(run_id, artifact_path, str(dst_dir))
    except Exception as exc:
        raise MLflowAdapterError("Не удалось скачать артефакт MLflow") from exc
    return MLflowDownloadedArtifact(
        run_id=run_id,
        artifact_path=artifact_path,
        local_path=str(local_path),
    )


def _metric_value_at_step(history: object, step: int) -> float | None:
    values = [
        float(item.value)
        for item in history or []
        if int(getattr(item, "step", -1)) == step
    ]
    return values[-1] if values else None


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
        tags = _run_tags(request)
        _ensure_experiment_dataset(mlflow, request.experiment_name, request.dataset)
        run_name = request.run_name or _auto_run_name(mlflow, request.experiment_name, tags)
        run = mlflow.start_run(run_name=run_name, tags=tags)
        _write_run_id_file(run.info.run_id)
        _log_input_dataset(mlflow, request.dataset, context=tags.get("pipeline"))
    except Exception as exc:
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run(status=MLflowRunStatus.FAILED.value)
        except Exception:
            pass
        raise MLflowAdapterError("Не удалось начать запуск MLflow") from exc
    return MLflowRunRef(
        run_id=run.info.run_id,
        experiment_name=request.experiment_name,
        tracking_uri=request.tracking_uri,
        active=True,
    )


def _write_run_id_file(run_id: str) -> None:
    path = os.environ.get("MLSYSTEM2_MLFLOW_RUN_ID_FILE")
    if not path:
        return
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{run_id}\n", encoding="utf-8")
    except OSError as exc:
        raise MLflowAdapterError(f"Не удалось записать id запуска MLflow в файл: {target}") from exc


def _run_tags(request: MLflowStartRunRequest) -> dict[str, str]:
    tags = dict(request.tags)
    if request.dataset:
        tags["dataset"] = request.dataset
    return tags


def _ensure_experiment_dataset(
    mlflow,
    experiment_name: str,
    dataset_name: str | None,
) -> None:
    if not dataset_name:
        return
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return
    client = mlflow.tracking.MlflowClient()
    search_datasets = getattr(client, "search_datasets", None)
    create_dataset = getattr(client, "create_dataset", None)
    if search_datasets is None or create_dataset is None:
        return
    datasets = search_datasets(experiment_ids=[experiment.experiment_id], max_results=1000)
    if any(dataset.name == dataset_name for dataset in datasets):
        return
    create_dataset(
        name=dataset_name,
        experiment_id=experiment.experiment_id,
        tags={"dataset": dataset_name, "source": "MLMarkup geojson stem"},
    )


def _log_input_dataset(mlflow, dataset_name: str | None, *, context: str | None) -> None:
    if not dataset_name:
        return
    try:
        import numpy as np

        digest = sha1(dataset_name.encode("utf-8")).hexdigest()[:8]
        dataset = mlflow.data.from_numpy(
            np.empty((0, 0), dtype=np.float32),
            source=dataset_name,
            name=dataset_name,
            digest=digest,
        )
        mlflow.log_input(
            dataset,
            context=context or "run",
            tags={"dataset": dataset_name},
        )
    except Exception as exc:
        raise MLflowAdapterError("Не удалось записать dataset input в MLflow") from exc


def log_dataset_preparation(run: MLflowRunRef, report: DatasetPreparationReport) -> None:
    if not run.active:
        return
    _ensure_run_active(run)
    _log_dict(_model_dump(report), "reports/dataset_preparation.json")


def log_dataset_artifacts(run: MLflowRunRef, files: dict[str, str | Path]) -> None:
    if not run.active:
        return
    _ensure_run_active(run)
    with tempfile.TemporaryDirectory(prefix="mlsystem2-mlflow-dataset-") as temp_dir:
        temp_root = Path(temp_dir)
        for artifact_name, source in files.items():
            target_name = _safe_dataset_artifact_name(artifact_name)
            source_path = Path(source)
            if not source_path.is_file():
                raise MLflowAdapterError(f"Файл датасета для MLflow не найден: {source_path}")
            temp_path = temp_root / target_name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source_path, temp_path)
            except OSError as exc:
                raise MLflowAdapterError(f"Не удалось подготовить файл датасета для MLflow: {source_path}") from exc
            artifact_parent = temp_path.relative_to(temp_root).parent
            artifact_path = "dataset" if artifact_parent == Path(".") else f"dataset/{artifact_parent.as_posix()}"
            _log_artifact(temp_path, artifact_path)


def log_tile_preparation(run: MLflowRunRef, report: dict[str, object]) -> None:
    if not run.active:
        return
    _ensure_run_active(run)
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
    _ensure_run_active(run)
    _log_text(content, "config/train_config.yaml")


def log_training_epoch(run: MLflowRunRef, metrics: EpochMetrics) -> None:
    if not run.active:
        return
    mlflow = _ensure_run_active(run)
    try:
        mlflow.log_metric("train/loss", metrics.train_loss, step=metrics.epoch)
        mlflow.log_metric("val/loss", metrics.val_loss, step=metrics.epoch)
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
        mlflow.log_metric("train/epoch_time_sec", metrics.epoch_time_sec, step=metrics.epoch)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось записать метрики эпохи в MLflow") from exc


def log_training_metrics(run: MLflowRunRef, result: TrainResult) -> None:
    if not run.active:
        return
    mlflow = _ensure_run_active(run)
    try:
        mlflow.log_metric("train/epochs_total", result.epochs_total)
        mlflow.log_metric("train/training_time_sec", result.training_time_sec)
        if result.history:
            mlflow.log_metric(
                "train/best_threshold_pixel_f1",
                max(item.val_best_threshold_pixel_f1 for item in result.history),
            )
    except Exception as exc:
        raise MLflowAdapterError("Не удалось записать метрики обучения в MLflow") from exc


def log_training_artifacts(run: MLflowRunRef, result: TrainResult) -> None:
    if not run.active:
        return
    _ensure_run_active(run)
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
    _ensure_run_active(run)
    _log_dict(_model_dump(report), "reports/pipeline_timings.json")


def log_pipeline_report(run: MLflowRunRef, report: PipelineReport) -> None:
    if not run.active:
        return
    _ensure_run_active(run)
    _log_dict(_model_dump(report), "reports/pipeline_summary.json")


def end_run(run: MLflowRunRef, status: MLflowRunStatus) -> None:
    if not run.active:
        return
    mlflow = _mlflow()
    try:
        _set_tracking_uri(mlflow, run.tracking_uri)
        active = _active_run(mlflow)
        if _active_run_id(active) == run.run_id:
            mlflow.end_run(status=status.value)
            return
        client = mlflow.tracking.MlflowClient()
        client.set_terminated(run.run_id, status.value)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось завершить запуск MLflow") from exc


def mark_run_killed(tracking_uri: str, run_id: str) -> None:
    mlflow = _mlflow()
    try:
        _set_tracking_uri(mlflow, tracking_uri)
        active = _active_run(mlflow)
        if _active_run_id(active) == run_id:
            mlflow.end_run(status=MLflowRunStatus.KILLED.value)
            return
        client = mlflow.tracking.MlflowClient()
        client.set_terminated(run_id, MLflowRunStatus.KILLED.value)
    except Exception as exc:
        raise MLflowAdapterError("Не удалось пометить запуск MLflow как убитый") from exc


def _ensure_run_active(run: MLflowRunRef):
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, run.tracking_uri)
    active = _active_run(mlflow)
    if _active_run_id(active) == run.run_id:
        return mlflow
    if active is not None:
        end_run_fn = getattr(mlflow, "end_run", None)
        if end_run_fn is not None:
            end_run_fn()
    start_run_fn = getattr(mlflow, "start_run", None)
    if start_run_fn is not None:
        start_run_fn(run_id=run.run_id)
    return mlflow


def _set_tracking_uri(mlflow, tracking_uri: str) -> None:
    set_tracking_uri = getattr(mlflow, "set_tracking_uri", None)
    if set_tracking_uri is not None:
        set_tracking_uri(tracking_uri)


def _active_run(mlflow):
    active_run_fn = getattr(mlflow, "active_run", None)
    return active_run_fn() if active_run_fn is not None else None


def _active_run_id(active_run) -> str | None:
    info = getattr(active_run, "info", None)
    run_id = getattr(info, "run_id", None)
    return str(run_id) if run_id is not None else None


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


def _log_artifact(path: str | Path, artifact_path: str) -> None:
    mlflow = _mlflow()
    try:
        mlflow.log_artifact(path, artifact_path=artifact_path)
    except Exception as exc:
        raise MLflowAdapterError(
            f"Не удалось записать файл артефакта MLflow: {path}"
        ) from exc


def _safe_dataset_artifact_name(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MLflowAdapterError(f"Некорректное имя артефакта датасета: {value}")
    suffix = path.suffix.lower()
    if suffix not in {".txt", ".geojson"}:
        raise MLflowAdapterError(f"Файл датасета должен быть .txt или .geojson: {value}")
    return path


def _model_dump(value: object) -> dict[str, object]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    raise MLflowAdapterError("Для сериализации в MLflow ожидалась модель Pydantic")

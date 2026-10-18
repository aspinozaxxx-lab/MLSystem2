"""Новый этап от сохранённых весов на прежнем либо актуальном составе датасета."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import download_run_artifact, get_finished_run_artifact
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import find_managed_dataset
from ._markup_lock import markup_read_lock
from ._models import JobRow, TrainingResultRow
from ._template_selection import effective_inference_template_row
from ._queueing import (
    POST_TRAINING_INFERENCE_CONFIG_KEY,
    POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
    SECONDARY_PRIORITY_CONFIG_KEY,
    STOP_AND_SAVE_BEST_CONFIG_KEY,
    TRAINING_STARTED_BY_CONFIG_KEY,
    next_queue_position,
)
from .contracts import (
    DatasetInfo, JobSource, JobStatus, JobType, ResultStatus,
    TrainingContinuationCreate, TrainingContinuationOptions, TrainingUIAPIError,
)


CONTINUATION_KEY = "ui.training_continuation"
BEST_ARTIFACT = "checkpoints/best.pt"
LAST_ARTIFACT = "checkpoints/final.pt"
CONFIG_ARTIFACT = "config/train_config.yaml"
RESOLVED_CONFIG_ARTIFACT = "config/resolved_train_config.json"
_LIMIT_KEYS = {
    "additional_epochs": "train.epochs",
    "additional_time_sec": "train.max_training_time_sec",
    "early_stopping_patience": "train.early_stopping_patience",
}
_OPTION_KEYS = {
    "run_inference_after_training": POST_TRAINING_INFERENCE_CONFIG_KEY,
    "secondary_priority": SECONDARY_PRIORITY_CONFIG_KEY,
}


def can_continue_training(result: TrainingResultRow, job: JobRow | None) -> bool:
    return bool(
        result.status == ResultStatus.OK.value
        and result.mlflow_run_id
        and job is not None
        and job.type == JobType.TRAINING.value
        and job.status == JobStatus.COMPLETED.value
        and job.config
    )


def _source(session: Session, result_id: uuid.UUID) -> tuple[TrainingResultRow, JobRow]:
    result = session.get(TrainingResultRow, result_id)
    job = session.get(JobRow, result.job_id) if result is not None and result.job_id else None
    if result is None:
        raise TrainingUIAPIError("Результат обучения не найден.")
    if not can_continue_training(result, job):
        raise TrainingUIAPIError("Продолжение доступно для завершённого обучения с сохранёнными параметрами и лучшими весами.")
    return result, job


def _check_artifact(config: TrainingUIAPIConfig, run_id: str, artifact: str) -> None:
    try:
        available = get_finished_run_artifact(config.mlflow_tracking_uri, run_id, artifact)
    except MLflowAdapterError as exc:
        raise TrainingUIAPIError("Не удалось проверить сохранённые файлы обучения в MLflow.") from exc
    if available is None:
        raise TrainingUIAPIError(f"Завершённый запуск не содержит файл {artifact}. Продолжение недоступно.")


def _current_dataset(
    session: Session, result: TrainingResultRow, job: JobRow, config: TrainingUIAPIConfig,
    *, input_channels: int | None = None,
) -> DatasetInfo:
    if job.custom_dataset_id is not None:
        raise TrainingUIAPIError("Для загруженного вручную датасета выберите прежний состав: его файлы неизменяемы.")
    key = job.dataset_key or result.dataset_key or result.class_key
    dataset = find_managed_dataset(session, config, key)
    if dataset is None or not dataset.source_available or dataset.images_dir is None:
        raise TrainingUIAPIError("Актуальный датасет удалён или его источник недоступен. Выберите прежний состав.")
    if dataset.image_count == 0 and not dataset.managed:
        raise TrainingUIAPIError("Актуальный датасет пуст. Добавьте и опубликуйте снимки с разметкой.")
    expected_schema = sorted((item["id"], item["slug"]) for item in result.class_schema or [])
    schema = sorted((item.id, item.slug) for item in dataset.object_types)
    if dataset.task != result.task or schema != expected_schema:
        raise TrainingUIAPIError("Схема типов объектов актуального датасета несовместима с выбранной сетью. Нужно новое обучение либо прежний состав.")
    channels = input_channels if input_channels is not None else job.config.get("train.input_channels")
    if channels is not None and dataset.input_channels is not None and int(channels) != dataset.input_channels:
        raise TrainingUIAPIError("Число каналов актуальных снимков несовместимо с выбранной сетью. Выберите прежний состав.")
    return dataset


def continuation_options(
    session: Session, result_id: uuid.UUID, config: TrainingUIAPIConfig,
) -> TrainingContinuationOptions:
    result, job = _source(session, result_id)
    _check_artifact(config, result.mlflow_run_id, BEST_ARTIFACT)
    flat = job.config
    try:
        last_available = get_finished_run_artifact(config.mlflow_tracking_uri, result.mlflow_run_id, LAST_ARTIFACT) is not None
    except MLflowAdapterError as exc:
        raise TrainingUIAPIError("Не удалось проверить последний чекпойнт в MLflow.") from exc
    return TrainingContinuationOptions(
        additional_epochs=flat.get("train.epochs") or 20,
        additional_time_sec=flat.get("train.max_training_time_sec") or 3600,
        early_stopping_patience=flat.get("train.early_stopping_patience") or 10,
        last_checkpoint_available=last_available,
        run_inference_after_training=bool(flat.get(POST_TRAINING_INFERENCE_CONFIG_KEY, False)),
        secondary_priority=bool(flat.get(SECONDARY_PRIORITY_CONFIG_KEY, False)),
    )


def create_continuation_job(
    session: Session, result_id: uuid.UUID, request: TrainingContinuationCreate,
    config: TrainingUIAPIConfig,
    *, started_by: str | None = None,
) -> JobRow:
    # Один идентификатор отправки не создаёт вторую работу при повторе HTTP-запроса.
    dedup_key = hashlib.sha256(f"training-continuation:{result_id}:{request.request_id}".encode()).hexdigest()
    result, source = _source(session, result_id)
    session.refresh(source, with_for_update=True)
    if not can_continue_training(result, source):
        raise TrainingUIAPIError("Исходное обучение больше недоступно для продолжения.")
    # Старые клиенты без этих полей сохраняют прежнее наследование опций.
    selected_options = {
        key: bool(source.config.get(key, False)) if getattr(request, name) is None else getattr(request, name)
        for name, key in _OPTION_KEYS.items()
    }
    existing = session.scalar(select(JobRow).where(JobRow.dedup_key == dedup_key))
    if existing is not None:
        if (any(existing.config.get(key) != getattr(request, name) for name, key in _LIMIT_KEYS.items())
            or any(bool(existing.config.get(key, False)) != value for key, value in selected_options.items())
            or existing.config[CONTINUATION_KEY].get("checkpoint") != request.checkpoint
            or existing.config[CONTINUATION_KEY].get("dataset_source", "original") != request.dataset_source):
            raise TrainingUIAPIError("Параметры повторной отправки изменены. Откройте диалог заново.")
        return existing
    if selected_options[POST_TRAINING_INFERENCE_CONFIG_KEY] and effective_inference_template_row(session, source.dataset_key) is None:
        raise TrainingUIAPIError("Классу не назначен шаблон инференса. Назначьте шаблон или отключите создание псевдоразметки.")
    _check_artifact(config, result.mlflow_run_id, LAST_ARTIFACT if request.checkpoint == "last" else BEST_ARTIFACT)
    current = None
    if request.dataset_source == "current":
        with markup_read_lock(config):
            current = _current_dataset(session, result, source, config)
    job_config = copy.deepcopy(source.config)
    for transient_key in (POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY, STOP_AND_SAVE_BEST_CONFIG_KEY,
                          TRAINING_STARTED_BY_CONFIG_KEY):
        job_config.pop(transient_key, None)
    if started_by:
        job_config[TRAINING_STARTED_BY_CONFIG_KEY] = started_by
    job_config.update({key: getattr(request, name) for name, key in _LIMIT_KEYS.items()})
    job_config.update(selected_options)
    job_config[CONTINUATION_KEY] = {
        "result_id": str(result.id), "job_id": str(source.id),
        "mlflow_run_id": result.mlflow_run_id,
        "epoch": result.epoch if request.checkpoint == "best" else None,
        "checkpoint": request.checkpoint,
        "dataset_source": request.dataset_source,
        "source_dataset_version": result.dataset_version,
    }
    row = JobRow(
        type=JobType.TRAINING.value, source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.TRAINING, JobSource.MANUAL),
        dedup_key=dedup_key,
        dataset_key=current.key if current else source.dataset_key,
        dataset_version=current.version if current else result.dataset_version,
        dataset_name=current.name if current else source.dataset_name,
        training_dataset_name=current.name if current else source.training_dataset_name or source.dataset_name,
        model_name=source.model_name, architecture=source.architecture,
        tile_size=source.tile_size, custom_dataset_id=source.custom_dataset_id,
        mlflow_experiment_id=source.mlflow_experiment_id,
        mlflow_experiment_name=source.mlflow_experiment_name,
        mlflow_run_name=f"{(source.mlflow_run_name or source.model_name)[:235]} — продолжение",
        config=job_config,
    )
    session.add(row)
    session.flush()
    session.add(TrainingResultRow(
        source=JobSource.MANUAL.value, dataset_key=current.key if current else result.dataset_key,
        dataset_version=current.version if current else result.dataset_version, class_key=result.class_key,
        class_display_name=result.class_display_name, architecture=result.architecture,
        model_name=result.model_name, quality_metric=result.quality_metric,
        task=result.task, class_schema=copy.deepcopy(result.class_schema or []),
        status=ResultStatus.RUNNING.value, job_id=row.id,
    ))
    session.flush()
    return row


def _download(config: TrainingUIAPIConfig, run_id: str, artifact: str, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = download_run_artifact(
            tracking_uri=config.mlflow_tracking_uri, run_id=run_id,
            artifact_path=artifact, dst_dir=root,
        )
    except MLflowAdapterError as exc:
        raise TrainingUIAPIError(f"Не удалось восстановить файл исходного обучения: {artifact}.") from exc
    path = Path(downloaded.local_path)
    if not path.exists():
        raise TrainingUIAPIError(f"Не найден сохранённый файл исходного обучения: {artifact}.")
    return path


def build_continuation_config(
    session: Session, row: JobRow, config: TrainingUIAPIConfig, run_dir: Path,
    *, current_dataset_config: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lineage = row.config[CONTINUATION_KEY]
    run_id = lineage["mlflow_run_id"]
    artifact = LAST_ARTIFACT if lineage["checkpoint"] == "last" else BEST_ARTIFACT
    _check_artifact(config, run_id, artifact)
    root = run_dir / "continuation"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = _download(config, run_id, artifact, root / "checkpoint")
    if not checkpoint.is_file():
        raise TrainingUIAPIError("Выбранный чекпойнт не является файлом.")
    try:
        resolved_artifact = get_finished_run_artifact(config.mlflow_tracking_uri, run_id, RESOLVED_CONFIG_ARTIFACT)
        if resolved_artifact is not None:
            source_config = _download(config, run_id, RESOLVED_CONFIG_ARTIFACT, root / "config")
            payload = json.loads(source_config.read_text(encoding="utf-8"))
        else:
            source = session.get(JobRow, uuid.UUID(lineage["job_id"]))
            source_config = Path(source.tmp_path) / "run.yml" if source is not None and source.tmp_path else None
            if source_config is None or not source_config.is_file():
                source_config = _download(config, run_id, CONFIG_ARTIFACT, root / "config")
            payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, ValueError, MLflowAdapterError) as exc:
        raise TrainingUIAPIError("Не удалось прочитать параметры исходного обучения.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("train"), dict) or not isinstance(payload.get("dataset"), dict):
        raise TrainingUIAPIError("В исходном запуске отсутствуют параметры обучения или датасета.")
    # Снимок содержит фактические параметры запуска; актуальный шаблон сюда не подмешивается.
    if lineage.get("dataset_source", "original") == "current":
        source_result = session.get(TrainingResultRow, uuid.UUID(lineage["result_id"]))
        if source_result is None or current_dataset_config is None:
            raise TrainingUIAPIError("Не удалось восстановить источник продолжения обучения.")
        with markup_read_lock(config):
            current = _current_dataset(session, source_result, row, config, input_channels=payload["train"].get("input_channels"))
            # Меняются только источники данных; параметры разбиения остаются от выбранной сети.
            previous = payload["dataset"]
            snapshot = current_dataset_config(current.key)
            payload["dataset"] = {
                key: value for key, value in previous.items()
                if key not in {"images_dir", "annotations_dir", "scenes_file", "annotation_file", "hard_negative_annotation_file", "classes"}
            } | snapshot
            row.config = {**row.config, CONTINUATION_KEY: {**lineage, "dataset_version": row.dataset_version}}
    else:
        _restore_dataset(payload["dataset"], config, run_id, root)
    payload["runtime"] = {
        **payload.get("runtime", {}), "project_root": str(config.project_root),
        "scratch_root": str(run_dir / "scratch"), "logs_root": str(run_dir / "logs"),
        "cleanup_scratch_after_mlflow_log": True,
    }
    payload["train"].update(
        initial_checkpoint_uri=str(checkpoint),
        epochs=row.config["train.epochs"],
        max_training_time_sec=row.config["train.max_training_time_sec"],
        early_stopping_patience=row.config["train.early_stopping_patience"],
    )
    payload["mlflow"] = {**payload.get("mlflow", {}), "experiment_name": row.mlflow_experiment_name or "MLSystem2"}
    return payload


def _restore_dataset(dataset: dict[str, Any], config: TrainingUIAPIConfig, run_id: str, root: Path) -> None:
    destination = root / "dataset_snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    restored: Path | None = None

    def artifact_root() -> Path:
        nonlocal restored
        if restored is None:
            restored = _download(config, run_id, "dataset", root / "artifacts")
        return restored

    if dataset.get("annotations_dir"):
        source = Path(dataset["annotations_dir"])
        if not source.is_dir() or not any(source.glob("*.geojson")):
            source = artifact_root() / "per_image"
        if not source.is_dir() or not any(source.glob("*.geojson")):
            raise TrainingUIAPIError("Не сохранился исходный снимок разметки датасета.")
        for path in source.iterdir():
            if path.is_file() and (path.suffix.lower() == ".geojson" or path.name == ".mlsystem2-dataset.json"):
                shutil.copy2(path, destination / path.name)
        dataset["annotations_dir"] = str(destination)
        return
    groups = dataset.get("classes") or [dataset]
    for group in groups:
        for field, short_name in (("scenes_file", "scenes"), ("annotation_file", "annotation"), ("hard_negative_annotation_file", "hard_negative")):
            if not group.get(field):
                continue
            source = Path(group[field])
            if group is dataset:
                artifact_name = source.name
            else:
                artifact_name = f"{group['slug']}_{short_name}{source.suffix}"
            if not source.is_file():
                source = artifact_root() / artifact_name
            if not source.is_file():
                raise TrainingUIAPIError("Не сохранился исходный снимок разметки датасета.")
            target = destination / artifact_name
            shutil.copy2(source, target)
            group[field] = str(target)
    if not dataset.get("classes") and not (dataset.get("scenes_file") and dataset.get("annotation_file")):
        raise TrainingUIAPIError("В исходном запуске отсутствует снимок разметки датасета.")

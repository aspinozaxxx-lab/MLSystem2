"""Новый этап от сохранённых весов на выбранных данных того же класса."""

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
import rasterio
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlsystem2.mlflow_adapter.api import download_run_artifact, get_finished_run_artifact
from mlsystem2.mlflow_adapter.contracts import MLflowAdapterError
from mlsystem2.dataset_preparing.api import per_image_annotation_name

from ._config import TrainingUIAPIConfig
from ._dataset_catalog import dataset_class_row, find_managed_dataset
from ._markup_lock import TestSampleStorageBusy, markup_read_lock, test_sample_storage_lock
from ._models import JobRow, TrainingResultRow, TestSampleRow, TestSampleTileRow
from ._test_samples import _sample_root, _sample_row
from ._result_metadata import next_result_number
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
TRAINING_TEST_SAMPLES_KEY = "ui.training_test_samples"
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
        and result.archived_at is None
        and result.mlflow_run_id
        and job is not None
        and job.type == JobType.TRAINING.value
        and job.status == JobStatus.COMPLETED.value
        and job.config
    )


def _source(session: Session, result_id: uuid.UUID) -> tuple[TrainingResultRow, JobRow]:
    result = session.get(TrainingResultRow, result_id)
    job = session.get(JobRow, result.job_id) if result is not None and result.job_id else None
    if result is None or result.archived_at is not None:
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
    *, input_channels: int | None = None, dataset_key: str | None = None,
) -> DatasetInfo:
    if job.custom_dataset_id is not None and dataset_key is None:
        raise TrainingUIAPIError("Для загруженного вручную датасета выберите прежний состав: его файлы неизменяемы.")
    key = dataset_key or job.dataset_key or result.dataset_key or result.class_key
    dataset = find_managed_dataset(session, config, key)
    if dataset is None or not dataset.source_available or dataset.images_dir is None:
        raise TrainingUIAPIError("Актуальный датасет удалён или его источник недоступен. Выберите прежний состав.")
    if dataset_key is not None:
        source_class = dataset_class_row(session, result.dataset_key or result.class_key)
        target_class = dataset_class_row(session, dataset.key)
        if source_class is None or target_class is None or source_class.id != target_class.id:
            raise TrainingUIAPIError("Для дообучения выберите датасет того же класса, что и исходная сеть.")
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


def _training_test_sample(session: Session, result: TrainingResultRow, sample_id: uuid.UUID) -> TestSampleRow:
    sample = _sample_row(session, sample_id)
    source_class = dataset_class_row(session, result.dataset_key or result.class_key)
    sample_class = dataset_class_row(session, sample.class_key)
    if source_class is None or sample_class is None or source_class.id != sample_class.id:
        raise TrainingUIAPIError("Тестовая разметка должна принадлежать классу исходной сети.")
    expected = sorted((int(item["id"]), item["slug"]) for item in result.class_schema or [])
    actual = sorted((int(item["id"]), item["slug"]) for item in sample.class_schema or [])
    if sample.task != result.task or actual != expected:
        raise TrainingUIAPIError("Схема типов объектов тестовой разметки несовместима с выбранной сетью.")
    if not any(tile.enabled for tile in sample.tiles):
        raise TrainingUIAPIError("В тестовой разметке нет включённых снимков для обучения.")
    return sample


def _test_sample_files(
    sample: TestSampleRow, config: TrainingUIAPIConfig, input_channels: int,
) -> list[tuple[TestSampleTileRow, Path, Path]]:
    root = _sample_root(config, sample.id)
    files = []
    for tile in sorted((tile for tile in sample.tiles if tile.enabled), key=lambda tile: tile.tile_index):
        image = root / f"tile_{tile.tile_index:03d}.tif"
        annotation = image.with_suffix(".geojson")
        if not image.is_file() or not annotation.is_file():
            raise TrainingUIAPIError(f"Файлы тестового снимка {tile.tile_index} недоступны.")
        try:
            with rasterio.open(image) as raster:
                if raster.crs is None or raster.count != input_channels:
                    raise TrainingUIAPIError(f"Проекция или число каналов тестового снимка {tile.tile_index} несовместимы с сетью.")
        except rasterio.errors.RasterioIOError as exc:
            raise TrainingUIAPIError(f"Не удалось открыть тестовый снимок {tile.tile_index}.") from exc
        files.append((tile, image, annotation))
    return files


def _test_sample_snapshot(
    session: Session, result: TrainingResultRow, row: JobRow, config: TrainingUIAPIConfig,
    run_dir: Path, input_channels: int,
) -> dict[str, Any]:
    lineage = row.config[CONTINUATION_KEY]
    sample_id = uuid.UUID(lineage["test_sample_id"])
    # Не ждать публикацию, которая может блокироваться строками активного задания.
    with test_sample_storage_lock(config, sample_id, wait=False):
        sample = _training_test_sample(session, result, sample_id)
        session.refresh(sample, with_for_update=True)
        session.expire(sample, ["tiles"])
        sample = _training_test_sample(session, result, sample_id)
        files = _test_sample_files(sample, config, input_channels)
        # TIFF должны пережить очистку временных папок: прежний состав следующего
        # этапа ссылается на них, как на обычные подготовленные снимки.
        image_root = (config.stored_files_root / "training-data").resolve()
        images = (image_root / str(row.id)).resolve()
        images.relative_to(image_root)
        annotations = run_dir / "dataset_snapshot"
        images.mkdir(parents=True, exist_ok=True)
        annotations.mkdir(parents=True, exist_ok=True)
        for _tile, image, annotation in files:
            destination = images / image.name
            shutil.copy2(image, destination)
            for suffix in (".msk", ".aux.xml"):
                companion = Path(str(image) + suffix)
                if companion.is_file():
                    shutil.copy2(companion, Path(str(destination) + suffix))
            shutil.copy2(annotation, annotations / per_image_annotation_name(str(destination)))
        if sample.task == "multiclass":
            (annotations / ".mlsystem2-dataset.json").write_text(json.dumps({
                "schema_version": 1, "task": "multiclass", "classes": sample.class_schema,
            }, ensure_ascii=False), encoding="utf-8")
        revision = f"test-sample:{sample.id}:r{sample.content_revision}"
        record = {"id": str(sample.id), "name": sample.name, "revision": sample.content_revision,
                  "tile_indices": [tile.tile_index for tile, _, _ in files]}
        previous = [item for item in row.config.get(TRAINING_TEST_SAMPLES_KEY, [])
                    if (item["id"], item["revision"]) != (record["id"], record["revision"])]
        row.config = {**row.config, TRAINING_TEST_SAMPLES_KEY: [*previous, record],
                      CONTINUATION_KEY: {**lineage, "test_sample_revision": sample.content_revision,
                                        "tile_indices": record["tile_indices"], "dataset_version": revision, "data_name": sample.name}}
        row.dataset_version = revision
        row.training_dataset_name = f"Тестовая разметка «{sample.name}»"
        child = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == row.id))
        if child is not None:
            child.dataset_version = revision
        return {"images_dir": str(images), "annotations_dir": str(annotations)}


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
        class_key=(class_row.key if (class_row := dataset_class_row(session, result.dataset_key or result.class_key)) else None),
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
            or existing.config[CONTINUATION_KEY].get("dataset_source", "original") != request.dataset_source
            or existing.config[CONTINUATION_KEY].get("dataset_key") != request.dataset_key
            or existing.config[CONTINUATION_KEY].get("test_sample_id") != (str(request.test_sample_id) if request.test_sample_id else None)):
            raise TrainingUIAPIError("Параметры повторной отправки изменены. Откройте диалог заново.")
        return existing
    if selected_options[POST_TRAINING_INFERENCE_CONFIG_KEY] and effective_inference_template_row(session, request.dataset_key or source.dataset_key) is None:
        raise TrainingUIAPIError("Классу не назначен шаблон инференса. Назначьте шаблон или отключите создание псевдоразметки.")
    _check_artifact(config, result.mlflow_run_id, LAST_ARTIFACT if request.checkpoint == "last" else BEST_ARTIFACT)
    current = None
    sample = None
    if request.dataset_source in {"current", "dataset"}:
        with markup_read_lock(config):
            current = _current_dataset(session, result, source, config, dataset_key=request.dataset_key)
    elif request.dataset_source == "test_sample":
        try:
            with test_sample_storage_lock(config, request.test_sample_id, wait=False):
                sample = _training_test_sample(session, result, request.test_sample_id)
                _test_sample_files(sample, config, int(source.config.get("train.input_channels") or 4))
        except TestSampleStorageBusy as exc:
            raise TrainingUIAPIError("Публикуется тестовая разметка. Повторите запуск после публикации.") from exc
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
        "dataset_key": request.dataset_key,
        "test_sample_id": str(request.test_sample_id) if request.test_sample_id else None,
        "data_name": current.name if current else sample.name if sample else None,
        "source_dataset_version": result.dataset_version,
    }
    if sample is not None:
        job_config[CONTINUATION_KEY]["test_sample_revision"] = sample.content_revision
    target_key = current.key if current else result.dataset_key or result.class_key
    version = current.version if current else f"test-sample:{sample.id}:r{sample.content_revision}" if sample else result.dataset_version
    row = JobRow(
        type=JobType.TRAINING.value, source=JobSource.MANUAL.value,
        status=JobStatus.QUEUED.value,
        queue_position=next_queue_position(session, JobType.TRAINING, JobSource.MANUAL),
        dedup_key=dedup_key,
        dataset_key=current.key if current else source.dataset_key,
        dataset_version=version,
        dataset_name=current.name if current else source.dataset_name,
        training_dataset_name=current.name if current else f"Тестовая разметка «{sample.name}»" if sample else source.training_dataset_name or source.dataset_name,
        model_name=source.model_name, architecture=source.architecture,
        tile_size=source.tile_size, custom_dataset_id=None if current or sample else source.custom_dataset_id,
        mlflow_experiment_id=source.mlflow_experiment_id,
        mlflow_experiment_name=source.mlflow_experiment_name,
        mlflow_run_name=f"{(source.mlflow_run_name or source.model_name)[:235]} — продолжение",
        config=job_config,
    )
    session.add(row)
    session.flush()
    session.add(TrainingResultRow(
        dataset_number=next_result_number(session, target_key),
        source=JobSource.MANUAL.value, dataset_key=current.key if current else result.dataset_key,
        dataset_version=version, class_key=target_key,
        class_display_name=current.name if current else result.class_display_name, architecture=result.architecture,
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
    data_source = lineage.get("dataset_source", "original")
    if data_source in {"current", "dataset", "test_sample"}:
        source_result = session.get(TrainingResultRow, uuid.UUID(lineage["result_id"]))
        if source_result is None or (data_source != "test_sample" and current_dataset_config is None):
            raise TrainingUIAPIError("Не удалось восстановить источник продолжения обучения.")
        if data_source == "test_sample":
            snapshot = _test_sample_snapshot(session, source_result, row, config, run_dir, int(payload["train"].get("input_channels") or 4))
        else:
            with markup_read_lock(config):
                current = _current_dataset(session, source_result, row, config, input_channels=payload["train"].get("input_channels"), dataset_key=lineage.get("dataset_key"))
                snapshot = current_dataset_config(current.key)
                row.dataset_name = row.training_dataset_name = current.name
                child = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == row.id))
                if child is not None:
                    child.class_display_name = current.name
                row.config = {**row.config, CONTINUATION_KEY: {**lineage, "dataset_version": row.dataset_version, "data_name": current.name}}
        # Меняются только источники данных; параметры разбиения остаются от выбранной сети.
        payload["dataset"] = {
            key: value for key, value in payload["dataset"].items()
            if key not in {"images_dir", "annotations_dir", "scenes_file", "annotation_file", "hard_negative_annotation_file", "classes"}
        } | snapshot
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

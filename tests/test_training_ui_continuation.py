from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import rasterio
import yaml
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from shapely.geometry import box, mapping, shape
from fastapi.testclient import TestClient
from sqlalchemy import select
from mlsystem2.settings.api import load_settings
from mlsystem2.dataset_preparing.api import prepare_dataset
from mlsystem2.dataset_preparing.contracts import DatasetPreparationRequest
from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, ModelHandle
from mlsystem2.train_pipeline import _runner

from mlsystem2.training_ui_api import _training_continuation as continuation, _worker, _service
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._dataset_catalog import list_managed_datasets
from mlsystem2.training_ui_api._models import DatasetClassRow, DatasetRow, JobRow, PseudoMarkupResultRow, StoredFileRow, TestSampleRow, TestSampleTileRow, TrainingResultRow
from mlsystem2.training_ui_api._queueing import (
    POST_TRAINING_INFERENCE_CONFIG_KEY, POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
    SECONDARY_PRIORITY_CONFIG_KEY, STOP_AND_SAVE_BEST_CONFIG_KEY, TRAINING_STARTED_BY_CONFIG_KEY, is_secondary_job,
)
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api.contracts import DatasetInfo, InferenceTemplateClassUpdate, InferenceTemplateCreate, TrainingContinuationCreate, TrainingUIAPIError


@pytest.fixture
def saved_training(tmp_path, monkeypatch):
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", uuid.uuid4().hex)
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SECURE_COOKIES", "false")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.delenv("MLSYSTEM2_TRAINING_UI_USERS_JSON", raising=False)
    username = uuid.uuid4().hex
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", username)
    password = uuid.uuid4().hex
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", password)
    config = get_config()
    configure_schema(None)
    factory = create_session_factory(config)
    Base.metadata.create_all(factory.kw["bind"])
    source_dir = tmp_path / "исходный-запуск"
    data = source_dir / "dataset_snapshot"
    data.mkdir(parents=True)
    (data / "scenes.txt").write_text("Снимок-1\n", encoding="utf-8")
    (data / "annotation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    payload = {
        "runtime": {"scratch_root": str(source_dir / "scratch")},
        "dataset": {"images_dir": str(tmp_path / "images"), "scenes_file": str(data / "scenes.txt"),
                    "annotation_file": str(data / "annotation.geojson"), "val_fraction": 0.15},
        "tile_preparation": {"tile_size": 512, "stride": 256, "augmentation_level": 2, "seed": 19},
        "train": {"model_name": "smp_unet_resnet34", "epochs": 30, "batch_size": 3, "loss": "focal_dice",
                  "learning_rate": 0.00037, "weight_decay": 0.002, "early_stopping_patience": 7,
                  "max_training_time_sec": 1800, "task": "binary", "quality_metric": "pixel", "pretrained": False,
                  "device": "cpu", "input_channels": 4, "output_channels": 1},
        "mlflow": {"experiment_name": "Проверка"},
    }
    (source_dir / "run.yml").write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    flat = {"train.epochs": 30, "train.max_training_time_sec": 1800, "train.early_stopping_patience": 7,
            "train.learning_rate": 0.00037, "train.loss": "focal_dice", "train.pipeline_variant": "legacy",
            "tile_preparation.augmentation_level": 2, "train.input_channels": 4,
            "ui.run_inference_after_training": True, POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY: ["старое-задание"],
            STOP_AND_SAVE_BEST_CONFIG_KEY: True}
    with factory() as session:
        class_row = DatasetClassRow(key="образец", name="Учебный класс", technical_name="образец", imagery_type="kanopus", quality_metric="pixel")
        session.add(class_row)
        session.flush()
        template = _service.create_inference_template(session, InferenceTemplateCreate(display_name="Настройки учебного класса"), config)
        _service.assign_inference_template(session, class_row.key, InferenceTemplateClassUpdate(template_id=template.id), config)
        job = JobRow(type="training", source="manual", status="completed", queue_position=30001,
                     dataset_key="образец", dataset_version="старая-версия", dataset_name="Учебный датасет",
                     model_name="Учебная сеть", architecture="smp_unet_resnet34", tile_size=512,
                     config=flat, tmp_path=str(source_dir), mlflow_experiment_name="Проверка")
        session.add(job)
        session.flush()
        result = TrainingResultRow(source="manual", dataset_key=job.dataset_key, dataset_version=job.dataset_version,
                                   class_key=job.dataset_key, class_display_name=job.dataset_name,
                                   model_name=job.model_name, architecture=job.architecture, status="ok",
                                   job_id=job.id, epoch=12, f1_score=0.81, mlflow_run_id="исходный-run")
        session.add(result)
        session.commit()
        result_id, job_id = result.id, job.id
    available = {continuation.BEST_ARTIFACT, continuation.LAST_ARTIFACT, continuation.CONFIG_ARTIFACT}
    downloads = []

    def get_artifact(_uri, run_id, artifact):
        assert run_id == "исходный-run"
        return SimpleNamespace() if artifact in available else None

    def download(**kwargs):
        assert kwargs["run_id"] == "исходный-run"
        artifact = kwargs["artifact_path"]
        downloads.append(artifact)
        root = Path(kwargs["dst_dir"])
        root.mkdir(parents=True, exist_ok=True)
        if artifact == "dataset":
            root = root / "dataset"
            root.mkdir()
            if payload["dataset"].get("annotations_dir"):
                source = root / "per_image"
                source.mkdir()
                (source / "Снимок-1.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
                (source / ".mlsystem2-dataset.json").write_text('{"task":"multiclass","classes":[]}', encoding="utf-8")
                return SimpleNamespace(local_path=str(root))
            (root / "scenes.txt").write_text("Снимок-1\n", encoding="utf-8")
            (root / "annotation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            return SimpleNamespace(local_path=str(root))
        target = root / Path(artifact).name
        if artifact == continuation.CONFIG_ARTIFACT:
            target.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        elif artifact == continuation.RESOLVED_CONFIG_ARTIFACT:
            target.write_text(json.dumps(payload), encoding="utf-8")
        else:
            target.write_bytes(artifact.encode())
        return SimpleNamespace(local_path=str(target))

    monkeypatch.setattr(continuation, "get_finished_run_artifact", get_artifact)
    monkeypatch.setattr(continuation, "download_run_artifact", download)
    client = TestClient(create_app())
    login = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200
    yield SimpleNamespace(config=config, factory=factory, client=client, result_id=result_id, job_id=job_id,
                          payload=payload, flat=copy.deepcopy(flat), available=available, downloads=downloads,
                          source_dir=source_dir)
    client.close()


def _request(checkpoint="best", **changes):
    return {"additional_epochs": 8, "additional_time_sec": 600, "early_stopping_patience": 3,
            "checkpoint": checkpoint, "request_id": str(uuid.uuid4()), **changes}


def test_unassigned_class_can_continue_only_without_pseudo_markup(saved_training):
    env = saved_training
    with env.factory() as session:
        class_row = session.scalar(select(DatasetClassRow).where(DatasetClassRow.key == "образец"))
        _service.assign_inference_template(session, class_row.key, InferenceTemplateClassUpdate(template_id=None), env.config)
        session.commit()
    url = f"/api/v1/results/training/{env.result_id}/continue"
    rejected = env.client.post(url, json=_request(run_inference_after_training=True))
    assert rejected.status_code == 400
    assert "назначен шаблон инференса" in rejected.json()["detail"]
    allowed = env.client.post(url, json=_request(run_inference_after_training=False))
    assert allowed.status_code == 200
    assert allowed.json()["run_inference_after_training"] is False


@pytest.mark.parametrize("checkpoint", ["best", "last"])
def test_continue_api_copies_parameters_and_keeps_original(saved_training, checkpoint):
    env = saved_training
    url = f"/api/v1/results/training/{env.result_id}/continue"
    options = env.client.get(url)
    assert options.status_code == 200
    assert options.json() == {"additional_epochs": 30, "additional_time_sec": 1800,
                              "early_stopping_patience": 7, "last_checkpoint_available": True,
                              "run_inference_after_training": True, "secondary_priority": False, "class_key": "образец", "pipeline_variants": ["legacy"]}
    request = _request(checkpoint)
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    child = response.json()
    assert child["status"] == "queued" and child["type"] == "training"
    assert child["dataset_version"] == "старая-версия"
    assert child["config"]["train.learning_rate"] == env.flat["train.learning_rate"]
    assert child["config"]["train.loss"] == env.flat["train.loss"]
    assert child["config"]["train.epochs"] == 8
    assert child["run_inference_after_training"] is True
    assert POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY not in child["config"]
    assert STOP_AND_SAVE_BEST_CONFIG_KEY not in child["config"]
    assert env.client.post(url, json=request).json()["id"] == child["id"]
    assert env.client.post(url, json={**request, "additional_epochs": 9}).status_code == 400
    rows = env.client.get("/api/v1/results/datasets/образец").json()["results"]
    source_info = next(item for item in rows if item["id"] == str(env.result_id))
    child_info = next(item for item in rows if item["job_id"] == child["id"])
    assert source_info["can_continue_training"] is True
    assert child_info["can_continue_training"] is False
    assert child_info["continued_from_result_id"] == str(env.result_id)
    assert child_info["continued_from_checkpoint"] == checkpoint
    with env.factory() as session:
        assert session.get(JobRow, env.job_id).config == env.flat
        assert len(session.scalars(select(JobRow)).all()) == 2
        assert session.get(TrainingResultRow, env.result_id).f1_score == 0.81


@pytest.mark.parametrize("login,author", [("alex-alias", "Alex"), ("Aspinoza", "Aspinoza")])
def test_training_author_comes_from_session_not_request(saved_training, monkeypatch, login, author):
    env = saved_training
    password = uuid.uuid4().hex
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USERS_JSON", json.dumps([
        {"username": "Alex", "password": password, "aliases": ["alex-alias"]},
        {"username": "Aspinoza", "password": password},
    ]))
    monkeypatch.setattr(_service, "_resolve_dataset_name", lambda *args: DatasetInfo(
        key="образец", name="Учебный датасет", version="текущая-версия",
    ))
    with TestClient(create_app()) as client:
        assert client.post("/api/v1/auth/login", json={"username": login, "password": password}).status_code == 200
        response = client.post("/api/v1/training-jobs", json={
            "dataset_key": "образец", "architecture": "smp_segformer_b1",
            "mlflow_experiment_name": "Проверка автора", "mlflow_experiment_id": "1",
            "run_inference_after_training": False,
            "config": {TRAINING_STARTED_BY_CONFIG_KEY: "Подставной автор"},
        })
        assert response.status_code == 200, response.text
        job_id = response.json()["id"]
        assert response.json()["started_by"] == author
        queue = client.get("/api/v1/queues").json()["jobs"]
        assert next(item for item in queue if item["id"] == job_id)["started_by"] == author
        assert TRAINING_STARTED_BY_CONFIG_KEY not in response.json()["config"]
        rows = client.get("/api/v1/results/datasets/образец").json()["results"]
        assert next(item for item in rows if item["job_id"] == job_id)["dataset_number"] == 1
        assert next(item for item in rows if item["job_id"] == job_id)["started_by"] == author
        assert next(item for item in rows if item["id"] == str(env.result_id))["started_by"] is None
        with env.factory() as session:
            assert session.get(JobRow, uuid.UUID(job_id)).config[TRAINING_STARTED_BY_CONFIG_KEY] == author


def test_continuation_records_own_author_and_keeps_it_on_retry(saved_training):
    env = saved_training
    author = env.client.get("/api/v1/auth/me").json()["username"]
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        source.config = {**source.config, TRAINING_STARTED_BY_CONFIG_KEY: "Aspinoza"}
        session.commit()
    request = _request(run_inference_after_training=False)
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=request)
    assert response.status_code == 200, response.text
    child_id = response.json()["id"]
    assert TRAINING_STARTED_BY_CONFIG_KEY not in response.json()["config"]
    with env.factory() as session:
        repeated = continuation.create_continuation_job(session, env.result_id,
            TrainingContinuationCreate(**request), env.config, started_by="Другой пользователь")
        assert str(repeated.id) == child_id
        assert repeated.config[TRAINING_STARTED_BY_CONFIG_KEY] == author
        assert session.get(JobRow, env.job_id).config[TRAINING_STARTED_BY_CONFIG_KEY] == "Aspinoza"
        without_author = continuation.create_continuation_job(session, env.result_id,
            TrainingContinuationCreate(**_request(run_inference_after_training=False)), env.config)
        assert TRAINING_STARTED_BY_CONFIG_KEY not in without_author.config
    rows = env.client.get("/api/v1/results/datasets/образец").json()["results"]
    assert next(item for item in rows if item["id"] == str(env.result_id))["started_by"] == "Aspinoza"
    assert next(item for item in rows if item["job_id"] == child_id)["started_by"] == author


def test_last_checkpoint_is_disabled_and_rejected_when_not_saved(saved_training):
    env = saved_training
    env.available.remove(continuation.LAST_ARTIFACT)
    url = f"/api/v1/results/training/{env.result_id}/continue"
    assert env.client.get(url).json()["last_checkpoint_available"] is False
    assert env.client.post(url, json=_request("last")).status_code == 400
    assert env.client.post(url, json=_request("best")).status_code == 200


@pytest.mark.parametrize("run_inference", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
def test_continuation_can_change_pseudo_markup_and_queue_priority(saved_training, run_inference, secondary):
    env = saved_training
    source_config = {**env.flat, POST_TRAINING_INFERENCE_CONFIG_KEY: not run_inference,
                     SECONDARY_PRIORITY_CONFIG_KEY: not secondary}
    with env.factory() as session:
        session.get(JobRow, env.job_id).config = source_config
        session.commit()
    url = f"/api/v1/results/training/{env.result_id}/continue"
    options = env.client.get(url).json()
    assert options["run_inference_after_training"] is (not run_inference)
    assert options["secondary_priority"] is (not secondary)
    request = _request(run_inference_after_training=run_inference, secondary_priority=secondary)
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    child = response.json()
    assert child["run_inference_after_training"] is run_inference
    assert child["secondary_priority"] is secondary
    with env.factory() as session:
        row = session.get(JobRow, uuid.UUID(child["id"]))
        assert row.config[POST_TRAINING_INFERENCE_CONFIG_KEY] is run_inference
        assert is_secondary_job(row) is secondary
        assert session.get(JobRow, env.job_id).config == source_config
    assert env.client.post(url, json=request).json()["id"] == child["id"]
    for name in ("run_inference_after_training", "secondary_priority"):
        assert env.client.post(url, json={**request, name: not request[name]}).status_code == 400


def test_omitted_continuation_option_inherits_the_original_choice(saved_training):
    env = saved_training
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        source.config = {**source.config, SECONDARY_PRIORITY_CONFIG_KEY: True}
        session.commit()
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue",
                               json=_request(run_inference_after_training=False))
    assert response.status_code == 200, response.text
    assert response.json()["run_inference_after_training"] is False
    assert response.json()["secondary_priority"] is True


def test_continuation_keeps_pseudo_markup_files_and_their_checkpoint_epoch(saved_training, monkeypatch):
    from mlsystem2.training_ui_api import _pseudo_viewer

    env = saved_training
    original_path = env.config.stored_files_root / "прежняя-псевдоразметка.geojson"
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_bytes = b'{"type":"FeatureCollection","features":[]}'
    original_path.write_bytes(original_bytes)
    with env.factory() as session:
        stored = StoredFileRow(kind="pseudo_markup_geojson", original_name=original_path.name, path=str(original_path),
                               content_type="application/geo+json", size_bytes=len(original_bytes), object_count=0)
        session.add(stored)
        session.flush()
        pseudo = PseudoMarkupResultRow(training_result_id=env.result_id, class_key="образец",
                                      source_dataset_name="Прежние снимки", status="ok", geojson_file_id=stored.id)
        session.add(pseudo)
        session.commit()
        pseudo_id, file_id = pseudo.id, stored.id
    child = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request("last")).json()
    rows = env.client.get("/api/v1/results/datasets/образец").json()["results"]
    source = next(item for item in rows if item["id"] == str(env.result_id))
    continued = next(item for item in rows if item["job_id"] == child["id"])
    assert source["pseudo_markup_results"][0]["id"] == str(pseudo_id)
    assert source["pseudo_markup_results"][0]["checkpoint_epoch"] == 12
    assert continued["pseudo_markup_results"] == []
    with env.factory() as session:
        assert session.get(PseudoMarkupResultRow, pseudo_id).training_result_id == env.result_id
        assert session.get(StoredFileRow, file_id).path == str(original_path)
        assert session.get(TrainingResultRow, env.result_id).mlflow_run_id == "исходный-run"
    assert original_path.read_bytes() == original_bytes
    monkeypatch.setattr(_pseudo_viewer, "_result_images", lambda *_args: ([], []))
    view = env.client.get(f"/api/v1/results/pseudo-markup/{pseudo_id}/view")
    assert view.status_code == 200, view.text
    assert view.json()["checkpoint_epoch"] == 12


@pytest.mark.parametrize("field,value", [("additional_epochs", 0), ("additional_epochs", 1.5),
    ("additional_time_sec", -1), ("early_stopping_patience", True), ("checkpoint", "произвольный-путь"),
    ("dataset_source", "произвольный-датасет")])
def test_continue_validates_only_allowed_parameters(saved_training, field, value):
    response = saved_training.client.post(f"/api/v1/results/training/{saved_training.result_id}/continue", json=_request(**{field: value}))
    assert response.status_code == 422


def test_continue_requires_login_and_successful_native_training(saved_training):
    env = saved_training
    url = f"/api/v1/results/training/{env.result_id}/continue"
    with env.factory() as session:
        session.get(JobRow, env.job_id).status = "running"
        session.commit()
    assert env.client.post(url, json=_request()).status_code == 400
    env.client.post("/api/v1/auth/logout")
    assert env.client.get(url).status_code == 401
    assert env.client.post(url, json=_request()).status_code == 401


@pytest.mark.parametrize("checkpoint", ["best", "last"])
@pytest.mark.parametrize("restore_from_mlflow", [False, True])
def test_worker_uses_selected_weights_original_data_and_full_parameters(saved_training, checkpoint, restore_from_mlflow):
    env = saved_training
    original = copy.deepcopy(env.payload)
    if restore_from_mlflow:
        env.available.add(continuation.RESOLVED_CONFIG_ARTIFACT)
        # Сохранённые пути недоступны; восстановление идёт из артефактов, не из нового датасета.
        env.payload["dataset"]["scenes_file"] = str(env.source_dir / "удалённые" / "scenes.txt")
        env.payload["dataset"]["annotation_file"] = str(env.source_dir / "удалённые" / "annotation.geojson")
        env.payload["train"]["learning_rate"] = 0.00019
    with env.factory() as session:
        child = continuation.create_continuation_job(session, env.result_id, TrainingContinuationCreate(**_request(checkpoint)), env.config)
        run_dir = env.config.scratch_root / "jobs" / str(child.id)
        payload = _worker._build_training_config(session, child, env.config, run_dir)
        assert payload["train"]["epochs"] == 8
        assert payload["train"]["max_training_time_sec"] == 600
        assert payload["train"]["early_stopping_patience"] == 3
        selected_artifact = continuation.LAST_ARTIFACT if checkpoint == "last" else continuation.BEST_ARTIFACT
        assert Path(payload["train"]["initial_checkpoint_uri"]).read_bytes() == selected_artifact.encode()
        expected_train = copy.deepcopy(env.payload["train"] if restore_from_mlflow else original["train"])
        for key in ("epochs", "max_training_time_sec", "early_stopping_patience", "initial_checkpoint_uri"):
            payload["train"].pop(key, None)
            expected_train.pop(key, None)
        assert payload["train"] == expected_train
        assert payload["tile_preparation"] == original["tile_preparation"]
        assert payload["dataset"]["val_fraction"] == 0.15
        assert Path(payload["dataset"]["scenes_file"]).read_text(encoding="utf-8") == "Снимок-1\n"
        assert Path(payload["dataset"]["annotation_file"]).is_file()
        assert selected_artifact in env.downloads
        assert continuation.RESOLVED_CONFIG_ARTIFACT in env.downloads if restore_from_mlflow else continuation.RESOLVED_CONFIG_ARTIFACT not in env.downloads
        assert payload["runtime"]["scratch_root"] == str(run_dir / "scratch")


def test_worker_never_starts_fresh_if_checkpoint_disappears(saved_training):
    env = saved_training
    with env.factory() as session:
        child = continuation.create_continuation_job(session, env.result_id, TrainingContinuationCreate(**_request()), env.config)
        env.available.remove(continuation.BEST_ARTIFACT)
        with pytest.raises(TrainingUIAPIError, match="Продолжение недоступно"):
            _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
    assert not env.downloads


def test_worker_restores_historical_config_from_mlflow(saved_training):
    env = saved_training
    (env.source_dir / "run.yml").unlink()
    with env.factory() as session:
        child = continuation.create_continuation_job(session, env.result_id, TrainingContinuationCreate(**_request()), env.config)
        payload = _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
    assert continuation.CONFIG_ARTIFACT in env.downloads
    assert payload["train"]["learning_rate"] == env.payload["train"]["learning_rate"]
    assert payload["tile_preparation"]["seed"] == 19


def test_worker_restores_per_image_annotations_and_manifest(saved_training):
    env = saved_training
    env.available.add(continuation.RESOLVED_CONFIG_ARTIFACT)
    env.payload["dataset"].pop("scenes_file")
    env.payload["dataset"].pop("annotation_file")
    env.payload["dataset"]["annotations_dir"] = str(env.source_dir / "удалённые")
    with env.factory() as session:
        child = continuation.create_continuation_job(session, env.result_id, TrainingContinuationCreate(**_request()), env.config)
        payload = _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
    restored = Path(payload["dataset"]["annotations_dir"])
    assert (restored / "Снимок-1.geojson").is_file()
    assert json.loads((restored / ".mlsystem2-dataset.json").read_text(encoding="utf-8")) == {"task": "multiclass", "classes": []}
    assert "dataset" in env.downloads


@pytest.mark.parametrize("variant", ["next_gen2", "object_f1"])
@pytest.mark.parametrize("checkpoint", ["best", "last"])
def test_segformer_continuation_reaches_checkpoint_loader_with_fixed_profile(saved_training, variant, checkpoint):
    env = saved_training
    env.payload["dataset"]["val_fraction"] = 0.2
    env.payload["tile_preparation"].update(context=0, augmentation_level=3, seed=42)
    env.payload["train"].update(
        pipeline_variant=variant, model_name="smp_segformer_b0", batch_size=16,
        quality_metric="objects" if variant == "object_f1" else "pixel",
        loss="cross_entropy_tversky", learning_rate=0.0001, weight_decay=0.01,
        threshold=0.5, tversky_alpha=0.75, tversky_beta=0.25,
    )
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        source.architecture = "smp_segformer_b0"
        source.config = {**source.config, "train.pipeline_variant": variant}
        result = session.get(TrainingResultRow, env.result_id)
        result.architecture = source.architecture
        result.quality_metric = env.payload["train"]["quality_metric"]
        child = continuation.create_continuation_job(session, env.result_id, TrainingContinuationCreate(**_request(checkpoint)), env.config)
        run_dir = env.config.scratch_root / str(child.id)
        payload = _worker._build_training_config(session, child, env.config, run_dir)
    run_path = run_dir / "run.yml"
    run_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    settings = load_settings(env.config.training_settings_path, run_path)
    assert settings.train.initial_checkpoint_uri.endswith("final.pt" if checkpoint == "last" else "best.pt")
    assert settings.train.epochs == 8 and settings.train.batch_size == 16
    assert settings.train.learning_rate == 0.0001
    model = ModelHandle(spec=_runner._model_spec(settings), model=object())
    loaded_requests = []

    def load(request):
        loaded_requests.append(request)
        return LoadedCheckpoint(model=model, artifact=CheckpointArtifact(uri=request.checkpoint_uri, format="torch_pt"))

    def create(_spec):
        pytest.fail("Продолжение не должно создавать сеть с новыми весами")

    restored = _runner._load_or_create_model(settings, SimpleNamespace(load_checkpoint=load, create_model=create))
    assert restored is model
    assert loaded_requests[0].checkpoint_uri == settings.train.initial_checkpoint_uri
    assert loaded_requests[0].model_spec is None


@pytest.fixture(params=["per_image", "legacy"])
def live_training(saved_training, request):
    env = saved_training
    folder = env.config.mlmarkup_root / "Учебный класс" / "main"
    folder.mkdir(parents=True)
    images = env.config.images_root / "kanopus" / "учебные"
    images.mkdir(parents=True)

    def image(name, left):
        with rasterio.open(images / f"{name}.tif", "w", driver="GTiff", width=16, height=16,
                           count=4, dtype="uint8", crs="EPSG:3857", transform=from_origin(left, 16, 1, 1), ALPHA="NO") as raster:
            raster.write(np.full((4, 16, 16), 40, dtype=np.uint8))
            raster.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)
            raster.descriptions = ("Red", "Green", "Blue", "NIR")

    def annotation(path, geometries):
        path.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "properties": {"_mlsystem2_role": "positive"}, "geometry": mapping(geometry)}
                         for geometry in geometries]}), encoding="utf-8")

    image("scene_a", 0)
    image("scene_removed", 200)
    original = box(2, 2, 4, 4)
    updated = box(8, 8, 12, 12)
    if request.param == "per_image":
        annotation(folder / "учебные_scene_a.geojson", [original])
        annotation(folder / "учебные_scene_removed.geojson", [box(202, 2, 204, 4)])
    else:
        (folder / "scenes.txt").write_text("scene_a\nscene_removed\n", encoding="utf-8")
        annotation(folder / "annotation.geojson", [original, box(202, 2, 204, 4)])
    with env.factory() as session:
        dataset = next(item for item in list_managed_datasets(session, env.config, include_custom=False) if item.source_path == "Учебный класс/main")
        assert dataset.source_available and dataset.images_dir, dataset.model_dump()
        source = session.get(JobRow, env.job_id)
        result = session.get(TrainingResultRow, env.result_id)
        source.dataset_key = result.dataset_key = result.class_key = dataset.key
        source.dataset_name = dataset.name
        if request.param == "per_image":
            Path(env.payload["dataset"]["scenes_file"]).unlink()
            Path(env.payload["dataset"]["annotation_file"]).unlink()
        env.payload["dataset"] = {**_worker._dataset_config(session, source, env.config, env.source_dir), "val_fraction": 0.15}
        session.commit()
        env.original_version = result.dataset_version
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")

    def change_live_dataset(geometry=updated):
        image("scene_b", 100)
        if request.param == "per_image":
            annotation(folder / "учебные_scene_a.geojson", [geometry])
            annotation(folder / "учебные_scene_b.geojson", [box(103, 3, 105, 5)])
            (folder / "учебные_scene_removed.geojson").unlink(missing_ok=True)
        else:
            (folder / "scenes.txt").write_text("scene_a\nscene_b\n", encoding="utf-8")
            annotation(folder / "annotation.geojson", [geometry, box(103, 3, 105, 5)])

    env.change_live_dataset = change_live_dataset
    env.original_geometry, env.updated_geometry = original, updated
    env.live_folder = folder
    yield env


@pytest.mark.parametrize("dataset_source", ["original", "current"])
def test_continuation_uses_actual_scene_list_and_edited_geometry(live_training, dataset_source):
    env = live_training
    url = f"/api/v1/results/training/{env.result_id}/continue"
    request = _request(dataset_source=dataset_source, run_inference_after_training=False)
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    child_id = uuid.UUID(response.json()["id"])
    # Правки после постановки в очередь должны попасть только в актуальный режим.
    env.change_live_dataset()
    assert env.client.post(url, json=request).json()["id"] == str(child_id)
    other_source = "current" if dataset_source == "original" else "original"
    assert env.client.post(url, json={**request, "dataset_source": other_source}).status_code == 400
    with env.factory() as session:
        child = session.get(JobRow, child_id)
        child.config = {**child.config, "dataset.images_dir": str(env.source_dir / "чужие-снимки")}
        payload = _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
        prepared = prepare_dataset(DatasetPreparationRequest(
            **payload["dataset"], expected_band_count=4, expected_dtype="uint8",
            expected_band_names=["RED", "GRN", "BLU", "NIR"],
        ))
        assert prepared.dataset is not None, prepared.report
        stems = {Path(scene.image_path).stem for scene in prepared.dataset.scenes}
        assert stems == {"scene_a", "scene_b" if dataset_source == "current" else "scene_removed"}
        annotation_path = (next(Path(scene.annotation_file) for scene in prepared.dataset.scenes if Path(scene.image_path).stem == "scene_a")
                           if payload["dataset"].get("annotations_dir") else Path(payload["dataset"]["annotation_file"]))
        geometry = shape(json.loads(annotation_path.read_text(encoding="utf-8"))["features"][0]["geometry"])
        assert geometry.equals(env.updated_geometry if dataset_source == "current" else env.original_geometry)
        frozen = annotation_path.read_bytes()
        env.change_live_dataset(box(9, 9, 10, 10))
        assert annotation_path.read_bytes() == frozen
        assert payload["train"]["learning_rate"] == env.payload["train"]["learning_rate"]
        assert payload["tile_preparation"] == env.payload["tile_preparation"]
        assert Path(payload["train"]["initial_checkpoint_uri"]).read_bytes() == continuation.BEST_ARTIFACT.encode()
        assert child.dataset_version == session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == child_id)).dataset_version
        assert (child.dataset_version != env.original_version) is (dataset_source == "current")
        assert session.get(TrainingResultRow, env.result_id).dataset_version == env.original_version
        session.commit()
    results = env.client.get(f"/api/v1/results/datasets/{response.json()['dataset_key']}").json()["results"]
    assert next(item for item in results if item["job_id"] == str(child_id))["continued_from_dataset_source"] == dataset_source


def test_current_continuation_rejects_missing_dataset_but_original_still_works(saved_training):
    env = saved_training
    url = f"/api/v1/results/training/{env.result_id}/continue"
    response = env.client.post(url, json=_request(dataset_source="current"))
    assert response.status_code == 400
    assert "источник недоступен" in response.json()["detail"]
    assert env.client.post(url, json=_request(dataset_source="original")).status_code == 200


def test_current_continuation_rechecks_schema_when_job_starts(live_training, monkeypatch):
    env = live_training
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(dataset_source="current", run_inference_after_training=False))
    assert response.status_code == 200, response.text
    find = continuation.find_managed_dataset
    monkeypatch.setattr(continuation, "find_managed_dataset", lambda *args, **kwargs: find(*args, **kwargs).model_copy(update={"task": "multiclass"}))
    with env.factory() as session:
        child = session.get(JobRow, uuid.UUID(response.json()["id"]))
        with pytest.raises(TrainingUIAPIError, match="Схема типов"):
            _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))


def test_explicit_current_dataset_never_falls_back_to_another_name(live_training, monkeypatch):
    env = live_training
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        monkeypatch.setattr(_worker, "find_managed_dataset", lambda *_a, **_k: None)
        monkeypatch.setattr(_worker, "list_managed_datasets", lambda *_a, **_k: pytest.fail("Подмена другим одноимённым датасетом запрещена"))
        with pytest.raises(RuntimeError, match="Датасет не найден"):
            _worker._dataset_config(session, source, env.config, env.source_dir / "проверка", dataset_key=source.dataset_key, use_dataset_images=True)


@pytest.mark.parametrize("change", ["task", "schema", "channels"])
def test_current_continuation_rejects_incompatible_dataset(live_training, monkeypatch, change):
    env = live_training
    find = continuation.find_managed_dataset

    def changed(*args, **kwargs):
        dataset = find(*args, **kwargs)
        if change == "task":
            return dataset.model_copy(update={"task": "multiclass"})
        if change == "schema":
            from mlsystem2.training_ui_api.contracts import DatasetObjectType
            return dataset.model_copy(update={"object_types": [DatasetObjectType(id=1, slug="другой", name="Другой", color="#112233")]})
        return dataset.model_copy(update={"input_channels": 3})

    monkeypatch.setattr(continuation, "find_managed_dataset", changed)
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(dataset_source="current", run_inference_after_training=False))
    assert response.status_code == 400
    assert "несовместим" in response.json()["detail"]
    with env.factory() as session:
        assert len(session.scalars(select(JobRow)).all()) == 1


def _another_dataset(env):
    folder = env.live_folder.parent / "дополнительный"
    shutil.copytree(env.live_folder, folder)
    with env.factory() as session:
        dataset = next(item for item in list_managed_datasets(session, env.config, include_custom=False)
                       if item.source_path == "Учебный класс/дополнительный")
        session.commit()
    return dataset, folder


def test_continuation_on_another_dataset_uses_target_and_its_result_number(live_training):
    env = live_training
    target, folder = _another_dataset(env)
    request = _request(dataset_source="dataset", dataset_key=target.key, run_inference_after_training=False)
    url = f"/api/v1/results/training/{env.result_id}/continue"
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["dataset_key"] == target.key
    assert env.client.post(url, json=request).json()["id"] == response.json()["id"]
    assert env.client.post(url, json={**request, "dataset_key": "другой"}).status_code == 400
    child_id = uuid.UUID(response.json()["id"])
    env.change_live_dataset()
    with env.factory() as session:
        child = session.get(JobRow, child_id)
        payload = _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
        prepared = prepare_dataset(DatasetPreparationRequest(**payload["dataset"], expected_band_count=4))
        assert prepared.dataset is not None, prepared.report
        assert {Path(scene.image_path).stem for scene in prepared.dataset.scenes} == {"scene_a", "scene_removed"}
        assert payload["dataset"]["val_fraction"] == env.payload["dataset"]["val_fraction"]
        assert payload["train"]["learning_rate"] == env.payload["train"]["learning_rate"]
        assert Path(payload["train"]["initial_checkpoint_uri"]).read_bytes() == continuation.BEST_ARTIFACT.encode()
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == child_id))
        assert result.class_key == target.key and result.dataset_key == target.key
        assert result.dataset_number == 1 and result.class_display_name == target.name
        assert result.dataset_version == child.dataset_version
        assert session.get(TrainingResultRow, env.result_id).dataset_version == env.original_version
        session.commit()
    info = env.client.get(f"/api/v1/results/datasets/{target.key}").json()["results"][0]
    assert info["continued_from_dataset_source"] == "dataset"
    assert info["continued_from_data_name"] == target.name
    assert folder.exists()


@pytest.mark.parametrize("change", ["class", "deleted"])
def test_selected_dataset_is_rechecked_at_start(live_training, change):
    env = live_training
    target, _ = _another_dataset(env)
    request = _request(dataset_source="dataset", dataset_key=target.key, run_inference_after_training=False)
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=request)
    assert response.status_code == 200, response.text
    with env.factory() as session:
        target_row = session.scalar(select(DatasetRow).where(DatasetRow.key == target.key))
        if change == "class":
            other = DatasetClassRow(key="другой-класс", name="Другой класс", technical_name="другой-класс", imagery_type="kanopus")
            session.add(other)
            session.flush()
            target_row.class_id = other.id
        else:
            from datetime import datetime, timezone
            target_row.deleted_at = datetime.now(timezone.utc)
        session.commit()
        child = session.get(JobRow, uuid.UUID(response.json()["id"]))
        with pytest.raises(TrainingUIAPIError, match="класса|недоступен"):
            _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
    rejected = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json={**request, "request_id": str(uuid.uuid4())})
    assert rejected.status_code == 400


@pytest.fixture(params=[3, 4])
def training_sample(saved_training, request):
    env = saved_training
    channels = request.param
    sample_id = uuid.uuid4()
    root = env.config.stored_files_root / "test-samples" / str(sample_id)
    root.mkdir(parents=True)
    for index in (1, 7):
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(root / f"tile_{index:03d}.tif", "w", driver="GTiff", width=16, height=16,
                               count=channels, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 16, 1, 1), ALPHA="NO") as image:
                image.write(np.full((channels, 16, 16), 40, dtype=np.uint8))
                image.descriptions = ("RED", "GRN", "BLU", "NIR")[:channels]
                image.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.undefined)[:channels]
                mask = np.full((16, 16), 255, dtype=np.uint8)
                mask[:2] = 0
                image.write_mask(mask)
        (root / f"tile_{index:03d}.geojson").write_text(json.dumps({
            "type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [{"type": "Feature", "id": "эталон", "properties": {"_mlsystem2_role": "positive"},
                          "geometry": mapping(box(4, 4, 10, 10))}],
        }), encoding="utf-8")
        from PIL import Image
        reference = np.zeros((16, 16), dtype=np.uint8)
        reference[6:12, 4:10] = 1
        Image.fromarray(reference).save(root / f"tile_{index:03d}_mask.png")
    env.payload["train"]["input_channels"] = channels
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        source.config = {**source.config, "train.input_channels": channels}
        session.scalar(select(DatasetClassRow)).imagery_type = "ortho" if channels == 3 else "kanopus"
        sample = TestSampleRow(id=sample_id, name="Уточнённый эталон", dataset_key="образец", dataset_name="Учебный датасет",
            class_key="образец", class_name="Учебный класс", dataset_short_name="main", task="binary", class_schema=[],
            tile_width=16, tile_height=16, image_count=3, requested_object_count=2, actual_object_count=2, territory_count=1)
        session.add(sample)
        session.flush()
        for index in (1, 7, 9):
            session.add(TestSampleTileRow(test_sample_id=sample_id, tile_index=index, source_name=f"Снимок {index}",
                territory="Территория", object_count=1 if index != 9 else 0, enabled=index != 9))
        session.commit()
    env.sample_id, env.sample_root, env.channels = sample_id, root, channels
    yield env


def test_continuation_on_test_sample_freezes_current_enabled_reference_and_valid_mask(training_sample):
    env = training_sample
    url = f"/api/v1/results/training/{env.result_id}/continue"
    request = _request(dataset_source="test_sample", test_sample_id=str(env.sample_id), run_inference_after_training=False)
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    assert env.client.post(url, json=request).json()["id"] == response.json()["id"]
    assert env.client.post(url, json={**request, "test_sample_id": str(uuid.uuid4())}).status_code == 400
    with env.factory() as session:
        sample = session.get(TestSampleRow, env.sample_id)
        sample.name = "Эталон после правки"
        sample.content_revision = 2
        next(tile for tile in sample.tiles if tile.tile_index == 7).enabled = False
        session.commit()
        child = session.get(JobRow, uuid.UUID(response.json()["id"]))
        payload = _worker._build_training_config(session, child, env.config, env.config.scratch_root / str(child.id))
        prepared = prepare_dataset(DatasetPreparationRequest(**payload["dataset"], expected_band_count=env.channels))
        assert prepared.dataset is not None, prepared.report
        assert len(prepared.dataset.scenes) == 1
        scene = prepared.dataset.scenes[0]
        assert Path(scene.image_path).stem == "tile_001"
        assert Path(scene.annotation_file).read_bytes() == (env.sample_root / "tile_001.geojson").read_bytes()
        with rasterio.open(scene.image_path) as frozen, rasterio.open(env.sample_root / "tile_001.tif") as original:
            np.testing.assert_array_equal(frozen.dataset_mask(), original.dataset_mask())
            np.testing.assert_array_equal(frozen.read(), original.read())
        frozen_image, frozen_markup = Path(scene.image_path).read_bytes(), Path(scene.annotation_file).read_bytes()
        assert child.config[continuation.CONTINUATION_KEY]["test_sample_revision"] == 2
        assert child.config[continuation.CONTINUATION_KEY]["tile_indices"] == [1]
        assert child.dataset_version == f"test-sample:{env.sample_id}:r2"
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == child.id))
        assert result.dataset_key == "образец" and result.dataset_version == child.dataset_version
        assert payload["train"]["learning_rate"] == 1e-4
        assert payload["tile_preparation"] == {**env.payload["tile_preparation"], "augmentation_level": 0}
        evaluation = json.loads(Path(payload["train"]["test_package_fit_config"]).read_text(encoding="utf-8"))
        assert evaluation["test_sample_revision"] == 2
        assert [tile["index"] for tile in evaluation["tiles"]] == [1]
        assert Path(evaluation["tiles"][0]["mask_path"]).is_file()
        assert evaluation["inference_template_id"] is not None
        assert Path(payload["train"]["initial_checkpoint_uri"]).read_bytes() == continuation.BEST_ARTIFACT.encode()
        session.commit()
        session.delete(sample)
        session.commit()
    assert env.sample_root.resolve().is_relative_to(env.config.stored_files_root.resolve())
    shutil.rmtree(env.sample_root)
    assert Path(scene.image_path).read_bytes() == frozen_image
    assert Path(scene.annotation_file).read_bytes() == frozen_markup
    infos = env.client.get("/api/v1/results/datasets/образец").json()["results"]
    info = next(item for item in infos if item["job_id"] == response.json()["id"])
    assert info["continued_from_data_name"] == "Эталон после правки"
    assert info["training_test_sample_names"] == ["Эталон после правки"]
    assert info["evaluation_scope"] == "training_package"


@pytest.mark.parametrize("historical", [False, True])
def test_explicit_and_historical_split_keep_old_test_sample_training(training_sample, historical):
    env = training_sample
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), test_sample_training_mode="split",
        run_inference_after_training=False))
    assert response.status_code == 200, response.text
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        if historical:
            lineage = dict(job.config[continuation.CONTINUATION_KEY])
            lineage.pop("test_sample_training_mode")
            job.config = {**job.config, continuation.CONTINUATION_KEY: lineage}
        payload = _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        assert "test_package_fit_config" not in payload["train"]
        assert payload["train"]["learning_rate"] == env.payload["train"]["learning_rate"]
        assert payload["tile_preparation"] == env.payload["tile_preparation"]


def test_return_from_fitting_to_split_restores_regular_profile(training_sample):
    env = training_sample
    previous = env.config.stored_files_root / "прежняя-подгонка.json"
    previous.write_text(json.dumps({"original_learning_rate": 0.00037, "original_augmentation_level": 2}), encoding="utf-8")
    env.payload["train"].update(test_package_fit_config=str(previous), learning_rate=1e-5)
    env.payload["tile_preparation"]["augmentation_level"] = 0
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), test_sample_training_mode="split",
        run_inference_after_training=False))
    assert response.status_code == 200, response.text
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        payload = _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        assert "test_package_fit_config" not in payload["train"]
        assert payload["train"]["learning_rate"] == 0.00037
        assert payload["tile_preparation"]["augmentation_level"] == 2


@pytest.mark.parametrize("change", ["class", "schema", "channels", "disabled", "missing"])
def test_test_sample_incompatible_inputs_are_rejected_before_queue(training_sample, change):
    env = training_sample
    with env.factory() as session:
        sample = session.get(TestSampleRow, env.sample_id)
        if change == "class":
            sample.class_key = "другой"
        elif change == "schema":
            sample.task = "multiclass"
        elif change == "channels":
            source = session.get(JobRow, env.job_id)
            source.config = {**source.config, "train.input_channels": 3 if env.channels == 4 else 4}
        elif change == "disabled":
            for tile in sample.tiles:
                tile.enabled = False
        else:
            (env.sample_root / "tile_001.geojson").unlink()
        session.commit()
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), run_inference_after_training=False))
    assert response.status_code == 400, response.text
    with env.factory() as session:
        assert len(session.scalars(select(JobRow)).all()) == 1


@pytest.mark.parametrize("selection", [{"dataset_source": "dataset"}, {"dataset_source": "test_sample"},
    {"dataset_source": "original", "dataset_key": "образец"},
    {"dataset_source": "dataset", "dataset_key": "образец", "test_sample_id": str(uuid.uuid4())}])
def test_continuation_requires_exact_data_selection(saved_training, selection):
    env = saved_training
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(**selection))
    assert response.status_code == 422


def test_test_sample_provenance_is_inherited_by_further_stages(training_sample, monkeypatch):
    env = training_sample
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), run_inference_after_training=False))
    assert response.status_code == 200, response.text
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        job.status = "completed"
        child = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        child.status = "ok"
        child.mlflow_run_id = "дочерний-run"
        session.commit()
        monkeypatch.setattr(continuation, "get_finished_run_artifact", lambda *_args: SimpleNamespace())
        next_job = continuation.create_continuation_job(session, child.id, TrainingContinuationCreate(**_request(
            run_inference_after_training=False)), env.config)
        assert next_job.config[continuation.TRAINING_TEST_SAMPLES_KEY] == job.config[continuation.TRAINING_TEST_SAMPLES_KEY]
        assert next_job.config[continuation.TRAINING_TEST_SAMPLES_KEY][0]["id"] == str(env.sample_id)


def test_publication_busy_keeps_test_sample_continuation_in_queue(training_sample, monkeypatch):
    from contextlib import contextmanager
    from mlsystem2.training_ui_api._markup_lock import TestSampleStorageBusy

    env = training_sample
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), run_inference_after_training=False))
    assert response.status_code == 200, response.text

    @contextmanager
    def busy(*_args, **_kwargs):
        raise TestSampleStorageBusy("Публикуется разметка")
        yield

    monkeypatch.setattr(continuation, "test_sample_storage_lock", busy)
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        _worker._start_training_job(session, job, env.config,
            popen_factory=lambda *_args, **_kwargs: pytest.fail("Процесс не должен запускаться во время публикации"))
        assert job.status == "queued" and job.tmp_path is None and job.error is None


def test_multiclass_test_sample_retains_schema_and_object_ids(training_sample):
    env = training_sample
    schema = [{"id": index, "slug": slug, "name": name, "color": color, "priority": index}
              for index, slug, name, color in [(1, "water", "Вода", "#00aaff"), (2, "land", "Суша", "#ffaa00")]]
    env.payload["train"].update(task="multiclass", output_channels=3)
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")
    with env.factory() as session:
        sample = session.get(TestSampleRow, env.sample_id)
        sample.task, sample.class_schema = "multiclass", schema
        result = session.get(TrainingResultRow, env.result_id)
        result.task, result.class_schema = "multiclass", schema
        session.commit()
    for path in env.sample_root.glob("*.geojson"):
        document = json.loads(path.read_text(encoding="utf-8"))
        document.update(_mlsystem2_schema_version=1, _mlsystem2_task="multiclass", _mlsystem2_classes=schema)
        document["features"][0]["properties"].update(_mlsystem2_origin_key="исходный-объект", _mlsystem2_class="water")
        path.write_text(json.dumps(document), encoding="utf-8")
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        dataset_source="test_sample", test_sample_id=str(env.sample_id), run_inference_after_training=False))
    assert response.status_code == 200, response.text
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        payload = _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        prepared = prepare_dataset(DatasetPreparationRequest(**payload["dataset"], expected_band_count=env.channels))
        assert prepared.dataset is not None, prepared.report
        assert prepared.dataset.format == "per_image_multiclass"
        assert [item.slug for item in prepared.dataset.classes] == ["water", "land"]
        annotation = json.loads(Path(prepared.dataset.scenes[0].annotation_file).read_text(encoding="utf-8"))
        assert annotation["features"][0]["id"] == "эталон"


def _use_segformer_pipeline(env, variant="next_gen2"):
    profile = continuation._pipeline_profile(SimpleNamespace(config={"tile_preparation.tile_size": 512}, tile_size=512),
        SimpleNamespace(architecture="smp_segformer_b0", task="binary"), variant)
    for key, value in profile.items():
        section, _, field = key.partition(".")
        if section in {"train", "tile_preparation", "next_gen"} or key == "dataset.val_fraction":
            env.payload.setdefault(section, {})[field] = value
    env.payload["train"]["model_name"] = "smp_segformer_b0"
    (env.source_dir / "run.yml").write_text(yaml.safe_dump(env.payload, allow_unicode=True), encoding="utf-8")
    with env.factory() as session:
        source = session.get(JobRow, env.job_id)
        source.architecture = "smp_segformer_b0"
        source.config = {**source.config, **profile}
        result = session.get(TrainingResultRow, env.result_id)
        result.architecture = source.architecture
        session.commit()


def test_continuation_changes_compatible_pipeline_and_uses_frozen_profile(saved_training):
    env = saved_training
    _use_segformer_pipeline(env)
    url = f"/api/v1/results/training/{env.result_id}/continue"
    options = env.client.get(url).json()
    assert "next_gen4" in options["pipeline_variants"]
    assert "object_f1_v2" not in options["pipeline_variants"]
    assert "next_gen4_consistency_v7" not in options["pipeline_variants"]
    request = _request(pipeline_variant="next_gen4", run_inference_after_training=False)
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    assert env.client.post(url, json=request).json()["id"] == response.json()["id"]
    assert env.client.post(url, json={**request, "pipeline_variant": "next_gen3"}).status_code == 400
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        payload = _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        assert payload["train"]["pipeline_variant"] == "next_gen4"
        assert payload["train"]["initial_checkpoint_weights_only"] is True
        assert payload["tile_preparation"]["stride"] == payload["tile_preparation"]["tile_size"] == 512
        assert payload["train"]["epochs"] == request["additional_epochs"]
        assert payload["train"]["max_training_time_sec"] == request["additional_time_sec"]
        assert Path(payload["dataset"]["annotation_file"]).is_file()
        assert session.get(JobRow, env.job_id).config["train.pipeline_variant"] == "next_gen2"
        run_path = env.config.scratch_root / str(job.id) / "run.yml"
        run_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        settings = load_settings(env.config.training_settings_path, run_path)
        target = ModelHandle(spec=_runner._model_spec(settings), model=object())
        requests, transfers = [], []

        def load(request):
            requests.append(request)
            return LoadedCheckpoint(model=target, artifact=CheckpointArtifact(uri=request.checkpoint_uri, format="torch_pt"))

        def initialize(source, spec, **kwargs):
            transfers.append((source, spec, kwargs))
            return target

        assert _runner._load_or_create_model(settings, SimpleNamespace(load_checkpoint=load, initialize_model=initialize)) is target
        assert requests[0].model_spec is None
        assert transfers[0][1].parameters["pipeline_variant"] == "next_gen4"


@pytest.mark.parametrize("variant", ["object_f1", "object_f1_v2", "legacy", "next_gen4_consistency_v2", "произвольный"])
def test_incompatible_pipeline_is_rejected_before_queue(saved_training, variant):
    env = saved_training
    _use_segformer_pipeline(env)
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue", json=_request(
        pipeline_variant=variant, run_inference_after_training=False))
    assert response.status_code == 400 and "несовместим" in response.json()["detail"]
    with env.factory() as session:
        assert len(session.scalars(select(JobRow)).all()) == 1


def test_layer_norm_network_offers_all_compatible_context_profiles(saved_training):
    env = saved_training
    _use_segformer_pipeline(env, "next_gen4_consistency_v5")
    response = env.client.get(f"/api/v1/results/training/{env.result_id}/continue")
    assert response.status_code == 200, response.text
    assert response.json()["pipeline_variants"] == [f"next_gen4_consistency_v{index}" for index in range(2, 8)]


def test_normalization_advice_uses_only_enabled_valid_pixels_and_never_queues(training_sample, monkeypatch):
    from mlsystem2.training_ui_api import _continuation_normalization as advice

    env = training_sample
    _use_segformer_pipeline(env, "next_gen4_global_norm")
    monkeypatch.setattr(advice, "_report", lambda *_args: {"mode":"global_minmax", "low":[50] * env.channels,
        "high":[200] * env.channels,"valid_pixel_count":1000000})
    response = env.client.post(f"/api/v1/results/training/{env.result_id}/continue/normalization", json={
        "dataset_source":"test_sample","test_sample_id":str(env.sample_id),"pipeline_variant":"next_gen4_global_norm"})
    assert response.status_code == 200, response.text
    info = response.json()
    assert info["recommendation"] == "preserve" and "невелик" in info["reason"]
    assert info["scene_count"] == 2 and info["source_count"] == 2
    assert info["sampled_valid_pixels"] == 2 * 14 * 16
    assert info["clipped_fractions"] == [1.0] * env.channels
    with env.factory() as session:
        assert len(session.scalars(select(JobRow)).all()) == 1


@pytest.mark.parametrize("policy", ["auto", "preserve", "recompute"])
def test_normalization_policy_controls_new_stage_and_is_idempotent(training_sample, policy):
    env = training_sample
    _use_segformer_pipeline(env, "next_gen4_global_norm")
    request = _request(dataset_source="test_sample", test_sample_id=str(env.sample_id),
        normalization_policy=policy, run_inference_after_training=False)
    url = f"/api/v1/results/training/{env.result_id}/continue"
    response = env.client.post(url, json=request)
    assert response.status_code == 200, response.text
    assert env.client.post(url, json={**request,"normalization_policy":"preserve" if policy != "preserve" else "recompute"}).status_code == 400
    with env.factory() as session:
        job = session.get(JobRow, uuid.UUID(response.json()["id"]))
        payload = _worker._build_training_config(session, job, env.config, env.config.scratch_root / str(job.id))
        assert payload["train"]["initial_checkpoint_weights_only"] is True
        assert payload["train"].get("initial_checkpoint_recalculate_normalization", False) is (policy == "recompute")

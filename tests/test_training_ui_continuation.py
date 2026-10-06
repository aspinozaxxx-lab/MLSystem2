from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select
from mlsystem2.settings.api import load_settings
from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, ModelHandle
from mlsystem2.train_pipeline import _runner

from mlsystem2.training_ui_api import _training_continuation as continuation, _worker
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import JobRow, PseudoMarkupResultRow, StoredFileRow, TrainingResultRow
from mlsystem2.training_ui_api._queueing import (
    POST_TRAINING_INFERENCE_CONFIG_KEY, POST_TRAINING_INFERENCE_JOB_IDS_CONFIG_KEY,
    SECONDARY_PRIORITY_CONFIG_KEY, STOP_AND_SAVE_BEST_CONFIG_KEY, is_secondary_job,
)
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api.contracts import TrainingContinuationCreate, TrainingUIAPIError


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


@pytest.mark.parametrize("checkpoint", ["best", "last"])
def test_continue_api_copies_parameters_and_keeps_original(saved_training, checkpoint):
    env = saved_training
    url = f"/api/v1/results/training/{env.result_id}/continue"
    options = env.client.get(url)
    assert options.status_code == 200
    assert options.json() == {"additional_epochs": 30, "additional_time_sec": 1800,
                              "early_stopping_patience": 7, "last_checkpoint_available": True,
                              "run_inference_after_training": True, "secondary_priority": False}
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
    ("additional_time_sec", -1), ("early_stopping_patience", True), ("checkpoint", "произвольный-путь")])
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

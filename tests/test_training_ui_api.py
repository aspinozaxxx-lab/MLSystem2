from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from mlsystem2.mlflow_adapter.contracts import MLflowBestCheckpoint, MLflowExperiment
from mlsystem2.training_ui_api import _automation, _service, _worker
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import JobRow, PseudoMarkupResultRow, TrainingResultRow
from mlsystem2.training_ui_api._service import create_training_job, ensure_seed_templates
from mlsystem2.training_ui_api._worker import dispatch_inference_queue_once, dispatch_training_queue_once
from mlsystem2.training_ui_api.contracts import (
    AutomationEnabledUpdate,
    AutomationRuleUpdate,
    JobSource,
    JobStatus,
    ResultStatus,
    TrainingJobCreate,
    TrainingUIAPIError,
)


def test_training_ui_api_contract_flow(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "deforestation.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "deforestation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    frontend_dist = tmp_path / "frontend" / "dist"
    (frontend_dist / "assets").mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<!doctype html><title>MLSystem2</title>", encoding="utf-8")
    (frontend_dist / "app.js").write_text("console.log('MLSystem2')", encoding="utf-8")
    (frontend_dist / "assets" / "app.css").write_text("body{margin:0}", encoding="utf-8")

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_FRONTEND_DIST", str(frontend_dist))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert client.get("/").text.startswith("<!doctype html>")
        app_js = client.get("/app.js")
        assert app_js.text == "console.log('MLSystem2')"
        assert app_js.headers["content-type"].startswith("text/javascript")
        assert client.get("/assets/app.css").text == "body{margin:0}"
        unauthorized = client.get("/api/v1/datasets")
        assert unauthorized.status_code == 401
        assert client.get("/auth/proxy-check").status_code == 401

        login = client.post(
            "/api/v1/auth/login",
            json={"username": "mluser", "password": "secret"},
        )
        assert login.status_code == 200
        proxy_check = client.get("/auth/proxy-check")
        assert proxy_check.status_code == 204
        assert proxy_check.headers["x-remote-user"] == "mluser"

        datasets = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in datasets] == ["Вырубки\\main", "Custom"]

        new_dir = mlmarkup_root / "Пожары" / "main"
        new_dir.mkdir(parents=True)
        (new_dir / "fires.txt").write_text("scene-2\n", encoding="utf-8")
        (new_dir / "fires.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        refreshed = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in refreshed] == ["Вырубки\\main", "Пожары\\main", "Custom"]

        models = client.get("/api/v1/models").json()["models"]
        assert [item["display_name"] for item in models] == [
            "deeplabV3+",
            "segformer b2",
            "segformer b3",
            "unet + resnet34",
            "unet + resnet50",
            "unet + resnet101",
            "unet + resnet152",
        ]
        automation = client.get("/api/v1/automation").json()
        assert automation["enabled"] is False
        assert [item["name"] for item in automation["datasets"]] == [
            "Вырубки\\main",
            "Пожары\\main",
        ]
        assert len(automation["rules"]) == len(automation["datasets"]) * len(models)
        enabled_automation = client.put("/api/v1/automation/enabled", json={"enabled": True}).json()
        assert enabled_automation["enabled"] is True
        rule = client.put(
            "/api/v1/automation/rules",
            json={
                "dataset_key": "Вырубки\\main",
                "architecture": "smp_segformer_b2",
                "training_enabled": True,
                "pseudo_markup_enabled": True,
            },
        ).json()
        assert rule["training_enabled"] is True
        assert rule["pseudo_markup_enabled"] is True

        templates = client.get("/api/v1/training-templates").json()["templates"]
        assert len(templates) == 7
        segformer_template = client.get("/api/v1/training-templates/smp_segformer_b2").json()
        assert segformer_template["source"] == "hpo_best"
        template_keys = {item["key"] for item in segformer_template["config_schema"]["fields"]}
        assert "dataset.split_granularity" not in template_keys
        assert "tile_preparation.num_workers" not in template_keys
        assert "train.device" not in template_keys
        assert "train.max_train_batches_per_epoch" in template_keys
        assert "train.max_val_batches_per_epoch" in template_keys
        assert "train.max_training_time_sec" in template_keys
        assert segformer_template["default_config"]["train.max_train_batches_per_epoch"] == 72
        assert segformer_template["default_config"]["train.max_val_batches_per_epoch"] == 1000
        assert segformer_template["default_config"]["train.max_training_time_sec"] == 1800
        loss_field = next(
            item for item in segformer_template["config_schema"]["fields"] if item["key"] == "train.loss"
        )
        assert loss_field["options"] == ["bce_dice", "focal_dice", "focal_tversky"]
        changed_config = dict(segformer_template["default_config"])
        changed_config["train.batch_size"] = 3
        updated = client.put(
            "/api/v1/training-templates/smp_segformer_b2",
            json={"default_config": changed_config},
        ).json()
        assert updated["source"] == "manual"
        assert updated["version"] == segformer_template["version"] + 1
        reset = client.put(
            "/api/v1/training-templates/smp_segformer_b2",
            json={"reset_to_baseline": True},
        ).json()
        assert reset["source"] == "hpo_best"

        custom = client.post(
            "/api/v1/custom-datasets",
            data={"name": "Custom"},
            files={
                "scenes_txt": ("custom.txt", b"scene-custom\n", "text/plain"),
                "annotation_geojson": (
                    "custom.geojson",
                    b'{"type":"FeatureCollection","features":[]}',
                    "application/geo+json",
                ),
            },
        ).json()
        assert custom["name"] == "Custom"
        scenes_download = client.get(custom["scenes_file"]["download_url"])
        assert scenes_download.status_code == 200
        assert scenes_download.text == "scene-custom\n"

        job = client.post(
            "/api/v1/training-jobs",
            json={
                "mlflow_experiment_id": "45",
                "mlflow_experiment_name": "Segformer-b2-HPO-deforest-2605",
                "mlflow_run_name": "ui_test_run",
                "dataset_key": "custom",
                "custom_dataset_id": custom["id"],
                "architecture": "smp_segformer_b2",
                "config": reset["default_config"],
            },
        ).json()
        assert job["status"] == "queued"
        assert job["dataset_name"] == "Custom"
        assert "train.device" not in job["config"]
        assert "dataset.split_granularity" not in job["config"]

        queues = client.get("/api/v1/queues").json()
        assert queues["training_enabled"] is True
        assert len(queues["training_jobs"]) == 1

        detail = client.get(f"/api/v1/jobs/{job['id']}").json()
        assert detail["readonly"] is True

        custom_results = client.get("/api/v1/results/classes/custom").json()
        training_result_id = custom_results["results"][0]["id"]
        pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
        ).json()
        assert pseudo["type"] == "inference"
        inference_queue = client.get("/api/v1/queues").json()["inference_jobs"]
        assert len(inference_queue) == 1
        pseudo_with_empty_upload = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки\\main", "training_result_id": training_result_id},
            files={"scenes_txt": ("", b"", "application/octet-stream")},
        ).json()
        assert pseudo_with_empty_upload["type"] == "inference"
        class_results = client.get("/api/v1/results/classes/custom").json()
        pseudo_scenes = class_results["results"][0]["pseudo_markup_results"][0]["scenes_file"]
        assert client.get(pseudo_scenes["download_url"]).text.splitlines() == ["scene-1"]

        deleted = client.delete(f"/api/v1/jobs/{job['id']}").json()
        assert deleted["status"] == "cancelled"
        assert client.get("/api/v1/queues").json()["training_jobs"] == []


def test_training_ui_worker_starts_first_training_job(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    started: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        started.append((args, kwargs))
        return SimpleNamespace(pid=4321)

    with session_factory() as session:
        ensure_seed_templates(session)
        template = session.query(JobRow).first()
        assert template is None
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="worker-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.positive_factor": 0.5,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
                    "train.max_train_batches_per_epoch": 72,
                    "train.max_val_batches_per_epoch": 1000,
                    "train.max_training_time_sec": None,
                },
            ),
            config,
        )
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        row = session.get(JobRow, job.id)
        assert row is not None
        assert row.status == JobStatus.RUNNING.value
        assert row.process_pid == 4321
        assert row.tmp_path is not None
        run_dir = Path(row.tmp_path)
        assert (run_dir / "run.yml").is_file()
        assert (run_dir / "run_training.sh").is_file()
        config_yaml = (run_dir / "run.yml").read_text(encoding="utf-8")
        run_script = (run_dir / "run_training.sh").read_text(encoding="utf-8")
        assert "split_granularity" not in config_yaml
        assert "num_workers" not in config_yaml
        assert "input_channels" not in config_yaml
        assert "images_dir" not in config_yaml
        assert "inference:" not in config_yaml
        assert "max_train_batches_per_epoch: 72" in config_yaml
        assert "max_val_batches_per_epoch: 1000" in config_yaml
        assert "max_training_time_sec: null" in config_yaml
        assert "--settings" in run_script
        assert "--run" in run_script
        assert "run.yml" in run_script
        assert "MLSYSTEM2_MLFLOW_RUN_ID_FILE" in run_script

    assert started
    assert started[0][0][0] == "bash"
    assert started[0][1]["cwd"] == str(tmp_path)
    assert started[0][1]["start_new_session"] is True


def test_training_ui_automation_has_lower_priority_than_manual_jobs(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    started: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        started.append((args, kwargs))
        return SimpleNamespace(pid=4321)

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=False,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        auto_job = session.scalar(select(JobRow).where(JobRow.source == JobSource.AUTOMATION.value))
        assert auto_job is not None
        assert auto_job.status == JobStatus.QUEUED.value
        assert auto_job.dataset_key == "Вырубки\\main"
        assert auto_job.dataset_version is not None
        old_auto_job_id = auto_job.id
        old_version = auto_job.dataset_version
        annotation_path = class_dir / "annotation.geojson"
        annotation_path.write_text('{"type":"FeatureCollection","features":[{"type":"Feature"}]}', encoding="utf-8")
        stat = annotation_path.stat()
        os.utime(annotation_path, ns=(stat.st_atime_ns + 2_000_000_000, stat.st_mtime_ns + 2_000_000_000))
        _automation.sync_automation_once(session, config)
        assert session.get(JobRow, old_auto_job_id).status == JobStatus.CANCELLED.value
        auto_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.status == JobStatus.QUEUED.value,
            )
        )
        assert auto_job is not None
        assert auto_job.dataset_version != old_version
        stale_auto_job_id = auto_job.id
        _service.set_automation(session, AutomationEnabledUpdate(enabled=False), config)
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        assert started == []
        assert session.get(JobRow, stale_auto_job_id).status == JobStatus.CANCELLED.value
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        auto_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.status == JobStatus.QUEUED.value,
            )
        )
        assert auto_job is not None
        assert auto_job.id != stale_auto_job_id

        manual_job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="manual-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config=_short_training_config(),
            ),
            config,
        )
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        running = session.scalar(select(JobRow).where(JobRow.status == JobStatus.RUNNING.value))
        assert running is not None
        assert running.id == manual_job.id
        assert running.source == JobSource.MANUAL.value
        assert session.get(JobRow, auto_job.id).status == JobStatus.QUEUED.value
        with pytest.raises(TrainingUIAPIError):
            _service.delete_job(session, auto_job.id)
        with pytest.raises(TrainingUIAPIError):
            _service.move_job(session, auto_job.id, direction=-1)


def test_training_ui_disabling_automation_cancels_running_jobs_and_kills_mlflow(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])
    killed_runs: list[tuple[str, str]] = []
    terminated_pids: list[int | None] = []

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    def fake_terminate(row: JobRow) -> None:
        terminated_pids.append(row.process_pid)
        row.process_pid = None

    def fake_mark_run_killed(tracking_uri: str, run_id: str) -> None:
        killed_runs.append((tracking_uri, run_id))

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)
    monkeypatch.setattr(_automation, "terminate_job_process", fake_terminate)
    monkeypatch.setattr(_automation, "mark_run_killed", fake_mark_run_killed)

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=False,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        auto_job = session.scalar(select(JobRow).where(JobRow.source == JobSource.AUTOMATION.value))
        assert auto_job is not None
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "mlflow_run_id").write_text("run-auto-kill\n", encoding="utf-8")
        auto_job.status = JobStatus.RUNNING.value
        auto_job.process_pid = 9876
        auto_job.tmp_path = str(run_dir)
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == auto_job.id))
        assert result is not None
        result.mlflow_run_id = "run-auto-kill"

        _service.set_automation(session, AutomationEnabledUpdate(enabled=False), config)

        assert auto_job.status == JobStatus.CANCELLED.value
        assert result.status == ResultStatus.CANCELLED.value
        assert auto_job.tmp_path is None
        assert not run_dir.exists()
        assert terminated_pids == [9876]
        assert killed_runs == [(config.mlflow_tracking_uri, "run-auto-kill")]


def test_training_ui_automation_creates_pseudo_after_training_and_does_not_retry_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    def fake_experiment(request) -> MLflowExperiment:
        return MLflowExperiment(experiment_id="auto-exp", name=request.name)

    def fake_best_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint:
        return MLflowBestCheckpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            metric_name="val/best_threshold_pixel_f1",
            f1_score=0.91,
            epoch=4,
            artifact_path="checkpoints/best.pt",
            artifact_uri="s3://mlflow-artifacts/auto/run/artifacts/checkpoints/best.pt",
            threshold=0.63,
        )

    monkeypatch.setattr(_automation, "create_experiment", fake_experiment)
    monkeypatch.setattr(_automation, "get_best_training_checkpoint", fake_best_checkpoint)

    with session_factory() as session:
        ensure_seed_templates(session)
        _service.set_automation(session, AutomationEnabledUpdate(enabled=True), config)
        _service.update_automation(
            session,
            AutomationRuleUpdate(
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                training_enabled=True,
                pseudo_markup_enabled=True,
            ),
            config,
        )
        _automation.sync_automation_once(session, config)
        training_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "training",
            )
        )
        assert training_job is not None
        training_result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == training_job.id))
        assert training_result is not None
        training_job.status = JobStatus.COMPLETED.value
        training_result.status = ResultStatus.OK.value
        training_result.mlflow_run_id = "run-auto"
        training_result.f1_score = 0.91
        training_result.epoch = 4

        _automation.sync_automation_once(session, config)
        pseudo_job = session.scalar(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "inference",
            )
        )
        assert pseudo_job is not None
        assert pseudo_job.dataset_key == "Вырубки\\main"
        assert pseudo_job.dataset_version == training_job.dataset_version
        assert pseudo_job.config["training_result_id"] == str(training_result.id)
        assert pseudo_job.config["checkpoint_uri"] == "s3://mlflow-artifacts/auto/run/artifacts/checkpoints/best.pt"
        pseudo_result = session.scalar(select(PseudoMarkupResultRow).where(PseudoMarkupResultRow.job_id == pseudo_job.id))
        assert pseudo_result is not None
        assert pseudo_result.scenes_file is not None
        assert Path(pseudo_result.scenes_file.path).read_text(encoding="utf-8") == "scene-1\n"

        training_result.status = ResultStatus.ERROR.value
        pseudo_job.status = JobStatus.CANCELLED.value
        pseudo_result.status = ResultStatus.CANCELLED.value
        _automation.sync_automation_once(session, config)
        training_jobs = session.scalars(
            select(JobRow).where(
                JobRow.source == JobSource.AUTOMATION.value,
                JobRow.type == "training",
            )
        ).all()
        assert len(training_jobs) == 1


def test_training_ui_worker_records_best_mlflow_metric(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки" / "main"
    class_dir.mkdir(parents=True)
    (class_dir / "scenes.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "annotation.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_IMAGES_ROOT", str(tmp_path / "images"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SCRATCH_ROOT", str(tmp_path / "scratch"))
    monkeypatch.setenv("MLSYSTEM2_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_WORKER_ENABLED", "false")

    config = get_config()
    configure_schema(None)
    session_factory = create_session_factory(config)
    Base.metadata.create_all(session_factory.kw["bind"])

    def fake_popen(args: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(pid=4321)

    def fake_best_checkpoint(tracking_uri: str, run_id: str) -> MLflowBestCheckpoint:
        assert tracking_uri == config.mlflow_tracking_uri
        assert run_id == "run-123"
        return MLflowBestCheckpoint(
            tracking_uri=tracking_uri,
            run_id=run_id,
            metric_name="val/best_threshold_pixel_f1",
            f1_score=0.8123,
            epoch=7,
            artifact_path="checkpoints/best.pt",
            artifact_uri="s3://mlflow-artifacts/1/run-123/artifacts/checkpoints/best.pt",
            threshold=0.7,
        )

    monkeypatch.setattr(_worker, "get_best_training_checkpoint", fake_best_checkpoint)

    with session_factory() as session:
        ensure_seed_templates(session)
        job = create_training_job(
            session,
            TrainingJobCreate(
                mlflow_experiment_id="1",
                mlflow_experiment_name="ui-test",
                mlflow_run_name="worker-test",
                dataset_key="Вырубки\\main",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.positive_factor": 0.5,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
                    "train.max_training_time_sec": None,
                },
            ),
            config,
        )
        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.flush()

        row = session.get(JobRow, job.id)
        assert row is not None
        assert row.tmp_path is not None
        run_dir = Path(row.tmp_path)
        (run_dir / "mlflow_run_id").write_text("run-123\n", encoding="utf-8")
        _worker._sync_training_run_id(session, row, config)
        running_result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert running_result is not None
        assert running_result.mlflow_run_id == "run-123"
        assert running_result.mlflow_run_url is not None

        (run_dir / "train.log").write_text("status=succeeded\nmlflow_run=run-123\n", encoding="utf-8")
        (run_dir / "exit_code").write_text("0\n", encoding="utf-8")

        dispatch_training_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        finished = session.get(JobRow, job.id)
        assert finished is not None
        assert finished.status == JobStatus.COMPLETED.value
        result = session.scalar(select(TrainingResultRow).where(TrainingResultRow.job_id == job.id))
        assert result is not None
        assert result.status == ResultStatus.OK.value
        assert result.mlflow_run_id == "run-123"
        assert result.f1_score == 0.8123
        assert result.epoch == 7

        monkeypatch.setattr(_service, "get_best_training_checkpoint", fake_best_checkpoint)
        pseudo_job = _service.create_pseudo_markup_job(
            session,
            class_key="Вырубки\\main",
            dataset_key="Вырубки\\main",
            training_result_id=result.id,
            scenes_name=None,
            scenes_content_type=None,
            scenes_bytes=None,
            config=config,
        )
        assert pseudo_job.config["mlflow_run_id"] == "run-123"
        assert pseudo_job.config["checkpoint_artifact_path"] == "checkpoints/best.pt"
        assert (
            pseudo_job.config["checkpoint_uri"]
            == "s3://mlflow-artifacts/1/run-123/artifacts/checkpoints/best.pt"
        )
        assert pseudo_job.config["checkpoint_f1_score"] == 0.8123
        assert pseudo_job.config["checkpoint_epoch"] == 7
        assert pseudo_job.config["checkpoint_threshold"] == 0.7

        dispatch_inference_queue_once(session, config, popen_factory=fake_popen)
        session.flush()

        pseudo_row = session.get(JobRow, pseudo_job.id)
        assert pseudo_row is not None
        assert pseudo_row.status == JobStatus.RUNNING.value
        assert pseudo_row.tmp_path is not None
        pseudo_run_dir = Path(pseudo_row.tmp_path)
        pseudo_config = (pseudo_run_dir / "pseudo_config.yaml").read_text(encoding="utf-8")
        assert "threshold: 0.7" in pseudo_config
        assert "checkpoint_artifact_path: checkpoints/best.pt" in pseudo_config

        output = pseudo_run_dir / "scratch" / "pseudo_markup.geojson"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        (pseudo_run_dir / "scratch" / "report.json").write_text(
            '{"status":"ok","feature_count":0}',
            encoding="utf-8",
        )
        (pseudo_run_dir / "exit_code").write_text("0\n", encoding="utf-8")

        dispatch_inference_queue_once(session, config, popen_factory=fake_popen)
        session.commit()

        completed_pseudo = session.get(JobRow, pseudo_job.id)
        assert completed_pseudo is not None
        assert completed_pseudo.status == JobStatus.COMPLETED.value
        refreshed_results = _service.class_results(session, "Вырубки\\main", config)
        pseudo_results = refreshed_results.results[0].pseudo_markup_results
        assert pseudo_results[0].status == ResultStatus.OK
        assert pseudo_results[0].geojson_file is not None
        assert re.fullmatch(
            r"Вырубки\\main_segformer b2_\d{2}_\d{2}_\d{2}_\d{2}\.geojson",
            pseudo_results[0].geojson_file.original_name,
        )
        geojson_path = config.stored_files_root / "pseudo_markup_geojson"
        assert any(path.suffix == ".geojson" for path in geojson_path.iterdir())


def _short_training_config() -> dict[str, object]:
    return {
        "dataset.val_fraction": 0.2,
        "tile_preparation.tile_size": 32,
        "tile_preparation.stride": 32,
        "tile_preparation.augmentation_level": 0,
        "tile_preparation.positive_factor": 0.5,
        "train.epochs": 1,
        "train.batch_size": 1,
        "train.learning_rate": 0.0001,
        "train.weight_decay": 0.0,
        "train.loss": "bce_dice",
        "train.focal_alpha": 0.6,
        "train.pos_weight": 1.0,
        "train.tversky_alpha": 0.4,
        "train.tversky_beta": 0.6,
        "train.threshold": 0.5,
        "train.early_stopping_patience": 1,
        "train.max_train_batches_per_epoch": 72,
        "train.max_val_batches_per_epoch": 1000,
        "train.max_training_time_sec": None,
    }

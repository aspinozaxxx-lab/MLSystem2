from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import select

from mlsystem2.mlflow_adapter.contracts import MLflowBestCheckpoint
from mlsystem2.training_ui_api import _service, _worker
from mlsystem2.training_ui_api.api import create_app
from mlsystem2.training_ui_api._config import get_config
from mlsystem2.training_ui_api._database import Base, configure_schema, create_session_factory
from mlsystem2.training_ui_api._models import JobRow, TrainingResultRow
from mlsystem2.training_ui_api._service import create_training_job, ensure_seed_templates
from mlsystem2.training_ui_api._worker import dispatch_inference_queue_once, dispatch_training_queue_once
from mlsystem2.training_ui_api.contracts import JobStatus, ResultStatus, TrainingJobCreate


def test_training_ui_api_contract_flow(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки"
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
        assert [item["name"] for item in datasets] == ["Вырубки", "Custom"]

        new_dir = mlmarkup_root / "Пожары"
        new_dir.mkdir()
        (new_dir / "fires.txt").write_text("scene-2\n", encoding="utf-8")
        (new_dir / "fires.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
        refreshed = client.get("/api/v1/datasets").json()["datasets"]
        assert [item["name"] for item in refreshed] == ["Вырубки", "Пожары", "Custom"]

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

        templates = client.get("/api/v1/training-templates").json()["templates"]
        assert len(templates) == 7
        segformer_template = client.get("/api/v1/training-templates/smp_segformer_b2").json()
        assert segformer_template["source"] == "hpo_best"
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

        queues = client.get("/api/v1/queues").json()
        assert queues["training_enabled"] is True
        assert len(queues["training_jobs"]) == 1

        detail = client.get(f"/api/v1/jobs/{job['id']}").json()
        assert detail["readonly"] is True

        custom_results = client.get("/api/v1/results/classes/custom").json()
        training_result_id = custom_results["results"][0]["id"]
        pseudo = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки", "training_result_id": training_result_id},
        ).json()
        assert pseudo["type"] == "inference"
        inference_queue = client.get("/api/v1/queues").json()["inference_jobs"]
        assert len(inference_queue) == 1
        pseudo_with_empty_upload = client.post(
            "/api/v1/results/classes/custom/pseudo-markup",
            data={"dataset_key": "Вырубки", "training_result_id": training_result_id},
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
    class_dir = mlmarkup_root / "Вырубки"
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
                dataset_key="Вырубки",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "dataset.split_granularity": "scene",
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.num_workers": 0,
                    "tile_preparation.prefetch_factor": 2,
                    "tile_preparation.seed": 42,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.smart_tiling": False,
                    "tile_preparation.positive_factor": 0.5,
                    "tile_preparation.class_balance": False,
                    "train.task": "binary",
                    "train.input_channels": 4,
                    "train.output_channels": 1,
                    "train.pretrained": False,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.device": "cpu",
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
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
        assert (run_dir / "config.yaml").is_file()
        assert (run_dir / "run_training.sh").is_file()

    assert started
    assert started[0][0][0] == "bash"
    assert started[0][1]["cwd"] == str(tmp_path)
    assert started[0][1]["start_new_session"] is True


def test_training_ui_worker_records_best_mlflow_metric(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки"
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
                dataset_key="Вырубки",
                architecture="smp_segformer_b2",
                config={
                    "dataset.val_fraction": 0.2,
                    "dataset.split_granularity": "scene",
                    "tile_preparation.tile_size": 32,
                    "tile_preparation.stride": 32,
                    "tile_preparation.num_workers": 0,
                    "tile_preparation.prefetch_factor": 2,
                    "tile_preparation.seed": 42,
                    "tile_preparation.augmentation_level": 0,
                    "tile_preparation.smart_tiling": False,
                    "tile_preparation.positive_factor": 0.5,
                    "tile_preparation.class_balance": False,
                    "train.task": "binary",
                    "train.input_channels": 4,
                    "train.output_channels": 1,
                    "train.pretrained": False,
                    "train.epochs": 1,
                    "train.batch_size": 1,
                    "train.device": "cpu",
                    "train.learning_rate": 0.0001,
                    "train.weight_decay": 0.0,
                    "train.loss": "bce_dice",
                    "train.focal_alpha": 0.6,
                    "train.pos_weight": 1.0,
                    "train.tversky_alpha": 0.4,
                    "train.tversky_beta": 0.6,
                    "train.threshold": 0.5,
                    "train.early_stopping_patience": 1,
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
            class_key="Вырубки",
            dataset_key="Вырубки",
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
        refreshed_results = _service.class_results(session, "Вырубки", config)
        pseudo_results = refreshed_results.results[0].pseudo_markup_results
        assert pseudo_results[0].status == ResultStatus.OK
        assert pseudo_results[0].geojson_file is not None
        geojson_path = config.stored_files_root / "pseudo_markup_geojson"
        assert any(path.suffix == ".geojson" for path in geojson_path.iterdir())

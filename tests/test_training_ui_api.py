from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mlsystem2.training_ui_api.api import create_app


def test_training_ui_api_contract_flow(tmp_path: Path, monkeypatch) -> None:
    mlmarkup_root = tmp_path / "MLMarkup"
    class_dir = mlmarkup_root / "Вырубки"
    class_dir.mkdir(parents=True)
    (class_dir / "deforestation.txt").write_text("scene-1\n", encoding="utf-8")
    (class_dir / "deforestation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    frontend_dist = tmp_path / "frontend" / "dist"
    (frontend_dist / "assets").mkdir(parents=True)
    (frontend_dist / "index.html").write_text("<!doctype html><title>MLSystem2</title>", encoding="utf-8")
    (frontend_dist / "assets" / "app.css").write_text("body{margin:0}", encoding="utf-8")

    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_URL", f"sqlite:///{tmp_path / 'ui.db'}")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_STORED_FILES_ROOT", str(tmp_path / "files"))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_FRONTEND_DIST", str(frontend_dist))
    monkeypatch.setenv("MLSYSTEM2_MLMARKUP_ROOT", str(mlmarkup_root))
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_USER", "mluser")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_PASSWORD", "secret")
    monkeypatch.setenv("MLSYSTEM2_TRAINING_UI_SESSION_SECRET", "test-session-secret")

    with TestClient(create_app()) as client:
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert client.get("/").text.startswith("<!doctype html>")
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
        class_results = client.get("/api/v1/results/classes/custom").json()
        pseudo_scenes = class_results["results"][0]["pseudo_markup_results"][0]["scenes_file"]
        assert client.get(pseudo_scenes["download_url"]).text.splitlines() == ["scene-1"]

        deleted = client.delete(f"/api/v1/jobs/{job['id']}").json()
        assert deleted["status"] == "cancelled"
        assert client.get("/api/v1/queues").json()["training_jobs"] == []

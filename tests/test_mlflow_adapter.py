from __future__ import annotations

from pathlib import Path

from mlsystem2.mlflow_adapter.api import (
    download_run_artifact,
    get_best_training_checkpoint,
    get_training_epoch_progress,
    get_usable_training_checkpoint,
    log_dataset_artifacts,
    log_run_config,
    log_tile_preparation,
    mark_run_killed,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef, MLflowStartRunRequest
from mlsystem2.mlflow_adapter import _client
from mlsystem2.train.contracts import EpochMetrics, TrainResult


def test_next_run_name_uses_class_date_and_daily_counter() -> None:
    name = _client._next_run_name(
        [
            "deforestation_2305_1",
            "deforestation_2305_2",
            "deforestation_2205_7",
            "other_2305_9",
        ],
        "deforestation",
        "2305",
    )

    assert name == "deforestation_2305_3"


def test_next_run_name_starts_from_one() -> None:
    assert _client._next_run_name([], "deforestation", "2305") == "deforestation_2305_1"


def test_config_and_tile_artifacts_are_noop_when_run_disabled(tmp_path: Path) -> None:
    run = MLflowRunRef(run_id="disabled", experiment_name="test", tracking_uri="file://mlruns", active=False)

    log_run_config(run, tmp_path / "missing.yaml")
    log_tile_preparation(run, {"splits": {}})


def test_start_run_writes_dataset_tag(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}
    run_id_path = tmp_path / "run_id.txt"
    monkeypatch.setenv("MLSYSTEM2_MLFLOW_RUN_ID_FILE", str(run_id_path))

    class RunInfo:
        run_id = "run-1"

    class Run:
        info = RunInfo()

    class Experiment:
        experiment_id = "exp-1"

    class Dataset:
        def __init__(self, name: str):
            self.name = name

    class MLflow:
        class data:
            @staticmethod
            def from_numpy(features, source: str, name: str, digest: str):
                calls["dataset_source"] = source
                calls["dataset_name"] = name
                calls["dataset_digest"] = digest
                calls["dataset_shape"] = tuple(features.shape)
                return {"name": name, "digest": digest}

        class tracking:
            class MlflowClient:
                def search_runs(self, experiment_ids, max_results=1000):
                    return []

                def search_datasets(self, experiment_ids, max_results=1000):
                    calls["search_datasets"] = (experiment_ids, max_results)
                    return []

                def create_dataset(self, name: str, experiment_id: str, tags: dict[str, str]):
                    calls["created_dataset"] = (name, experiment_id, tags)
                    return Dataset(name)

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uri"] = uri

        @staticmethod
        def set_experiment(name: str) -> None:
            calls["experiment_name"] = name

        @staticmethod
        def get_experiment_by_name(name: str):
            calls["search_experiment_name"] = name
            return Experiment()

        @staticmethod
        def start_run(run_name: str | None = None, tags: dict[str, str] | None = None):
            calls["run_name"] = run_name
            calls["tags"] = tags
            return Run()

        @staticmethod
        def log_input(dataset, context: str | None = None, tags: dict[str, str] | None = None):
            calls["input_dataset"] = dataset
            calls["input_context"] = context
            calls["input_tags"] = tags

        @staticmethod
        def active_run():
            return None

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    run = _client.start_run(
        MLflowStartRunRequest(
            enabled=True,
            tracking_uri="file://mlruns",
            experiment_name="exp",
            dataset="deforestation",
            run_name="manual",
            tags={"pipeline": "train"},
        )
    )

    assert run.run_id == "run-1"
    assert run_id_path.read_text(encoding="utf-8") == "run-1\n"
    assert calls["tags"] == {"pipeline": "train", "dataset": "deforestation"}
    assert calls["dataset_name"] == "deforestation"
    assert calls["dataset_source"] == "deforestation"
    assert calls["dataset_shape"] == (0, 0)
    assert calls["created_dataset"] == (
        "deforestation",
        "exp-1",
        {"dataset": "deforestation", "source": "MLMarkup geojson stem"},
    )
    assert calls["input_context"] == "train"
    assert calls["input_tags"] == {"dataset": "deforestation"}


def test_get_best_training_checkpoint_uses_metric_history(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Metric:
        def __init__(self, value: float, step: int) -> None:
            self.value = value
            self.step = step

    class RunInfo:
        artifact_uri = "s3://mlflow-artifacts/45/run-1/artifacts"

    class Run:
        info = RunInfo()

    class Client:
        def get_run(self, run_id: str):
            calls["run_id"] = run_id
            return Run()

        def get_metric_history(self, run_id: str, metric_name: str):
            calls["metric"] = (run_id, metric_name)
            if metric_name == "val/best_threshold":
                return [Metric(0.5, 1), Metric(0.7, 4), Metric(0.8, 5)]
            return [Metric(0.3, 1), Metric(0.8, 4), Metric(0.8, 5), Metric(0.7, 6)]

    class MLflow:
        class tracking:
            @staticmethod
            def MlflowClient():
                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uri"] = uri

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    checkpoint = get_best_training_checkpoint("http://mlflow:5000", "run-1")

    assert checkpoint is not None
    assert checkpoint.metric_name == "val/best_threshold_pixel_f1"
    assert checkpoint.f1_score == 0.8
    assert checkpoint.epoch == 4
    assert checkpoint.threshold == 0.7
    assert checkpoint.artifact_path == "checkpoints/best.pt"
    assert checkpoint.artifact_uri == "s3://mlflow-artifacts/45/run-1/artifacts/checkpoints/best.pt"
    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["run_id"] == "run-1"


# Proveriaet status run i fakticheskoe nalichie best.pt.
def test_get_usable_training_checkpoint_requires_finished_run_and_artifact(monkeypatch) -> None:
    checkpoint = _client.MLflowBestCheckpoint(
        tracking_uri="http://mlflow:5000",
        run_id="run-1",
        metric_name="val/best_threshold_pixel_f1",
        f1_score=0.8,
        epoch=4,
        artifact_path="checkpoints/best.pt",
        artifact_uri="s3://artifacts/checkpoints/best.pt",
        threshold=0.7,
    )

    class Info:
        """Testovyi status MLflow run."""

        status = "FINISHED"

    class Run:
        """Testovyi MLflow run."""

        info = Info()

    class Artifact:
        """Testovyi artifact best checkpoint."""

        path = "checkpoints/best.pt"

    class Client:
        """Minimalnyi fake MLflow client."""

        # Vozvrashchaet zavershennyi run.
        def get_run(self, run_id: str):
            assert run_id == "run-1"
            return Run()

        # Vozvrashchaet spisok artefaktov checkpoints.
        def list_artifacts(self, run_id: str, path: str):
            assert (run_id, path) == ("run-1", "checkpoints")
            return [Artifact()]

    class MLflow:
        """Minimalnyi fake modul MLflow."""

        class tracking:
            """Prostranstvo imen tracking klienta."""

            @staticmethod
            def MlflowClient():
                """Sozdat fake client."""

                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            """Proverit peredannyi tracking URI."""

            assert uri == "http://mlflow:5000"

    monkeypatch.setattr(_client, "get_best_training_checkpoint", lambda uri, run_id: checkpoint)
    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    assert get_usable_training_checkpoint("http://mlflow:5000", "run-1") == checkpoint


def test_get_training_epoch_progress_uses_epoch_time_history(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Metric:
        def __init__(self, step: int) -> None:
            self.step = step

    class Client:
        def get_metric_history(self, run_id: str, metric_name: str):
            calls["metric"] = (run_id, metric_name)
            return [Metric(1), Metric(3), Metric(2)]

    class MLflow:
        class tracking:
            @staticmethod
            def MlflowClient():
                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uri"] = uri

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    progress = get_training_epoch_progress("http://mlflow:5000", "run-1")

    assert progress.completed_epochs == 3
    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["metric"] == ("run-1", "train/epoch_time_sec")


def test_download_run_artifact_uses_mlflow_client(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class Client:
        def download_artifacts(self, run_id: str, artifact_path: str, dst_path: str) -> str:
            calls["download"] = (run_id, artifact_path, dst_path)
            return str(Path(dst_path) / "checkpoints" / "best.pt")

    class MLflow:
        class tracking:
            @staticmethod
            def MlflowClient():
                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uri"] = uri

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    artifact = download_run_artifact(
        tracking_uri="http://mlflow:5000",
        run_id="run-1",
        artifact_path="checkpoints/best.pt",
        dst_dir=tmp_path,
    )

    assert artifact.local_path == str(tmp_path / "checkpoints" / "best.pt")
    assert artifact.run_id == "run-1"
    assert artifact.artifact_path == "checkpoints/best.pt"
    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["download"] == ("run-1", "checkpoints/best.pt", str(tmp_path))


def test_mark_run_killed_uses_mlflow_client(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Client:
        def set_terminated(self, run_id: str, status: str) -> None:
            calls["terminated"] = (run_id, status)

    class MLflow:
        class tracking:
            @staticmethod
            def MlflowClient():
                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls["tracking_uri"] = uri

        @staticmethod
        def active_run():
            return None

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    mark_run_killed("http://mlflow:5000", "run-1")

    assert calls["tracking_uri"] == "http://mlflow:5000"
    assert calls["terminated"] == ("run-1", "KILLED")


def test_log_run_config_uses_fixed_artifact_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = tmp_path / "source-name.yaml"
    config.write_text("train:\n  epochs: 1\n", encoding="utf-8")
    logged: list[tuple[str, str]] = []

    class MLflow:
        @staticmethod
        def log_text(content: str, artifact_file: str) -> None:
            logged.append((content, artifact_file))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True)

    log_run_config(run, config)

    assert logged == [("train:\n  epochs: 1\n", "config/train_config.yaml")]


def test_log_tile_preparation_uses_report_artifact_path(monkeypatch) -> None:
    logged: list[tuple[dict[str, object], str]] = []

    class MLflow:
        @staticmethod
        def log_dict(payload: dict[str, object], artifact_file: str) -> None:
            logged.append((payload, artifact_file))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True)

    log_tile_preparation(run, {"tile_size": 1024})

    assert logged == [({"tile_size": 1024}, "reports/tile_preparation.json")]


def test_log_dataset_artifacts_writes_files_under_dataset(tmp_path: Path, monkeypatch) -> None:
    scenes = tmp_path / "source-scenes.txt"
    scenes.write_text("scene-1\n", encoding="utf-8")
    annotation = tmp_path / "source-annotation.geojson"
    annotation.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    logged: list[tuple[str, str, str]] = []

    class MLflow:
        @staticmethod
        def log_artifact(path: str | Path, artifact_path: str) -> None:
            logged.append((Path(path).name, Path(path).read_text(encoding="utf-8"), artifact_path))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True)

    log_dataset_artifacts(
        run,
        {
            "scenes.txt": scenes,
            "annotation.geojson": annotation,
        },
    )

    assert logged == [
        ("scenes.txt", "scene-1\n", "dataset"),
        ("annotation.geojson", '{"type":"FeatureCollection","features":[]}', "dataset"),
    ]


def test_log_training_epoch_reactivates_run_by_id(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class RunInfo:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class ActiveRun:
        def __init__(self, run_id: str) -> None:
            self.info = RunInfo(run_id)

    class MLflow:
        current_run: ActiveRun | None = None

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            calls.append(("tracking_uri", uri))

        @staticmethod
        def active_run():
            return MLflow.current_run

        @staticmethod
        def start_run(run_id: str):
            calls.append(("start_run", run_id))
            MLflow.current_run = ActiveRun(run_id)
            return MLflow.current_run

        @staticmethod
        def log_metric(name: str, value: float, step: int = 0) -> None:
            active_id = MLflow.current_run.info.run_id if MLflow.current_run is not None else None
            calls.append(("metric", name, value, step, active_id))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run-42", experiment_name="test", tracking_uri="file://mlruns", active=True)

    _client.log_training_epoch(
        run,
        EpochMetrics(
            epoch=1,
            train_loss=1.0,
            val_loss=1.0,
            epoch_time_sec=1.0,
        ),
    )

    assert ("tracking_uri", "file://mlruns") in calls
    assert ("start_run", "run-42") in calls
    assert ("metric", "train/loss", 1.0, 1, "run-42") in calls


def test_log_training_epoch_writes_only_epoch_hpo_metrics(monkeypatch) -> None:
    logged: list[tuple[str, float, int]] = []

    class MLflow:
        @staticmethod
        def log_metric(name: str, value: float, step: int = 0) -> None:
            logged.append((name, value, step))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True)

    _client.log_training_epoch(
        run,
        EpochMetrics(
            epoch=3,
            train_loss=1.0,
            val_loss=1.0,
            val_best_threshold=0.75,
            val_best_threshold_pixel_f1=0.6,
            val_best_threshold_precision=0.7,
            val_best_threshold_recall=0.52,
            epoch_time_sec=1.0,
        ),
    )

    assert logged == [
        ("train/loss", 1.0, 3),
        ("val/loss", 1.0, 3),
        ("val/best_threshold", 0.75, 3),
        ("val/best_pixel_threshold", 0.75, 3),
        ("val/quality_f1", 0.6, 3),
        ("val/quality_precision", 0.7, 3),
        ("val/quality_recall", 0.52, 3),
        ("val/best_threshold_pixel_f1", 0.6, 3),
        ("val/pixel_f1", 0.6, 3),
        ("val/pixel_precision", 0.7, 3),
        ("val/pixel_recall", 0.52, 3),
        ("val/best_threshold_precision", 0.7, 3),
        ("val/best_threshold_recall", 0.52, 3),
        ("train/epoch_time_sec", 1.0, 3),
    ]


def test_log_training_metrics_writes_train_best_hpo_metric(monkeypatch) -> None:
    logged: list[tuple[str, float, int | None]] = []

    class MLflow:
        @staticmethod
        def log_metric(name: str, value: float, step: int | None = None) -> None:
            logged.append((name, value, step))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True)

    _client.log_training_metrics(
        run,
        TrainResult(
            history=[
                EpochMetrics(
                    epoch=1,
                    train_loss=1.0,
                    val_loss=1.0,
                    val_best_threshold_pixel_f1=0.4,
                    epoch_time_sec=1.0,
                ),
                EpochMetrics(
                    epoch=2,
                    train_loss=0.8,
                    val_loss=0.9,
                    val_best_threshold_pixel_f1=0.6,
                    epoch_time_sec=1.2,
                ),
            ],
            epochs_total=2,
            training_time_sec=2.2,
        ),
    )

    assert logged == [
        ("train/epochs_total", 2, None),
        ("train/training_time_sec", 2.2, None),
        ("train/best_quality_f1", 0.6, None),
        ("train/best_threshold_pixel_f1", 0.6, None),
    ]

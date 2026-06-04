from __future__ import annotations

from pathlib import Path

from mlsystem2.mlflow_adapter.api import (
    download_run_artifact,
    get_best_training_checkpoint,
    log_run_config,
    log_tile_preparation,
    mark_run_killed,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef, MLflowStartRunRequest
from mlsystem2.mlflow_adapter import _client
from mlsystem2.train.contracts import EpochMetrics


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
            train_optimizer_steps=2,
            train_skipped_optimizer_steps=0,
            val_loss=1.0,
            val_pixel_precision=0.0,
            val_pixel_recall=0.0,
            val_pixel_f1=0.0,
            val_positive_pixels=0,
            val_pred_positive_pixels=0,
            val_true_positive=0,
            val_false_positive=0,
            val_false_negative=0,
            epoch_time_sec=1.0,
        ),
    )

    assert ("tracking_uri", "file://mlruns") in calls
    assert ("start_run", "run-42") in calls
    assert ("metric", "train/loss", 1.0, 1, "run-42") in calls


def test_log_training_epoch_writes_optimizer_step_metrics(monkeypatch) -> None:
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
            train_loss_focal=0.2,
            train_loss_tversky=0.3,
            train_loss_bce=0.4,
            train_loss_dice=None,
            train_optimizer_steps=71,
            train_skipped_optimizer_steps=1,
            val_loss=1.0,
            val_pixel_precision=0.0,
            val_pixel_recall=0.0,
            val_pixel_f1=0.0,
            val_positive_pixels=0,
            val_pred_positive_pixels=0,
            val_true_positive=0,
            val_false_positive=0,
            val_false_negative=0,
            epoch_time_sec=1.0,
        ),
    )

    assert ("train/optimizer_steps", 71, 3) in logged
    assert ("train/skipped_optimizer_steps", 1, 3) in logged
    assert ("train/loss_focal", 0.2, 3) in logged
    assert ("train/loss_tversky", 0.3, 3) in logged
    assert ("train/loss_bce", 0.4, 3) in logged
    assert not any(item[0] == "train/loss_dice" for item in logged)
    assert ("val/best_threshold", 0.0, 3) in logged
    assert ("val/prob_mean", 0.0, 3) in logged


def test_log_training_epoch_writes_multiclass_metrics(monkeypatch) -> None:
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
            epoch=2,
            train_loss=1.0,
            train_optimizer_steps=1,
            train_skipped_optimizer_steps=0,
            val_loss=1.0,
            val_pixel_precision=0.5,
            val_pixel_recall=0.25,
            val_pixel_f1=0.333,
            val_positive_pixels=10,
            val_pred_positive_pixels=8,
            val_true_positive=4,
            val_false_positive=4,
            val_false_negative=6,
            val_macro_f1=0.333,
            val_mean_iou=0.25,
            val_pixel_accuracy=0.9,
            val_per_class_metrics={
                "class_a": {
                    "precision": 0.5,
                    "recall": 0.25,
                    "f1": 0.333,
                    "iou": 0.2,
                    "support_pixels": 10.0,
                }
            },
            epoch_time_sec=1.0,
        ),
    )

    assert ("val/macro_f1", 0.333, 2) in logged
    assert ("val/mean_iou", 0.25, 2) in logged
    assert ("val/pixel_accuracy", 0.9, 2) in logged
    assert ("val/class_a/f1", 0.333, 2) in logged
    assert ("val/class_a/iou", 0.2, 2) in logged
    assert ("val/class_a/precision", 0.5, 2) in logged
    assert ("val/class_a/recall", 0.25, 2) in logged
    assert ("val/class_a/support_pixels", 10.0, 2) in logged

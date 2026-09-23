from __future__ import annotations

from pathlib import Path

import pytest

from mlsystem2.mlflow_adapter.api import (
    download_run_artifact,
    get_best_training_checkpoint,
    get_finished_run_artifact,
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


def test_adapter_roundtrip_with_real_mlflow(tmp_path: Path, monkeypatch) -> None:
    """Проверить совместимость с установленным MLflow без подмены его клиента."""
    import mlflow

    from mlsystem2.mlflow_adapter import api
    from mlsystem2.mlflow_adapter.contracts import MLflowRunStatus

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    previous_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.delenv("MLSYSTEM2_MLFLOW_RUN_ID_FILE", raising=False)
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "Проверка совместимости MLflow"
    experiment_id = client.create_experiment(
        experiment_name, artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"mlsystem2-checkpoint-roundtrip")
    config = tmp_path / "train.yaml"
    config.write_text("model: segformer_b2\n", encoding="utf-8")
    dataset_name = "Проверка датасета"
    scene = "снимки/Канопус (1)"
    history = [
        EpochMetrics(
            epoch=epoch, train_loss=0.3, val_loss=0.2,
            val_best_threshold=threshold, val_best_threshold_pixel_f1=f1,
            val_best_threshold_precision=f1, val_best_threshold_recall=f1,
            learning_rate=0.001 / epoch, epoch_time_sec=1.0,
        )
        for epoch, f1, threshold in ((1, 0.8, 0.4), (2, 0.7, 0.6))
    ]
    try:
        run = api.start_run(MLflowStartRunRequest(
            enabled=True, tracking_uri=tracking_uri, experiment_name=experiment_name,
            dataset=dataset_name, run_name="Проверка записи и чтения",
        ))
        api.log_run_config(run, config)
        for epoch in history:
            api.log_training_epoch(run, epoch)
        result = TrainResult(
            history=history, epochs_total=2, training_time_sec=2,
            best_checkpoint_path=str(checkpoint), best_threshold=0.4,
            diagnostics={"flattened_params": {f"split.scenes.{scene}": 149}},
        )
        api.log_training_metrics(run, result)
        api.log_training_artifacts(run, result)
        assert api.get_usable_training_checkpoint(tracking_uri, run.run_id) is None
        api.end_run(run, MLflowRunStatus.FINISHED)

        stored = client.get_run(run.run_id)
        assert stored.info.status == "FINISHED"
        assert stored.data.tags["dataset"] == dataset_name
        assert list(stored.data.params.values()) == ["149"]
        assert stored.inputs.dataset_inputs[0].dataset.name == dataset_name
        assert [item.name for item in client.search_datasets(
            experiment_ids=[experiment_id],
        )] == [dataset_name]
        assert [item.step for item in client.get_metric_history(
            run.run_id, "val/best_threshold_pixel_f1",
        )] == [1, 2]
        for name in ("val/quality_f1", "val/quality_precision", "val/quality_recall"):
            assert name not in stored.data.metrics
            assert client.get_metric_history(run.run_id, name) == []
        assert [(item.step, item.value) for item in client.get_metric_history(
            run.run_id, "train/learning_rate",
        )] == [(1, 0.001), (2, 0.0005)]
        assert api.get_training_epoch_progress(tracking_uri, run.run_id).completed_epochs == 2
        best = api.get_usable_training_checkpoint(tracking_uri, run.run_id)
        assert best is not None
        assert (best.epoch, best.f1_score, best.threshold) == (1, 0.8, 0.4)
        assert api.get_finished_run_artifact(
            tracking_uri, run.run_id, "config/train_config.yaml",
        ) is not None
        download_dir = tmp_path / "download"
        download_dir.mkdir()
        downloaded = api.download_run_artifact(
            tracking_uri=tracking_uri, run_id=run.run_id,
            artifact_path="checkpoints/best.pt", dst_dir=download_dir,
        )
        assert Path(downloaded.local_path).read_bytes() == checkpoint.read_bytes()
        assert experiment_name in {item.name for item in api.list_experiments(tracking_uri)}
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        mlflow.set_tracking_uri(previous_tracking_uri)


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
    run = MLflowRunRef(
        run_id="disabled", experiment_name="test", tracking_uri="file://mlruns", active=False
    )

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


@pytest.mark.parametrize("quality_metric,metric_name", [
    ("pixel", "val/best_threshold_pixel_f1"),
    ("pixel", "val/macro_pixel_f1"),
    ("objects", "val/object_f1"),
    ("objects", "val/best_threshold_object_f1"),
    ("pixel", "val/quality_f1"),
    ("objects", "val/quality_f1"),
])
def test_checkpoint_reads_matching_f1_and_preserves_historical_quality(
    monkeypatch, quality_metric, metric_name,
) -> None:
    """Выбор эпохи сохраняет критерий и при равном F1 выбирает раннюю эпоху."""
    from types import SimpleNamespace

    values = {"val/best_threshold_pixel_f1": [0.95, 0.4, 0.5]}
    if metric_name == "val/quality_f1":
        values["val/macro_pixel_f1"] = [0.95, 0.4, 0.5]
        values["val/object_f1"] = [0.95, 0.4, 0.5]
    values[metric_name] = [0.6, 0.9, 0.9]
    values["val/best_threshold"] = [0.3, 0.6, 0.8]
    client = SimpleNamespace(
        get_run=lambda _: SimpleNamespace(
            data=SimpleNamespace(tags={"quality_metric": quality_metric}),
            info=SimpleNamespace(artifact_uri="s3://artifacts/run"),
        ),
        get_metric_history=lambda _, name: [
            SimpleNamespace(step=step, value=value)
            for step, value in enumerate(values.get(name, []), 1)
        ],
    )
    monkeypatch.setattr(_client, "_mlflow", lambda: SimpleNamespace(
        set_tracking_uri=lambda _: None,
        tracking=SimpleNamespace(MlflowClient=lambda: client),
    ))

    checkpoint = get_best_training_checkpoint("http://mlflow:5000/mlflow", "run")

    assert checkpoint is not None
    assert checkpoint.metric_name == metric_name
    assert checkpoint.quality_metric == quality_metric
    assert (checkpoint.epoch, checkpoint.f1_score, checkpoint.threshold) == (2, 0.9, 0.6)


# Проверяет статус запуска и фактическое наличие best.pt.
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


def test_get_finished_run_artifact_requires_finished_run_and_exact_file(monkeypatch) -> None:
    class Info:
        status = "FINISHED"
        artifact_uri = "s3://artifacts/run-1/artifacts"

    class Run:
        info = Info()

    class Artifact:
        path = "models/model.zip"
        is_dir = False

    class Client:
        def get_run(self, run_id: str):
            assert run_id == "run-1"
            return Run()

        def list_artifacts(self, run_id: str, path: str | None):
            assert (run_id, path) == ("run-1", "models")
            return [Artifact()]

    class MLflow:
        class tracking:
            @staticmethod
            def MlflowClient():
                return Client()

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            assert uri == "http://mlflow:5000"

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)

    artifact = get_finished_run_artifact(
        "http://mlflow:5000",
        "run-1",
        "models/model.zip",
    )

    assert artifact is not None
    assert artifact.artifact_path == "models/model.zip"
    assert artifact.artifact_uri == "s3://artifacts/run-1/artifacts/models/model.zip"


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
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

    log_run_config(run, config)

    assert logged == [("train:\n  epochs: 1\n", "config/train_config.yaml")]


def test_log_tile_preparation_uses_report_artifact_path(monkeypatch) -> None:
    logged: list[tuple[dict[str, object], str]] = []

    class MLflow:
        @staticmethod
        def log_dict(payload: dict[str, object], artifact_file: str) -> None:
            logged.append((payload, artifact_file))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

    log_tile_preparation(run, {"tile_size": 1024})

    assert logged == [({"tile_size": 1024}, "reports/tile_preparation.json")]


def test_log_dataset_artifacts_writes_files_under_dataset(tmp_path: Path, monkeypatch) -> None:
    scenes = tmp_path / "source-scenes.txt"
    scenes.write_text("scene-1\n", encoding="utf-8")
    annotation = tmp_path / "source-annotation.geojson"
    annotation.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    manifest = tmp_path / ".mlsystem2-dataset.json"
    manifest.write_text('{"task":"multiclass"}', encoding="utf-8")
    logged: list[tuple[str, str, str]] = []

    class MLflow:
        @staticmethod
        def log_artifact(path: str | Path, artifact_path: str) -> None:
            logged.append((Path(path).name, Path(path).read_text(encoding="utf-8"), artifact_path))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

    log_dataset_artifacts(
        run,
        {
            "scenes.txt": scenes,
            "annotation.geojson": annotation,
            "per_image/.mlsystem2-dataset.json": manifest,
        },
    )

    assert logged == [
        ("scenes.txt", "scene-1\n", "dataset"),
        ("annotation.geojson", '{"type":"FeatureCollection","features":[]}', "dataset"),
        (".mlsystem2-dataset.json", '{"task":"multiclass"}', "dataset/per_image"),
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
    run = MLflowRunRef(
        run_id="run-42", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

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
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

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
        ("val/best_threshold_pixel_f1", 0.6, 3),
        ("val/pixel_f1", 0.6, 3),
        ("val/pixel_precision", 0.7, 3),
        ("val/pixel_recall", 0.52, 3),
        ("val/best_threshold_precision", 0.7, 3),
        ("val/best_threshold_recall", 0.52, 3),
        ("train/epoch_time_sec", 1.0, 3),
    ]


@pytest.mark.parametrize("validation_performed,per_scene_metrics", [
    (True, []),
    (True, [{"scene_id": "снимок", "pixel_f1": 0.7}]),
    (False, []),
])
def test_learning_rate_is_logged_once_per_epoch(
    monkeypatch, validation_performed, per_scene_metrics,
) -> None:
    """LR сохраняется и без поснимочных метрик, и без validation."""
    logged: list[tuple[str, float, int]] = []

    class MLflow:
        @staticmethod
        def log_metric(name: str, value: float, step: int = 0) -> None:
            logged.append((name, value, step))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True,
    )
    _client.log_training_epoch(run, EpochMetrics(
        epoch=3, train_loss=1.0, val_loss=1.0 if validation_performed else None,
        validation_performed=validation_performed, val_per_scene_metrics=per_scene_metrics,
        learning_rate=0.0005, epoch_time_sec=1.0,
    ))

    assert [item for item in logged if item[0] == "train/learning_rate"] == [
        ("train/learning_rate", 0.0005, 3),
    ]


def test_log_training_epoch_multiclass_skips_empty_object_metrics(monkeypatch) -> None:
    logged: list[tuple[str, float, int]] = []

    class MLflow:
        @staticmethod
        def log_metric(name: str, value: float, step: int = 0) -> None:
            assert value is not None
            logged.append((name, value, step))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(
        run_id="run",
        experiment_name="test",
        tracking_uri="file://mlruns",
        active=True,
    )

    _client.log_training_epoch(
        run,
        EpochMetrics(
            epoch=1,
            train_loss=1.0,
            val_loss=1.0,
            val_macro_pixel_f1=0.4,
            val_micro_pixel_f1=0.5,
            val_per_class_metrics=[
                {
                    "slug": "flooding",
                    "precision": 0.3,
                    "recall": 0.4,
                    "f1": 0.34,
                    "iou": 0.2,
                },
                {
                    "slug": "waterlogging",
                    "precision": 0.5,
                    "recall": 0.6,
                    "f1": 0.54,
                    "iou": 0.37,
                },
            ],
            epoch_time_sec=1.0,
        ),
    )

    names = {name for name, _value, _step in logged}
    assert "val/macro_pixel_f1" in names
    assert "val/micro_pixel_f1" in names
    assert "val/class/flooding/f1" in names
    assert "val/class/waterlogging/iou" in names
    assert not any("object" in name for name in names)


def test_log_training_metrics_writes_train_best_hpo_metric(monkeypatch) -> None:
    logged: list[tuple[str, float, int | None]] = []

    class MLflow:
        @staticmethod
        def log_metric(name: str, value: float, step: int | None = None) -> None:
            logged.append((name, value, step))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True
    )

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
        ("train/stopped_early", 0, None),
        ("train/best_quality_f1", 0.6, None),
        ("train/best_threshold_pixel_f1", 0.6, None),
    ]


@pytest.mark.parametrize(
    "raw_key",
    [
        "model.split.purged_windows_by_scene.hlam2/"
        "KVI_07000_04503-02_KANOPUS_20181018_075153_8.L2.PMS.SCN01 (1)",
        "model.split.Захламнения/снимок [копия]:1",
        "model.split.снимок\\папка\nимя",
        "model." + "снимок" * 80,
        "/папка/снимок",
        "папка//снимок",
        "папка/../снимок",
        "..",
    ],
)
def test_safe_mlflow_params_accepts_scene_names(raw_key: str) -> None:
    from mlflow.utils.validation import _validate_param_name

    params = _client._safe_mlflow_params({raw_key: 149})

    assert len(params) == 1
    key, value = next(iter(params.items()))
    _validate_param_name(key)
    assert len(key) <= 240
    assert value == 149
    assert params == _client._safe_mlflow_params({raw_key: 149})


def test_safe_mlflow_params_preserves_distinct_and_existing_keys() -> None:
    first = "model.split.снимок (1)"
    second = "model.split.снимок [1]"
    existing = next(iter(_client._safe_mlflow_params({first: 1})))
    values = {
        first: 1,
        second: 2,
        existing: 3,
        "model." + "a" * 300 + "1": 4,
        "model." + "a" * 300 + "2": 5,
    }

    params = _client._safe_mlflow_params(values)

    assert len(params) == len(values)
    assert params[existing] == 3
    assert set(params.values()) == {1, 2, 3, 4, 5}
    assert params == _client._safe_mlflow_params(dict(reversed(list(values.items()))))


def test_safe_mlflow_params_keeps_safe_keys_and_value_contract() -> None:
    params = _client._safe_mlflow_params(
        {"train.pipeline_variant": "next_gen2", "split/сцена_1-2.3": 7,
         "config": {"режим": "проверка"}, "long": "я" * 600, "": "пропустить"}
    )

    assert params == {
        "train.pipeline_variant": "next_gen2",
        "split/сцена_1-2.3": 7,
        "config": '{"режим": "проверка"}',
        "long": "я" * 500,
    }


def test_training_artifacts_upload_checkpoints_with_parenthesized_scene_name(
    tmp_path: Path, monkeypatch,
) -> None:
    from mlflow.utils.validation import _validate_param_name

    scene = "hlam2/KVI_07000_04503-02_KANOPUS_20181018_075153_8.L2.PMS.SCN01 (1)"
    split = {"purged_windows_by_scene": {scene: 149}}
    reports: dict[str, object] = {}
    uploaded: list[tuple[str, str]] = []
    params: dict[str, object] = {}

    class MLflow:
        @staticmethod
        def log_dict(payload, artifact_file):
            reports[artifact_file] = payload

        @staticmethod
        def log_params(values):
            for key in values:
                _validate_param_name(key)
            params.update(values)

        @staticmethod
        def log_artifact(path, artifact_path):
            uploaded.append((Path(path).name, artifact_path))

    monkeypatch.setattr(_client, "_mlflow", lambda: MLflow)
    best, final = tmp_path / "best.pt", tmp_path / "final.pt"
    best.write_bytes(b"best")
    final.write_bytes(b"final")
    run = MLflowRunRef(
        run_id="run", experiment_name="test", tracking_uri="file://mlruns", active=True,
    )

    _client.log_training_artifacts(
        run,
        TrainResult(
            history=[], epochs_total=1, training_time_sec=1,
            best_checkpoint_path=str(best), final_checkpoint_path=str(final),
            diagnostics={
                "pipeline_variant": "next_gen2", "split_manifest": split,
                "flattened_params": {f"model.split.purged_windows_by_scene.{scene}": 149},
            },
        ),
    )

    assert list(params.values()) == [149]
    assert reports["reports/split_manifest.json"] == split
    assert uploaded == [("best.pt", "checkpoints"), ("final.pt", "checkpoints")]
    assert set(reports["reports/checkpoint_hashes.json"]) == {"best.pt", "final.pt"}

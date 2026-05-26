from __future__ import annotations

from pathlib import Path

from mlsystem2.mlflow_adapter.api import log_run_config, log_tile_preparation
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
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

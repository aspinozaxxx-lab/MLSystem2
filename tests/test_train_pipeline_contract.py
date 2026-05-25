from __future__ import annotations

import inspect
from pathlib import Path
from typing import get_type_hints

from mlsystem2.dataset_preparing.contracts import (
    DatasetPreparationReport,
    DatasetPreparationResult,
    PreparedDataset,
)
from mlsystem2.mlflow_adapter.contracts import MLflowRunRef
from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, ModelHandle, ModelSpec
from mlsystem2.settings.contracts import (
    DatasetSettings,
    InferenceSettings,
    MLflowSettings,
    RuntimeSettings,
    SystemSettings,
    TilePreparationSettings,
    TrainSettings,
)
from mlsystem2.train.contracts import EpochMetrics, TrainResult
from mlsystem2.train_pipeline.api import run_train_pipeline
from mlsystem2.train_pipeline import _runner
from mlsystem2.train_pipeline.contracts import TrainPipelineRequest, TrainPipelineResult


def test_run_train_pipeline_signature_uses_request_contract() -> None:
    signature = inspect.signature(run_train_pipeline)
    parameters = list(signature.parameters.values())
    hints = get_type_hints(run_train_pipeline)
    assert len(parameters) == 1
    assert parameters[0].name == "request"
    assert hints["request"] is TrainPipelineRequest


def test_train_pipeline_result_fields() -> None:
    assert {"status", "mlflow_run", "timings", "report"} <= set(TrainPipelineResult.model_fields)


def test_train_pipeline_request_has_only_run_name() -> None:
    assert set(TrainPipelineRequest.model_fields) == {"run_name"}


def test_train_pipeline_uses_load_checkpoint_branch() -> None:
    calls: list[str] = []
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b2", input_channels=4, output_channels=1),
        model=object(),
    )
    deps = _runner._PipelineDependencies(
        get_settings=lambda: _settings(initial_checkpoint_uri="/tmp/initial.pt"),
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="disabled",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=False,
        ),
        prepare_dataset=lambda request: _dataset_result(),
        create_tile_dataloader=lambda request: object(),
        create_model=lambda spec: calls.append("create_model") or model,
        load_checkpoint=lambda request: calls.append("load_checkpoint")
        or LoadedCheckpoint(
            model=model,
            artifact=CheckpointArtifact(uri=request.checkpoint_uri, format="torch_pt"),
        ),
        train_model=lambda request, progress_sink=None: _train_result(),
        log_dataset_preparation=lambda run, report: None,
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, metrics: None,
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert calls == ["load_checkpoint"]


def test_train_pipeline_logs_epoch_metrics_from_progress_sink() -> None:
    logged_epochs: list[int] = []
    metrics = _train_result().history[0]
    model = ModelHandle(
        spec=ModelSpec(name="segformer_b0", input_channels=4, output_channels=1),
        model=object(),
    )

    def train_model(request, progress_sink=None):
        if progress_sink is not None:
            from mlsystem2.train.contracts import TrainProgressEvent

            progress_sink(TrainProgressEvent(epoch=1, message="epoch_finished", metrics=metrics))
        return _train_result()

    deps = _runner._PipelineDependencies(
        get_settings=lambda: _settings(initial_checkpoint_uri=None),
        get_settings_path=lambda: Path("config.yaml"),
        start_run=lambda request: MLflowRunRef(
            run_id="run",
            experiment_name=request.experiment_name,
            tracking_uri=request.tracking_uri,
            active=True,
        ),
        prepare_dataset=lambda request: _dataset_result(),
        create_tile_dataloader=lambda request: object(),
        create_model=lambda spec: model,
        load_checkpoint=lambda request: None,
        train_model=train_model,
        log_dataset_preparation=lambda run, report: None,
        log_tile_preparation=lambda run, report: None,
        log_run_config=lambda run, config_path: None,
        log_training_epoch=lambda run, item: logged_epochs.append(item.epoch),
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert logged_epochs == [1]


def test_counting_loader_counts_observed_tiles_and_augmentations() -> None:
    class Dataset:
        source_rect_count = 1
        candidate_window_count = 5
        candidate_window_count_before_valid_filter = 7
        black_filtered_window_count = 2
        valid_footprint_stride = 64
        valid_footprint_valid_cells = 10
        valid_footprint_total_cells = 12
        uses_vrt_source_rects = True
        estimated_positive_tiles = 2
        estimated_negative_tiles = 3

        def __len__(self) -> int:
            return 5

    class Images:
        def __init__(self, batch_size: int) -> None:
            self.shape = (batch_size, 1, 4, 4)

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            yield (
                Images(2),
                object(),
                {"augmented_tile_count": 1, "positive_tile_count": 2},
            )
            yield Images(1), object()

        def __len__(self) -> int:
            return 2

    loader = _runner._CountingLoader(Loader(), "train")

    assert len(list(loader)) == 2
    assert loader.snapshot() == {
        "tile_count": 5,
        "batch_count": 2,
        "source_rect_count": 1,
        "candidate_window_count": 5,
        "candidate_window_count_before_valid_filter": 7,
        "black_filtered_window_count": 2,
        "valid_footprint_stride": 64,
        "valid_footprint_valid_cells": 10,
        "valid_footprint_total_cells": 12,
        "uses_vrt_source_rects": True,
        "estimated_positive_tiles": 2,
        "estimated_negative_tiles": 3,
        "sampling_mode": "sequential",
        "positive_factor_used": None,
        "is_diagnostic_sampling": False,
        "observed_batches": 2,
        "observed_tiles": 3,
        "observed_positive_tiles": 2,
        "observed_augmented_tiles": 1,
        "observed_real_tiles": 2,
        "warnings": [],
    }


def _settings(*, initial_checkpoint_uri: str | None) -> SystemSettings:
    return SystemSettings(
        runtime=RuntimeSettings(
            project_root=".",
            scratch_root="./scratch",
            logs_root="./logs",
            cleanup_scratch_after_mlflow_log=False,
        ),
        dataset=DatasetSettings(
            images_dir="./images",
            scenes_file="./scenes.txt",
            annotation_file="./annotations.geojson",
            val_fraction=0.2,
        ),
        tile_preparation=TilePreparationSettings(tile_size=512, stride=512),
        train=TrainSettings(
            model_name="segformer_b2",
            input_channels=4,
            output_channels=1,
            pretrained=False,
            initial_checkpoint_uri=initial_checkpoint_uri,
            epochs=1,
            batch_size=1,
            device="cpu",
            learning_rate=0.00001,
            weight_decay=0.0001,
            loss="bce_dice",
            early_stopping_patience=1,
        ),
        inference=InferenceSettings(
            checkpoint_uri="./checkpoint.pt",
            threshold=0.5,
            batch_size=1,
            device="cpu",
        ),
        mlflow=MLflowSettings(
            enabled=False,
            tracking_uri="./mlruns",
            experiment_name="test",
        ),
    )


def _dataset_result() -> DatasetPreparationResult:
    return DatasetPreparationResult(
        dataset=PreparedDataset(
            train_vrt_xml="<VRTDataset />",
            val_vrt_xml="<VRTDataset />",
            annotation_file="./annotations.geojson",
        ),
        report=DatasetPreparationReport(
            status="ok",
            scenes_total=0,
            scenes_found=0,
            objects_total=0,
            train_scenes_count=0,
            train_objects_count=0,
            val_scenes_count=0,
            val_objects_count=0,
            scenes=[],
            missing_files=[],
            errors=[],
        ),
    )


def _train_result() -> TrainResult:
    return TrainResult(
        history=[
            EpochMetrics(
                epoch=1,
                train_loss=1.0,
                train_optimizer_steps=1,
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
                epoch_time_sec=0.1,
            )
        ],
        epochs_total=1,
        training_time_sec=0.1,
        best_checkpoint_path="/tmp/best.pt",
        final_checkpoint_path="/tmp/final.pt",
    )

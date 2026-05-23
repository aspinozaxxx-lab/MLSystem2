from __future__ import annotations

import inspect
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
        train_model=lambda request: _train_result(),
        log_dataset_preparation=lambda run, report: None,
        log_training_metrics=lambda run, result: None,
        log_training_artifacts=lambda run, result: None,
        log_timing_report=lambda run, report: None,
        log_pipeline_report=lambda run, report: None,
        end_run=lambda run, status: None,
    )

    result = _runner.run_train_pipeline(TrainPipelineRequest(), dependencies=deps)

    assert result.status.value == "succeeded"
    assert calls == ["load_checkpoint"]


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
                val_loss=1.0,
                val_pixel_precision=0.0,
                val_pixel_recall=0.0,
                val_pixel_f1=0.0,
                epoch_time_sec=0.1,
            )
        ],
        epochs_total=1,
        training_time_sec=0.1,
        best_checkpoint_path="/tmp/best.pt",
        final_checkpoint_path="/tmp/final.pt",
    )

from __future__ import annotations

import importlib


EXPECTED_API = {
    "settings.api": ["load_settings", "get_settings"],
    "dataset_preparing.api": ["prepare_dataset"],
    "tile_preparation.api": ["create_tile_dataloader"],
    "models.api": ["list_supported_models", "create_model", "load_checkpoint", "save_checkpoint"],
    "metrics.api": ["compute_pixel_f1", "summarize_epoch_metrics"],
    "train.api": ["train_model"],
    "mlflow_adapter.api": [
        "start_run",
        "log_dataset_preparation",
        "log_training_epoch",
        "log_training_metrics",
        "log_training_artifacts",
        "log_timing_report",
        "log_pipeline_report",
        "end_run",
    ],
    "train_pipeline.api": ["run_train_pipeline"],
    "inference.api": ["run_inference"],
    "inference_pipeline.api": ["run_inference_pipeline"],
}


def test_public_api_all_is_exact() -> None:
    for module_name, expected in EXPECTED_API.items():
        module = importlib.import_module(f"mlsystem2.{module_name}")
        assert list(module.__all__) == expected

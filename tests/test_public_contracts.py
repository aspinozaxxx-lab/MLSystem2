from __future__ import annotations

import importlib


EXPECTED_API = {
    "settings.api": ["load_settings", "get_settings", "get_settings_path"],
    "dataset_preparing.api": ["prepare_dataset", "resolve_scene_images"],
    "tile_preparation.api": ["create_tile_dataloader"],
    "models.api": ["list_supported_models", "create_model", "load_checkpoint", "save_checkpoint"],
    "metrics.api": ["compute_object_f1", "compute_pixel_f1", "summarize_epoch_metrics"],
    "train.api": ["train_model"],
    "mlflow_adapter.api": [
        "list_experiments",
        "create_experiment",
        "get_best_training_checkpoint",
        "get_training_epoch_progress",
        "download_run_artifact",
        "start_run",
        "log_dataset_preparation",
        "log_dataset_artifacts",
        "log_tile_preparation",
        "log_run_config",
        "log_training_epoch",
        "log_training_metrics",
        "log_training_artifacts",
        "log_timing_report",
        "log_pipeline_report",
        "end_run",
        "mark_run_killed",
    ],
    "train_pipeline.api": ["run_train_pipeline"],
    "inference.api": ["run_inference"],
    "inference_pipeline.api": ["run_inference_pipeline"],
    "training_ui_api.api": ["create_app", "get_openapi_schema", "main"],
}


def test_public_api_all_is_exact() -> None:
    for module_name, expected in EXPECTED_API.items():
        module = importlib.import_module(f"mlsystem2.{module_name}")
        assert list(module.__all__) == expected

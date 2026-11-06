"""Оценка подгонки пакета тем же исполнителем, что считает тестовый F1."""

from __future__ import annotations

import json
from pathlib import Path

from mlsystem2.models.api import load_checkpoint
from mlsystem2.models.contracts import CheckpointArtifact, LoadedCheckpoint, LoadCheckpointRequest
from mlsystem2.train.contracts import TrainError

from ._pseudo_runner import _metric_values, run_test_sample_f1


class _PackagePause:
    def __init__(self, pause):
        self.pause_if_requested = pause

    def close(self):
        pass


def create_evaluator(config_path):
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    source = load_checkpoint(LoadCheckpointRequest(
        checkpoint_uri=configuration["initial_checkpoint_uri"], map_location="cpu",
    ))
    # Сотни тысяч окон исходного split для оценки не нужны.
    metadata = {key: value for key, value in source.artifact.metadata.items() if key in {
        "confidence_threshold", "val_best_threshold", "task", "class_schema",
        "sample_size", "inference_context", "inference_core_size",
    }}
    del source

    def evaluate(model, epoch, pause):
        pause()
        effective_metadata = {
            **metadata, "pipeline_variant": configuration["pipeline_variant"],
            "sample_size": configuration["tile_size"],
            "inference_context": configuration["context"],
            "inference_core_size": configuration["tile_size"] - 2 * configuration["context"],
        }
        loaded = LoadedCheckpoint(model=model, artifact=CheckpointArtifact(
            uri=configuration["initial_checkpoint_uri"], format="torch_pt",
            metadata=effective_metadata,
        ))
        report = run_test_sample_f1(
            {**configuration, "run_root": str(Path(configuration["evaluations_root"]) / f"epoch_{epoch:04d}"),
             "save_test_f1_view": False},
            loaded_checkpoint=loaded, pause_controller_override=_PackagePause(pause),
        )
        if report["status"] != "ok" or report["processed"] != len(configuration["tiles"]):
            raise TrainError("Не удалось полностью оценить обучаемый пакет.")
        pixel = _metric_values(report)
        objects = _metric_values({name: report[f"object_{name}"] for name in (
            "true_positive", "false_positive", "false_negative",
        )})
        structured = report.get("metrics")
        quality = objects if configuration["quality_metric"] == "objects" else pixel
        evaluation = {
            "scope": "training_package", "independent": False, "epoch": epoch,
            "test_sample_id": configuration["test_sample_id"],
            "test_sample_revision": configuration["test_sample_revision"],
            "inference_config_hash": configuration["inference_config_hash"],
            "inference_template_id": configuration["inference_template_id"],
            "inference_template_version": configuration["inference_template_version"],
            "postprocess_config": configuration["postprocess_config"],
            "pixel": pixel, "objects": objects,
            "quality_counts": {short: quality[long] for short, long in (
                ("tp", "true_positive"), ("fp", "false_positive"), ("fn", "false_negative"),
            )},
            "report": report,
        }
        return {
            "quality_f1": quality["f1"], "quality_precision": quality["precision"],
            "quality_recall": quality["recall"], "threshold": report["threshold"],
            "pixel_f1": pixel["f1"], "pixel_precision": pixel["precision"],
            "pixel_recall": pixel["recall"], "object_f1": objects["f1"],
            "object_precision": objects["precision"], "object_recall": objects["recall"],
            "per_scene_metrics": report["tiles"],
            "object_separation": model.spec.parameters.get("object_separation", {}),
            "package_evaluation": evaluation,
            **({"macro_pixel_f1": structured["pixel"]["macro"]["f1"],
                "micro_pixel_f1": structured["pixel"]["micro"]["f1"],
                "per_class_metrics": list(structured["pixel"]["per_class"].values())} if structured else {}),
        }

    return evaluate

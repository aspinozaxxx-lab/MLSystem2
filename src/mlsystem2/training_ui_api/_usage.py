"""Учёт востребованности без содержимого запросов и служебного опроса."""

import hashlib
import hmac

from ._config import TrainingUIAPIConfig


def metrica_user_id(username: str, config: TrainingUIAPIConfig) -> str:
    """Стабильный непрозрачный ID; логин остаётся только в Гровике."""
    return hmac.new(
        config.session_secret.encode(), f"grovika-usage:{username}".encode(), hashlib.sha256,
    ).hexdigest()[:32]


def action_for_request(method: str, route: str) -> str | None:
    """Выделить явное действие; не учитывать чтение, опрос и автосохранение."""
    if method == "GET":
        return "file_download" if route.endswith(("/download", "/download-by-type")) else None
    if method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return None
    path = route.removeprefix("/api/v1/")
    exact = {
        "training-jobs": "training_start", "feedback": "feedback_send",
        "results/training/{result_id}/continue": "training_continue",
        "results/datasets/{dataset_key}/pseudo-markup": "pseudo_create",
        "results/datasets/{dataset_key}/test-f1": "test_f1_calculate",
        "results/training/{result_id}/primary": "primary_select",
        "dataset-classes": "class_create", "dataset-classes/{class_key}": "class_save",
        "dataset-classes/{class_key}/primary-dataset": "primary_select",
        "dataset-classes/{class_key}/inference-template": "template_assign",
        "custom-datasets": "dataset_create", "managed-datasets": "dataset_create",
        "managed-datasets/compose": "dataset_create", "managed-datasets/{dataset_key}": "class_save",
        "test-sample-batches": "test_create", "test-samples": "test_create",
        "scene-list-export": "scene_export", "markup-export": "markup_export",
        "test-samples/download": "markup_export",
    }
    if path == "feedback" and method != "POST":
        return None
    if path in exact:
        return exact[path]
    if path.startswith(("training-templates", "inference-templates")):
        prefix = "training_template" if path.startswith("training-templates") else "inference_template"
        operation = {"POST": "create", "DELETE": "delete"}.get(method, "save")
        return f"{prefix}_{operation}"
    if path.startswith("automation/"):
        return "automation_save"
    if path.startswith("queues/") and path.endswith("/enabled"):
        return "queue_toggle"
    if path.startswith("jobs/"):
        if path.endswith(("/move-up", "/move-down")):
            return "queue_move"
        if path.endswith("/stop-and-save-best"):
            return "job_finish"
        return "job_cancel" if method == "DELETE" else None
    if path.startswith("test-sample-batches/{batch_id}"):
        return "queue_move" if path.endswith("/move") else "job_cancel"
    if path.endswith("/triton-zip"):
        return "model_export"
    if path.startswith("dataset-editor/datasets/"):
        suffix = path.removeprefix("dataset-editor/datasets/{dataset_key}")
        return {
            "/copy": "dataset_copy", "/drafts/import": "dataset_import",
            "/drafts/publish": "dataset_publish", "/rebuild": "dataset_rebuild",
            "/scenes/{annotation_name}/pseudo-markup": "pseudo_create",
            "": "dataset_delete" if method == "DELETE" else None,
        }.get(suffix)
    if path.startswith("test-samples/{sample_id}"):
        suffix = path.removeprefix("test-samples/{sample_id}")
        if not suffix:
            return "test_delete" if method == "DELETE" else "test_save"
        return {
            "/primary": "primary_select", "/evaluate": "test_f1_calculate",
            "/pseudo-markup": "pseudo_create", "/optimize": "test_optimize",
            "/download": "markup_export", "/merge-annotations": "test_save",
            "/tiles/{tile_index}": "test_save",
        }.get(suffix)
    if path == "results/pseudo-markup/{result_id}" and method == "DELETE":
        return "pseudo_delete"
    return None


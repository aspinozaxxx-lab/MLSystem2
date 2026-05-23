"""Ввод-вывод локальных чекпойнтов моделей."""

from __future__ import annotations

from pathlib import Path

from .contracts import CheckpointArtifact, LoadCheckpointRequest, LoadedCheckpoint, SaveCheckpointRequest
from .contracts import ModelsError
from ._factory import create_model


def load_checkpoint(request: LoadCheckpointRequest) -> LoadedCheckpoint:
    try:
        import torch
    except ImportError as exc:
        raise ModelsError(
            "Для загрузки checkpoint требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc

    checkpoint_path = Path(request.checkpoint_uri)
    if not checkpoint_path.is_file():
        raise ModelsError(f"Checkpoint не найден: {checkpoint_path}")

    try:
        payload = torch.load(checkpoint_path, map_location=request.map_location)
    except Exception as exc:
        raise ModelsError(f"Не удалось прочитать checkpoint: {checkpoint_path}") from exc

    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ModelsError(f"Некорректный checkpoint: {checkpoint_path}")

    spec = request.model_spec
    if spec is None:
        raw_spec = payload.get("model_spec")
        if not isinstance(raw_spec, dict):
            raise ModelsError("Checkpoint не содержит model_spec, а request.model_spec не задан.")
        try:
            from .contracts import ModelSpec

            spec = ModelSpec.model_validate(raw_spec)
        except Exception as exc:
            raise ModelsError("Не удалось восстановить model_spec из checkpoint.") from exc

    model = create_model(spec)
    try:
        model.model.load_state_dict(payload["model_state_dict"])
    except Exception as exc:
        raise ModelsError(f"Не удалось загрузить веса checkpoint: {checkpoint_path}") from exc

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return LoadedCheckpoint(
        model=model,
        artifact=CheckpointArtifact(
            uri=str(checkpoint_path),
            format="torch_pt",
            metadata=metadata,
        ),
    )


def save_checkpoint(request: SaveCheckpointRequest) -> CheckpointArtifact:
    try:
        import torch
    except ImportError as exc:
        raise ModelsError(
            "Для сохранения checkpoint требуется optional dependency torch. "
            "Установите пакет через `pip install -e .[torch]`."
        ) from exc

    checkpoint_path = Path(request.checkpoint_uri)
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": request.model.model.state_dict(),
                "model_spec": request.model.spec.model_dump(mode="json"),
                "metadata": request.metadata,
            },
            checkpoint_path,
        )
    except Exception as exc:
        raise ModelsError(f"Не удалось сохранить checkpoint: {checkpoint_path}") from exc

    return CheckpointArtifact(
        uri=str(checkpoint_path),
        format="torch_pt",
        metadata=request.metadata,
    )

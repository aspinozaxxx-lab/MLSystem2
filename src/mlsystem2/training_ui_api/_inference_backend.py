"""Выбор единственного backend инференса для задания Training UI."""

from __future__ import annotations

from typing import Any


PYTORCH_INFERENCE_BACKEND = "pytorch_one_off"
GEOALERT_INFERENCE_BACKEND = "geoalert_workflow_engine"


def inference_backend_for_imagery(imagery_type: object) -> str:
    """Ортофото обрабатывается штатным Workflow Engine, Канопус — совместимым путём."""

    return (
        GEOALERT_INFERENCE_BACKEND
        if str(imagery_type or "").strip().casefold() == "ortho"
        else PYTORCH_INFERENCE_BACKEND
    )


def configured_inference_backend(config: dict[str, Any]) -> str:
    """Вернуть backend из snapshot, не меняя уже поставленные в очередь задания."""

    value = str(config.get("inference_backend") or "").strip()
    if value:
        return value
    return inference_backend_for_imagery(
        config.get("model_imagery_type") or config.get("imagery_type")
    )


__all__ = [
    "GEOALERT_INFERENCE_BACKEND",
    "PYTORCH_INFERENCE_BACKEND",
    "configured_inference_backend",
    "inference_backend_for_imagery",
]

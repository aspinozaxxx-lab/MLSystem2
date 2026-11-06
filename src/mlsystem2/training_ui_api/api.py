"""Публичный фасад FastAPI-сервиса training UI."""

from __future__ import annotations

from typing import Any

from ._app import create_app as _create_app
from ._app import main as _main
from ._app import worker_main as _worker_main
from .contracts import NativeTileInferenceRequest, NativeTileInferenceResult
from mlsystem2.train.contracts import TrainingPackageEvaluator


def create_app() -> Any:
    return _create_app()


def get_openapi_schema() -> dict[str, Any]:
    return create_app().openapi()


def main() -> None:
    _main()


def worker_main() -> None:
    _worker_main()


def infer_native_tile(request: NativeTileInferenceRequest) -> NativeTileInferenceResult:
    from ._pseudo_runner import _infer_native_tile

    return _infer_native_tile(request)


def create_training_package_evaluator(config_path: str) -> TrainingPackageEvaluator:
    """Собрать оценщик зафиксированного пакета для композиции CLI-обучения."""
    from ._training_package_fit import create_evaluator

    return create_evaluator(config_path)


__all__ = ["create_app", "get_openapi_schema", "main", "worker_main", "infer_native_tile", "create_training_package_evaluator"]


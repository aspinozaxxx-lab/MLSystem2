"""Публичный фасад FastAPI-сервиса training UI."""

from __future__ import annotations

from typing import Any

from ._app import create_app as _create_app
from ._app import main as _main


def create_app() -> Any:
    return _create_app()


def get_openapi_schema() -> dict[str, Any]:
    return create_app().openapi()


def main() -> None:
    _main()


__all__ = ["create_app", "get_openapi_schema", "main"]


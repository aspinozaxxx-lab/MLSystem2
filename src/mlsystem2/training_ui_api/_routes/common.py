"""Route registration helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._config import TrainingUIAPIConfig


@dataclass(frozen=True)
class RouteContext:
    config: TrainingUIAPIConfig
    get_db: Callable[[], Iterator[Session]]
    authenticated: Callable[[Request], str]


__all__ = ["RouteContext"]

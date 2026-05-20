"""Вспомогательные функции замера времени."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from .contracts import ModuleTiming

T = TypeVar("T")


def timed_call(module: str, func: Callable[[], T]) -> tuple[T, ModuleTiming]:
    started = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - started
    return result, ModuleTiming(module=module, elapsed_sec=elapsed)


def elapsed_since(started: float) -> float:
    return time.perf_counter() - started


def now() -> float:
    return time.perf_counter()

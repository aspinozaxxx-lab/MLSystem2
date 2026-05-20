"""Публичный фасад настроек."""

from __future__ import annotations

from pathlib import Path

from ._loader import load_settings as _load_settings
from .contracts import SystemSettings


def load_settings(path: str | Path) -> SystemSettings:
    return _load_settings(path)


__all__ = ["load_settings"]

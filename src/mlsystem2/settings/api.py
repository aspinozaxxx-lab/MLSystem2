"""Публичный фасад настроек."""

from __future__ import annotations

from pathlib import Path

from ._loader import load_settings as _load_settings
from .contracts import SettingsError, SystemSettings


_CURRENT_SETTINGS: SystemSettings | None = None


def load_settings(path: str | Path) -> SystemSettings:
    global _CURRENT_SETTINGS
    _CURRENT_SETTINGS = _load_settings(path)
    return _CURRENT_SETTINGS


def get_settings() -> SystemSettings:
    if _CURRENT_SETTINGS is None:
        raise SettingsError(
            "Настройки не инициализированы. Сначала вызовите settings.api.load_settings(...)."
        )
    return _CURRENT_SETTINGS


__all__ = ["load_settings", "get_settings"]

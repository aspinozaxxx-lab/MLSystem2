"""Загрузчик YAML-настроек."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .contracts import SettingsError, SystemSettings


def load_settings(path: str | Path) -> SystemSettings:
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Файл настроек не существует: {settings_path}")
    if not settings_path.is_file():
        raise SettingsError(f"Путь настроек не является файлом: {settings_path}")

    try:
        with settings_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise SettingsError(f"Не удалось прочитать файл настроек: {settings_path}") from exc
    except yaml.YAMLError as exc:
        raise SettingsError(f"Не удалось разобрать YAML-настройки: {settings_path}") from exc

    if not isinstance(payload, dict):
        raise SettingsError(f"Файл настроек должен содержать словарь: {settings_path}")

    try:
        return SystemSettings.model_validate(payload)
    except ValidationError as exc:
        raise SettingsError(f"Некорректные настройки в {settings_path}: {exc}") from exc

"""Загрузчик YAML-настроек."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .contracts import SettingsError, SystemSettings


def load_settings(path: str | Path, run_path: str | Path | None = None) -> SystemSettings:
    payload = _read_yaml_dict(path, label="настроек приложения")
    if run_path is not None:
        run_payload = _read_yaml_dict(run_path, label="задания запуска")
        payload = _deep_merge(payload, run_payload)

    try:
        return SystemSettings.model_validate(payload)
    except ValidationError as exc:
        source = f"{path} + {run_path}" if run_path is not None else str(path)
        raise SettingsError(f"Некорректные настройки в {source}: {exc}") from exc


def _read_yaml_dict(path: str | Path, *, label: str) -> dict:
    settings_path = Path(path)
    if not settings_path.exists():
        raise SettingsError(f"Файл {label} не существует: {settings_path}")
    if not settings_path.is_file():
        raise SettingsError(f"Путь {label} не является файлом: {settings_path}")

    try:
        with settings_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except OSError as exc:
        raise SettingsError(f"Не удалось прочитать файл {label}: {settings_path}") from exc
    except yaml.YAMLError as exc:
        raise SettingsError(f"Не удалось разобрать YAML {label}: {settings_path}") from exc

    if not isinstance(payload, dict):
        raise SettingsError(f"Файл {label} должен содержать словарь: {settings_path}")
    return payload


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result

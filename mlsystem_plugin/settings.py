"""Настройки QGIS-плагина."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qgis.PyQt.QtCore import QStandardPaths
from qgis.core import QgsProject, QgsSettings


_ROOT = "mlsystem2/pseudolabel"
SERVER_URL = "https://grovika.ru"
SERVER_USERNAME = "mlsystem"
_PASSWORD_KEY = f"{_ROOT}/password"


@dataclass(frozen=True)
class PluginSettings:
    """Минимальные сохраняемые настройки плагина."""

    request_timeout_ms: int = 30_000
    poll_interval_ms: int = 2_000
    max_part_area_m2: float = 1_000_000.0
    min_area_m2: float = 100.0
    min_confidence: float = 0.0


def load_settings() -> PluginSettings:
    """Загрузить настройки из QgsSettings."""

    settings = QgsSettings()
    legacy_min_area = settings.value(f"{_ROOT}/min_part_area_m2", 100.0)
    for legacy_key in (
        "mapping",
        "mapping_version",
        "auto_split",
        "token",
        "server_url",
        "username",
    ):
        settings.remove(f"{_ROOT}/{legacy_key}")
    return PluginSettings(
        request_timeout_ms=int(settings.value(f"{_ROOT}/request_timeout_ms", 30_000)),
        poll_interval_ms=int(settings.value(f"{_ROOT}/poll_interval_ms", 2_000)),
        max_part_area_m2=float(settings.value(f"{_ROOT}/max_part_area_m2", 1_000_000.0)),
        min_area_m2=float(settings.value(f"{_ROOT}/min_area_m2", legacy_min_area)),
        min_confidence=float(settings.value(f"{_ROOT}/min_confidence", 0.0)),
    )


def load_connection_password() -> str:
    """Загрузить пароль, подготовленный локально для профиля QGIS."""

    return str(QgsSettings().value(_PASSWORD_KEY, ""))


def save_settings(value: PluginSettings) -> None:
    """Сохранить несекретные настройки плагина."""

    settings = QgsSettings()
    settings.remove(f"{_ROOT}/token")
    settings.remove(f"{_ROOT}/min_part_area_m2")
    for key, item in value.__dict__.items():
        settings.setValue(f"{_ROOT}/{key}", item)


def session_directory(project: QgsProject | None = None) -> Path:
    """Определить постоянный каталог локальных результатов распознавания."""

    project = project or QgsProject.instance()
    filename = project.fileName()
    if filename:
        root = Path(filename).resolve().parent / ".mlsystem2" / "review_sessions"
    else:
        standard_locations = getattr(QStandardPaths, "StandardLocation", QStandardPaths)
        base = QStandardPaths.writableLocation(standard_locations.AppDataLocation)
        root = Path(base) / "MLSystem2" / "review_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


__all__ = [
    "PluginSettings",
    "SERVER_URL",
    "SERVER_USERNAME",
    "load_connection_password",
    "load_settings",
    "save_settings",
    "session_directory",
]

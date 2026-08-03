"""Nastroiki QGIS-plagina."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qgis.PyQt.QtCore import QStandardPaths
from qgis.core import QgsProject, QgsSettings


_ROOT = "mlsystem2/pseudolabel"
_MAPPING_KEYS = ("class_id", "source", "confidence", "candidate_id", "model_version")


@dataclass(frozen=True)
class PluginSettings:
    """Minimalnye sohranyaemye nastroiki plagina."""

    server_url: str = "http://127.0.0.1:8091"
    token: str = ""
    request_timeout_ms: int = 30_000
    poll_interval_ms: int = 2_000
    max_part_area_m2: float = 1_000_000.0
    min_part_area_m2: float = 100.0
    auto_split: bool = False


def load_settings() -> PluginSettings:
    """Zagruzit nastroiki iz QgsSettings."""

    settings = QgsSettings()
    return PluginSettings(
        server_url=str(settings.value(f"{_ROOT}/server_url", "http://127.0.0.1:8091")),
        token=str(settings.value(f"{_ROOT}/token", "")),
        request_timeout_ms=int(settings.value(f"{_ROOT}/request_timeout_ms", 30_000)),
        poll_interval_ms=int(settings.value(f"{_ROOT}/poll_interval_ms", 2_000)),
        max_part_area_m2=float(settings.value(f"{_ROOT}/max_part_area_m2", 1_000_000.0)),
        min_part_area_m2=float(settings.value(f"{_ROOT}/min_part_area_m2", 100.0)),
        auto_split=str(settings.value(f"{_ROOT}/auto_split", "false")).lower() == "true",
    )


def save_settings(value: PluginSettings) -> None:
    """Sohranit nastroiki bez vyvoda tokena v log."""

    settings = QgsSettings()
    for key, item in value.__dict__.items():
        settings.setValue(f"{_ROOT}/{key}", item)


def load_field_mapping(role: str) -> dict[str, str]:
    """Zagruzit sopostavlenie polei celevoi roli."""

    settings = QgsSettings()
    return {
        key: str(settings.value(f"{_ROOT}/mapping/{role}/{key}", ""))
        for key in _MAPPING_KEYS
    }


def save_field_mapping(role: str, mapping: dict[str, str]) -> None:
    """Sohranit sopostavlenie sushchestvuyushchih polei."""

    settings = QgsSettings()
    for key in _MAPPING_KEYS:
        settings.setValue(f"{_ROOT}/mapping/{role}/{key}", mapping.get(key, ""))


def session_directory(project: QgsProject | None = None) -> Path:
    """Opredelit postoyannyi katalog sessii proverki."""

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
    "load_field_mapping",
    "load_settings",
    "save_field_mapping",
    "save_settings",
    "session_directory",
]

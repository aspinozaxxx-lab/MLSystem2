"""Nastroiki QGIS-plagina."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qgis.PyQt.QtCore import QStandardPaths
from qgis.core import QgsProject, QgsSettings, QgsVectorLayer


_ROOT = "mlsystem2/pseudolabel"
_MAPPING_KEYS = ("class_id", "source", "confidence", "candidate_id", "model_version")
_MAPPING_VERSION = 2
_DISABLED_FIELD = "__none__"
_FIELD_ALIASES = {
    "class_id": ("classid", "classname", "class", "класс", "имякласса"),
    "source": ("sourceimageids", "sourceids", "source", "источник"),
    "confidence": ("confidence", "score", "уверенность"),
    "candidate_id": ("candidateid", "candidateuuid"),
    "model_version": ("modelversion", "version", "версиямодели"),
}
SERVER_URL = "https://grovika.ru"
SERVER_USERNAME = "mlsystem"
_PASSWORD_KEY = f"{_ROOT}/password"


@dataclass(frozen=True)
class PluginSettings:
    """Minimalnye sohranyaemye nastroiki plagina."""

    request_timeout_ms: int = 30_000
    poll_interval_ms: int = 2_000
    max_part_area_m2: float = 1_000_000.0
    min_part_area_m2: float = 100.0
    min_confidence: float = 0.0
    auto_split: bool = False


def load_settings() -> PluginSettings:
    """Zagruzit nastroiki iz QgsSettings."""

    settings = QgsSettings()
    mapping_version = int(settings.value(f"{_ROOT}/mapping_version", 0))
    if mapping_version < _MAPPING_VERSION:
        settings.remove(f"{_ROOT}/mapping")
        settings.setValue(f"{_ROOT}/mapping_version", _MAPPING_VERSION)
    settings.remove(f"{_ROOT}/token")
    settings.remove(f"{_ROOT}/server_url")
    settings.remove(f"{_ROOT}/username")
    return PluginSettings(
        request_timeout_ms=int(settings.value(f"{_ROOT}/request_timeout_ms", 30_000)),
        poll_interval_ms=int(settings.value(f"{_ROOT}/poll_interval_ms", 2_000)),
        max_part_area_m2=float(settings.value(f"{_ROOT}/max_part_area_m2", 1_000_000.0)),
        min_part_area_m2=float(settings.value(f"{_ROOT}/min_part_area_m2", 100.0)),
        min_confidence=float(settings.value(f"{_ROOT}/min_confidence", 0.0)),
        auto_split=str(settings.value(f"{_ROOT}/auto_split", "false")).lower() == "true",
    )


def load_connection_password() -> str:
    """Загрузить пароль, подготовленный локально для профиля QGIS."""

    return str(QgsSettings().value(_PASSWORD_KEY, ""))


def save_settings(value: PluginSettings) -> None:
    """Сохранить несекретные настройки плагина."""

    settings = QgsSettings()
    settings.remove(f"{_ROOT}/token")
    for key, item in value.__dict__.items():
        settings.setValue(f"{_ROOT}/{key}", item)


def load_field_mapping(
    role: str,
    layer: QgsVectorLayer | None = None,
) -> dict[str, str]:
    """Загрузить ручное сопоставление и дополнить его по схеме слоя."""

    settings = QgsSettings()
    automatic = automatic_field_mapping(layer) if layer is not None else {}
    result: dict[str, str] = {}
    for key in _MAPPING_KEYS:
        path = f"{_ROOT}/mapping/{role}/{key}"
        saved = str(settings.value(path, ""))
        if saved == _DISABLED_FIELD:
            result[key] = ""
        elif layer is not None and saved and layer.fields().indexOf(saved) >= 0:
            result[key] = saved
        else:
            result[key] = automatic.get(key, saved if layer is None else "")
    return result


def save_field_mapping(role: str, mapping: dict[str, str]) -> None:
    """Sohranit sopostavlenie sushchestvuyushchih polei."""

    settings = QgsSettings()
    for key in _MAPPING_KEYS:
        settings.setValue(
            f"{_ROOT}/mapping/{role}/{key}",
            mapping.get(key) or _DISABLED_FIELD,
        )


def automatic_field_mapping(layer: QgsVectorLayer) -> dict[str, str]:
    """Сопоставить известные варианты имён без изменения схемы слоя."""

    normalized_fields = {_normalized_field_name(field.name()): field.name() for field in layer.fields()}
    return {
        key: next(
            (normalized_fields[alias] for alias in aliases if alias in normalized_fields),
            "",
        )
        for key, aliases in _FIELD_ALIASES.items()
    }


def _normalized_field_name(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


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
    "SERVER_URL",
    "SERVER_USERNAME",
    "automatic_field_mapping",
    "load_connection_password",
    "load_field_mapping",
    "load_settings",
    "save_field_mapping",
    "save_settings",
    "session_directory",
]

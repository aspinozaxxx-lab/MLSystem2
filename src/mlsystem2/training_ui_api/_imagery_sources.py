"""Каталог источников снимков для AOI-инференса."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

from ._config import TrainingUIAPIConfig
from ._datasets import imagery_images_dir


LOGGER = logging.getLogger(__name__)
GOOGLE_HOST_SUFFIXES = (".google.com", ".googleapis.com", ".google.cn")


@dataclass(frozen=True)
class ImagerySource:
    source_id: str
    display_name: str
    kind: Literal["local", "catalog", "xyz", "tms", "wmts"]
    protocol: str
    native_channels: int
    attribution: str
    license_url: str
    imagery_type: str | None = None
    available: bool = True
    settings: dict[str, Any] = field(default_factory=dict)


def list_imagery_sources(config: TrainingUIAPIConfig) -> tuple[ImagerySource, ...]:
    """Вернуть локальные, открытый OAM и разрешённые администратором источники."""

    items = [
        ImagerySource(
            source_id="kanopus",
            display_name="Канопус",
            kind="local",
            protocol="filesystem",
            native_channels=4,
            imagery_type="kanopus",
            attribution="MLSystem2: подготовленные снимки Канопус",
            license_url="",
            available=imagery_images_dir(config.images_root, "kanopus").is_dir(),
        ),
        ImagerySource(
            source_id="ortho",
            display_name="Ортофото",
            kind="local",
            protocol="filesystem",
            native_channels=3,
            imagery_type="ortho",
            attribution="MLSystem2: подготовленные ортофотопланы",
            license_url="",
            available=imagery_images_dir(config.images_root, "ortho").is_dir(),
        ),
        ImagerySource(
            source_id="openaerialmap",
            display_name="OpenAerialMap",
            kind="catalog",
            protocol="tms",
            native_channels=3,
            imagery_type="external_rgb",
            attribution="OpenAerialMap / Open Imagery Network contributors",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            settings={
                "catalog_url": "https://api.openaerialmap.org/meta",
                "request_timeout_seconds": 60,
            },
        ),
    ]
    items.extend(_configured_sources(config.pseudolabel_imagery_providers_path))
    return tuple(items)


def find_imagery_source(config: TrainingUIAPIConfig, source_id: str) -> ImagerySource | None:
    normalized = source_id.strip().casefold()
    return next(
        (item for item in list_imagery_sources(config) if item.source_id.casefold() == normalized),
        None,
    )


def _configured_sources(path: Path | None) -> list[ImagerySource]:
    if path is None or not path.is_file():
        return []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.warning("Не удалось прочитать каталог внешних снимков %s: %s", path, exc)
        return []
    raw_items = payload.get("providers") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        LOGGER.warning("В %s отсутствует массив providers.", path)
        return []
    result: list[ImagerySource] = []
    known_ids = {"kanopus", "ortho", "openaerialmap"}
    for raw in raw_items:
        try:
            source = _configured_source(raw)
        except ValueError as exc:
            LOGGER.warning("Пропущен внешний источник снимков: %s", exc)
            continue
        if source.source_id.casefold() in known_ids:
            LOGGER.warning("Пропущен повторяющийся source_id %s.", source.source_id)
            continue
        known_ids.add(source.source_id.casefold())
        result.append(source)
    return result


def _configured_source(raw: Any) -> ImagerySource:
    if not isinstance(raw, dict):
        raise ValueError("описание провайдера должно быть объектом")
    source_id = str(raw.get("id") or "").strip()
    name = str(raw.get("name") or "").strip()
    protocol = str(raw.get("protocol") or "").strip().lower()
    attribution = str(raw.get("attribution") or "").strip()
    license_url = str(raw.get("license_url") or "").strip()
    if not source_id or not name:
        raise ValueError("id и name обязательны")
    if protocol not in {"xyz", "tms", "wmts"}:
        raise ValueError(f"{source_id}: protocol должен быть xyz, tms или wmts")
    if raw.get("machine_analysis_permitted") is not True:
        raise ValueError(f"{source_id}: не подтверждено machine_analysis_permitted=true")
    if not attribution or not license_url:
        raise ValueError(f"{source_id}: attribution и license_url обязательны")
    endpoint = str(raw.get("url_template") or raw.get("capabilities_url") or "").strip()
    if not endpoint:
        raise ValueError(f"{source_id}: не задан URL сервиса")
    host = (urlparse(endpoint).hostname or "").casefold()
    if host == "google.com" or any(host.endswith(suffix) for suffix in GOOGLE_HOST_SUFFIXES):
        raise ValueError(f"{source_id}: стандартные Google Maps endpoints запрещены для инференса")
    _validate_provider_auth(source_id, raw.get("auth"))
    if protocol in {"xyz", "tms"} and not all(
        placeholder in endpoint for placeholder in ("{z}", "{x}", "{y}")
    ):
        raise ValueError(f"{source_id}: url_template должен содержать {{z}}, {{x}} и {{y}}")
    if protocol == "wmts" and not str(raw.get("layer") or "").strip():
        raise ValueError(f"{source_id}: для WMTS обязателен layer")
    settings = dict(raw)
    settings.pop("machine_analysis_permitted", None)
    return ImagerySource(
        source_id=source_id,
        display_name=name,
        kind=protocol,
        protocol=protocol,
        native_channels=3,
        imagery_type="external_rgb",
        attribution=attribution,
        license_url=license_url,
        settings=settings,
    )


def _validate_provider_auth(source_id: str, raw: Any) -> None:
    auth = raw if isinstance(raw, dict) else {"type": "none"}
    auth_type = str(auth.get("type") or "none").strip().lower()
    allowed = {"none", "query_api_key", "header_api_key", "bearer", "oauth2_client_credentials"}
    if auth_type not in allowed:
        raise ValueError(f"{source_id}: неизвестный тип auth")
    secret_fields = {"api_key", "token", "bearer", "client_id", "client_secret", "password"}
    if any(str(auth.get(field) or "").strip() for field in secret_fields):
        raise ValueError(f"{source_id}: секреты задаются только ссылками на переменные окружения")
    if auth_type in {"query_api_key", "header_api_key", "bearer"}:
        if not str(auth.get("env") or "").strip():
            raise ValueError(f"{source_id}: для auth требуется имя переменной окружения env")
    if auth_type == "oauth2_client_credentials":
        required = ("token_url", "client_id_env", "client_secret_env")
        if any(not str(auth.get(key) or "").strip() for key in required):
            raise ValueError(
                f"{source_id}: OAuth2 требует token_url, client_id_env и client_secret_env"
            )


__all__ = ["ImagerySource", "find_imagery_source", "list_imagery_sources"]

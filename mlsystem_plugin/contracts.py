"""Chistye kontrakty HTTP-obmena plagina."""

from __future__ import annotations

import math
from typing import Any


class PluginContractError(ValueError):
    """Oshibka proverki otveta servera."""


def build_job_payload(
    class_id: str,
    aoi: dict[str, Any],
    aoi_crs: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Sobrat minimalnyi zapros bez rasterov i klientskih putei."""

    if aoi.get("type") not in {"Polygon", "MultiPolygon"}:
        raise PluginContractError("AOI должна быть Polygon или MultiPolygon.")
    if not class_id.strip():
        raise PluginContractError("Класс распознавания не выбран.")
    if not aoi_crs.strip():
        raise PluginContractError("CRS зоны интереса не задана.")
    payload = {"class_id": class_id, "aoi": aoi, "aoi_crs": aoi_crs}
    if source_id is not None:
        if not source_id.strip():
            raise PluginContractError("Источник снимков не выбран.")
        payload["source_id"] = source_id
    return payload


def validate_feature_collection(payload: object) -> dict[str, Any]:
    """Proverit minimalnyi kontrakt rezultata."""

    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise PluginContractError("Сервер вернул не GeoJSON FeatureCollection.")
    features = payload.get("features")
    if not isinstance(features, list):
        raise PluginContractError("В GeoJSON отсутствует массив features.")
    candidate_ids: set[str] = set()
    for index, feature in enumerate(features, start=1):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise PluginContractError(f"Объект {index} не является GeoJSON Feature.")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise PluginContractError(f"У объекта {index} отсутствуют properties.")
        candidate_id = str(properties.get("candidate_id") or "").strip()
        if not candidate_id:
            raise PluginContractError(f"У объекта {index} отсутствует candidate_id.")
        if candidate_id in candidate_ids:
            raise PluginContractError(f"candidate_id повторяется: {candidate_id}.")
        candidate_ids.add(candidate_id)
        confidence = properties.get("confidence")
        if confidence is not None:
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError) as exc:
                raise PluginContractError(
                    f"У объекта {index} некорректная уверенность модели."
                ) from exc
            if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
                raise PluginContractError(
                    f"У объекта {index} уверенность модели должна быть от 0 до 1."
                )
        if not isinstance(feature.get("geometry"), dict):
            raise PluginContractError(f"У объекта {index} отсутствует геометрия.")
    return payload


__all__ = ["PluginContractError", "build_job_payload", "validate_feature_collection"]

"""Аудит геометрии в транзакции завершения подтверждённой Git-публикации."""

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session
from shapely.geometry import shape
from shapely import normalize

from ._models import AnnotationFeatureEventRow, AnnotationPublicationRow, AnnotationVersionRow
from .contracts import AnnotationFeatureHistory, AnnotationFeatureHistoryEntry


def identified_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Стабильные виртуальные ID старых файлов; само чтение не переписывает разметку."""
    if not isinstance(payload.get("features", []), list):
        return dict(payload)
    features = []
    used = {str(item["id"]) for item in payload.get("features", []) if isinstance(item, dict) and item.get("id") is not None}
    occurrences: dict[str, int] = {}
    for item in payload.get("features", []):
        if not isinstance(item, dict) or item.get("id") is not None:
            features.append(item)
            continue
        digest = hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        number = occurrences.get(digest, 0)
        key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"grovika-polygon:{digest}:{number}"))
        while key in used:
            number += 1
            key = str(uuid.uuid5(uuid.NAMESPACE_URL, f"grovika-polygon:{digest}:{number}"))
        occurrences[digest] = number + 1
        used.add(key)
        features.append({**item, "id": key})
    return {**payload, "features": features}


def _features(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in identified_payload(payload).get("features", []) if item.get("id") is not None}


def _same(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first == second:
        return True
    def properties(feature):
        return {key: value for key, value in (feature.get("properties") or {}).items()
                if not key.startswith("_mlsystem2_") or key in {"_mlsystem2_role", "_mlsystem2_class"}}
    return properties(first) == properties(second) and normalize(shape(first["geometry"])).equals_exact(normalize(shape(second["geometry"])), 0)


def record_polygon_history(session: Session, publication: AnnotationPublicationRow, previous: dict[str, dict[str, Any]]) -> None:
    """Вызывается только после push и создания версий, в одной транзакции с completed."""
    if publication.completed:
        return
    at = datetime.now(timezone.utc)
    for item in publication.payload["versions"]:
        version_id = uuid.UUID(item["id"])
        old_payload = previous.get(item["id"], {})
        if not item["deleted"] and old_payload == item["geojson"]:
            continue
        before = _features(old_payload)
        after = {} if item["deleted"] else _features(item["geojson"])
        changed = [key for key in before.keys() | after.keys()
                   if key not in before or key not in after or not _same(before[key], after[key])]
        if not changed:
            continue
        known = set(session.scalars(select(AnnotationFeatureEventRow.feature_id).where(
            AnnotationFeatureEventRow.version_id == version_id, AnnotationFeatureEventRow.feature_id.in_(changed))).all())
        for key in sorted(changed):
            if key in before and key not in known:
                session.add(AnnotationFeatureEventRow(publication_id=publication.id, version_id=version_id,
                    feature_id=key, action="baseline", username=None, recorded_at=at,
                    feature=deepcopy(before[key]), crs=old_payload.get("crs")))
            session.add(AnnotationFeatureEventRow(publication_id=publication.id, version_id=version_id,
                feature_id=key, action="created" if key not in before else "deleted" if key not in after else "updated",
                username=publication.username, recorded_at=at, feature=deepcopy(after.get(key)), crs=item["geojson"].get("crs")))


def polygon_history(session: Session, version: AnnotationVersionRow, feature_id: str) -> AnnotationFeatureHistory:
    rows = session.execute(select(AnnotationFeatureEventRow, AnnotationPublicationRow.commit_sha)
        .join(AnnotationPublicationRow, AnnotationPublicationRow.id == AnnotationFeatureEventRow.publication_id)
        .where(AnnotationFeatureEventRow.version_id == version.id, AnnotationFeatureEventRow.feature_id == feature_id,
               AnnotationPublicationRow.completed.is_(True))
        .order_by(AnnotationFeatureEventRow.recorded_at, AnnotationFeatureEventRow.action, AnnotationFeatureEventRow.id)).all()
    entries = [AnnotationFeatureHistoryEntry(id=str(row.id), action=row.action, username=row.username,
        published_at=(None if row.action == "baseline" else row.recorded_at.replace(tzinfo=timezone.utc) if row.recorded_at.tzinfo is None else row.recorded_at),
        commit=commit, feature=row.feature, crs=row.crs) for row, commit in rows]
    if not entries:
        feature = _features(version.geojson).get(feature_id)
        if feature is not None:
            entries.append(AnnotationFeatureHistoryEntry(id="baseline", action="baseline", feature=feature, crs=version.geojson.get("crs")))
    return AnnotationFeatureHistory(version_id=version.id, feature_id=feature_id, entries=entries)

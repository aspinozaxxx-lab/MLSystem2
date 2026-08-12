"""Stored file routes."""

from __future__ import annotations

import json
import re
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._contracts.common import StoredFileKind
from mlsystem2.training_ui_api._models import StoredFileRow
from mlsystem2.training_ui_api._service import stored_file, stored_file_download_name

from .common import RouteContext


def register_file_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/files/{file_id}/download")
    def download_file(
        file_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        row: StoredFileRow = stored_file(db, file_id)
        path = row.path
        if not Path(path).is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден")
        return FileResponse(path, filename=stored_file_download_name(row))

    @app.get("/api/v1/files/{file_id}/download-by-type")
    def download_file_by_type(
        file_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> Response:
        row: StoredFileRow = stored_file(db, file_id)
        path = Path(row.path)
        multiclass_geojson_kinds = {
            StoredFileKind.PSEUDO_MARKUP_GEOJSON.value,
            StoredFileKind.PSEUDOLABEL_GEOJSON.value,
        }
        if row.kind not in multiclass_geojson_kinds or not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="GeoJSON не найден")
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="GeoJSON повреждён",
            ) from exc
        try:
            archive_content = _multiclass_geojson_archive(payload)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return Response(
            content=archive_content,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="pseudolabel_by_type.zip"'},
        )


def _multiclass_geojson_archive(payload: object) -> bytes:
    features = payload.get("features") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    schema = metadata.get("class_schema") if isinstance(metadata, dict) else None
    if not isinstance(features, list) or not isinstance(schema, list) or not schema:
        raise ValueError("Файл не содержит мультиклассовую схему")
    archive_buffer = BytesIO()
    written = 0
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for item in schema:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug") or "")
            if re.fullmatch(r"[a-z][a-z0-9_]*", slug) is None:
                raise ValueError(f"Некорректный slug типа объектов: {slug!r}")
            selected = [
                feature
                for feature in features
                if isinstance(feature, dict)
                and isinstance(feature.get("properties"), dict)
                and str(feature["properties"].get("object_type_slug") or "") == slug
            ]
            type_payload = {
                "type": "FeatureCollection",
                "features": selected,
                "metadata": {**metadata, "class_schema": [item]},
            }
            archive.writestr(
                f"{slug}.geojson",
                json.dumps(type_payload, ensure_ascii=False, indent=2) + "\n",
            )
            written += 1
    if written != len(schema):
        raise ValueError("Схема типов объектов повреждена")
    return archive_buffer.getvalue()


__all__ = ["register_file_routes"]

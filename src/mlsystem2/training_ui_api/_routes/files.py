"""Stored file routes."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

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


__all__ = ["register_file_routes"]

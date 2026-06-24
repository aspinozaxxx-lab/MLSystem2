"""Model export routes."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from mlsystem2.training_ui_api._model_export import build_triton_model_export_zip
from mlsystem2.training_ui_api._service import export_training_result_triton_zip
from mlsystem2.training_ui_api.contracts import TrainingUIAPIError

from .common import RouteContext


def register_export_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.post("/api/v1/model-export/triton-zip")
    async def post_model_export_triton_zip(
        _: str = Depends(ctx.authenticated),
        model_name: str = Form(default=""),
        sample_size: int | None = Form(default=None),
        checkpoint: UploadFile | None = File(default=None),
    ) -> FileResponse:
        if checkpoint is None:
            raise TrainingUIAPIError("Нужен checkpoint MLSystem2 в формате .pt.")
        archive = build_triton_model_export_zip(
            model_name=model_name,
            checkpoint_filename=checkpoint.filename or "",
            checkpoint_bytes=await checkpoint.read(),
            sample_size=sample_size,
        )
        return FileResponse(
            archive.zip_path,
            filename=archive.filename,
            media_type="application/zip",
            background=BackgroundTask(archive.cleanup),
        )

    @app.post("/api/v1/results/training/{result_id}/triton-zip")
    async def post_training_result_triton_zip(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
        model_name: str = Form(default=""),
        sample_size: int | None = Form(default=None),
    ) -> FileResponse:
        archive = export_training_result_triton_zip(
            db,
            result_id=result_id,
            model_name=model_name,
            sample_size=sample_size,
            config=ctx.config,
        )
        return FileResponse(
            archive.zip_path,
            filename=archive.filename,
            media_type="application/zip",
            background=BackgroundTask(archive.cleanup),
        )


__all__ = ["register_export_routes"]

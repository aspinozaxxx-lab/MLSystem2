"""HTTP-маршруты экспорта моделей и тестовой разметки."""

from __future__ import annotations

import uuid
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from mlsystem2.training_ui_api._markup_export import (
    MarkupExportUnavailable,
    build_markup_export,
    build_scene_list_export,
    load_markup_export,
)
from mlsystem2.training_ui_api._model_export import build_triton_model_export_zip
from mlsystem2.training_ui_api._service import export_training_result_triton_zip, export_training_results_triton_zip
from mlsystem2.training_ui_api.contracts import (
    ImageryType,
    MarkupExportInfo,
    MarkupExportRequest,
    TrainingResultBatchExportRequest,
    TrainingUIAPIError,
)

from .common import RouteContext


def register_export_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.post(
        "/api/v1/scene-list-export",
        response_class=Response,
        responses={
            200: {
                "description": (
                    "TXT с относительными путями сцен либо ZIP с TXT и GeoJSON "
                    "футпринтов выбранных снимков."
                ),
                "content": {
                    "text/plain": {"schema": {"type": "string", "format": "binary"}},
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    },
                },
            }
        },
    )
    async def post_scene_list_export(
        _: str = Depends(ctx.authenticated),
        imagery_type: ImageryType = Form(...),
        include_footprints: bool = Form(default=False),
        geojson: UploadFile = File(...),
    ) -> Response:
        artifact = await run_in_threadpool(
            build_scene_list_export,
            imagery_type=imagery_type,
            geojson_filename=geojson.filename or "",
            geojson_bytes=await geojson.read(),
            config=ctx.config,
        )
        if include_footprints:
            return Response(
                content=artifact.archive_content,
                media_type="application/zip",
                headers={
                    "Content-Disposition": (
                        "attachment; filename=\"scene-list.zip\"; "
                        f"filename*=UTF-8''{quote(artifact.archive_filename, safe='')}"
                    )
                },
            )
        return Response(
            content=artifact.content,
            media_type="text/plain",
            headers={
                "Content-Disposition": (
                    "attachment; filename=\"scene-list.txt\"; "
                    f"filename*=UTF-8''{quote(artifact.filename, safe='')}"
                )
            },
        )

    @app.post("/api/v1/markup-export", response_model=MarkupExportInfo)
    def post_markup_export(
        request: MarkupExportRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> MarkupExportInfo:
        return build_markup_export(request, ctx.config, db)

    @app.get(
        "/api/v1/markup-export/{export_id}/tiles/{tile_index}/preview",
        response_class=FileResponse,
        responses={
            200: {
                "description": "PNG-превью тайла с наложенной маской.",
                "content": {
                    "image/png": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    def get_markup_export_preview(
        export_id: uuid.UUID,
        tile_index: int,
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _markup_artifact_or_404(export_id, ctx)
        preview_path = artifact.preview_paths.get(tile_index)
        if preview_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Превью тестового тайла не найдено.",
            )
        return FileResponse(preview_path, media_type="image/png")

    @app.get(
        "/api/v1/markup-export/{export_id}/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP-архив набора тестовой разметки.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    def get_markup_export_download(
        export_id: uuid.UUID,
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _markup_artifact_or_404(export_id, ctx)
        return FileResponse(
            artifact.archive_path,
            filename=artifact.archive_filename,
            media_type="application/zip",
        )

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

    @app.post("/api/v1/results/training/triton-zip")
    def post_training_results_triton_zip(
        request: TrainingResultBatchExportRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        archive = export_training_results_triton_zip(
            db,
            request=request,
            config=ctx.config,
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


def _markup_artifact_or_404(export_id: uuid.UUID, ctx: RouteContext):
    try:
        return load_markup_export(export_id, ctx.config)
    except MarkupExportUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Экспорт разметки не найден или срок его хранения истек.",
        ) from exc


__all__ = ["register_export_routes"]

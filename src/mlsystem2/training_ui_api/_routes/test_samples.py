"""HTTP-маршруты постоянных тестовых выборок."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from mlsystem2.training_ui_api._test_samples import (
    TestSampleUnavailable,
    build_test_sample_download,
    create_test_sample,
    delete_test_sample,
    evaluate_test_sample_by_id,
    test_sample_catalog,
    test_sample_detail,
    test_sample_preview_path,
    update_test_sample,
    update_test_sample_tile,
)
from mlsystem2.training_ui_api.contracts import (
    TestSampleCatalogResponse,
    TestSampleCreate,
    TestSampleDetail,
    TestSampleTileUpdate,
    TestSampleUpdate,
)

from .common import RouteContext


def register_test_sample_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/test-samples", response_model=TestSampleCatalogResponse)
    def get_test_samples(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleCatalogResponse:
        return test_sample_catalog(db)

    @app.post("/api/v1/test-samples", response_model=TestSampleDetail)
    def post_test_sample(
        request: TestSampleCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        return create_test_sample(db, request, ctx.config)

    @app.get("/api/v1/test-samples/{sample_id}", response_model=TestSampleDetail)
    def get_test_sample(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        return _sample_or_404(lambda: test_sample_detail(db, sample_id))

    @app.patch("/api/v1/test-samples/{sample_id}", response_model=TestSampleDetail)
    def patch_test_sample(
        sample_id: uuid.UUID,
        request: TestSampleUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        return _sample_or_404(lambda: update_test_sample(db, sample_id, request))

    @app.delete("/api/v1/test-samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_test_sample(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> Response:
        _sample_or_404(lambda: delete_test_sample(db, sample_id, ctx.config))
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.patch(
        "/api/v1/test-samples/{sample_id}/tiles/{tile_index}",
        response_model=TestSampleDetail,
    )
    def patch_test_sample_tile(
        sample_id: uuid.UUID,
        tile_index: int,
        request: TestSampleTileUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        return _sample_or_404(
            lambda: update_test_sample_tile(db, sample_id, tile_index, request)
        )

    @app.post(
        "/api/v1/test-samples/{sample_id}/evaluate",
        response_model=TestSampleDetail,
    )
    def post_test_sample_evaluation(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        return _sample_or_404(
            lambda: evaluate_test_sample_by_id(db, sample_id, ctx.config)
        )

    @app.get(
        "/api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Постоянное PNG-превью тестового тайла.",
                "content": {
                    "image/png": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    def get_test_sample_preview(
        sample_id: uuid.UUID,
        tile_index: int,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        path = _sample_or_404(
            lambda: test_sample_preview_path(db, sample_id, tile_index, ctx.config)
        )
        return FileResponse(path, media_type="image/png")

    @app.get(
        "/api/v1/test-samples/{sample_id}/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP включённых тайлов постоянной тестовой выборки.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    def download_test_sample(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _sample_or_404(
            lambda: build_test_sample_download(db, sample_id, ctx.config)
        )
        return FileResponse(
            artifact.path,
            filename=artifact.filename,
            media_type="application/zip",
            background=BackgroundTask(artifact.cleanup),
        )


def _sample_or_404(operation):
    try:
        return operation()
    except TestSampleUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Тестовая выборка или её файл не найдены.",
        ) from exc


__all__ = ["register_test_sample_routes"]

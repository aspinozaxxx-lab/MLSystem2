"""HTTP-маршруты постоянных тестовых выборок."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from mlsystem2.training_ui_api._test_samples import (
    TestSampleBatchUnavailable,
    TestSampleUnavailable,
    build_primary_test_samples_download,
    build_test_sample_download,
    create_test_sample_batch,
    create_test_sample,
    delete_test_sample,
    evaluate_test_sample_by_id,
    latest_test_sample_batch,
    optimize_test_sample,
    test_sample_batch_detail,
    test_sample_catalog,
    test_sample_detail,
    test_sample_preview_path,
    update_test_sample,
    update_test_sample_primary,
    update_test_sample_tile,
)
from mlsystem2.training_ui_api.contracts import (
    TestSampleBatchCreate,
    TestSampleBatchInfo,
    TestSampleCatalogResponse,
    TestSampleCreate,
    TestSampleDetail,
    TestSampleOptimizeRequest,
    TestSamplePrimaryUpdate,
    TestSampleTileUpdate,
    TestSampleUpdate,
)

from .common import RouteContext


def register_test_sample_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.post("/api/v1/test-sample-batches", response_model=TestSampleBatchInfo)
    def post_test_sample_batch(
        request: TestSampleBatchCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchInfo:
        detail = create_test_sample_batch(db, request, ctx.config)
        db.commit()
        return detail

    @app.get("/api/v1/test-sample-batches/latest", response_model=TestSampleBatchInfo)
    def get_latest_test_sample_batch(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchInfo:
        return _batch_or_404(lambda: latest_test_sample_batch(db))

    @app.get("/api/v1/test-sample-batches/{batch_id}", response_model=TestSampleBatchInfo)
    def get_test_sample_batch(
        batch_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchInfo:
        return _batch_or_404(lambda: test_sample_batch_detail(db, batch_id))

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
        detail = create_test_sample(db, request, ctx.config)
        db.commit()
        return detail

    @app.get(
        "/api/v1/test-samples/primary/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP всех основных тестовых выборок.",
                "content": {
                    "application/zip": {
                        "schema": {"type": "string", "format": "binary"}
                    }
                },
            }
        },
    )
    def download_primary_test_samples(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = build_primary_test_samples_download(db, ctx.config)
        return FileResponse(
            artifact.path,
            filename=artifact.filename,
            media_type="application/zip",
            background=BackgroundTask(artifact.cleanup),
        )

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
        detail = _sample_or_404(lambda: update_test_sample(db, sample_id, request))
        db.commit()
        return detail

    @app.put(
        "/api/v1/test-samples/{sample_id}/primary",
        response_model=TestSampleDetail,
    )
    def put_test_sample_primary(
        sample_id: uuid.UUID,
        request: TestSamplePrimaryUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        detail = _sample_or_404(
            lambda: update_test_sample_primary(db, sample_id, request)
        )
        db.commit()
        return detail

    @app.delete("/api/v1/test-samples/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
    def remove_test_sample(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> Response:
        _sample_or_404(lambda: delete_test_sample(db, sample_id, ctx.config))
        db.commit()
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
        detail = _sample_or_404(
            lambda: update_test_sample_tile(db, sample_id, tile_index, request)
        )
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/{sample_id}/evaluate",
        response_model=TestSampleDetail,
    )
    def post_test_sample_evaluation(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        detail = _sample_or_404(
            lambda: evaluate_test_sample_by_id(db, sample_id, ctx.config)
        )
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/{sample_id}/optimize",
        response_model=TestSampleDetail,
    )
    def post_test_sample_optimization(
        sample_id: uuid.UUID,
        request: TestSampleOptimizeRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        detail = _sample_or_404(
            lambda: optimize_test_sample(db, sample_id, request, ctx.config)
        )
        db.commit()
        return detail

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


def _batch_or_404(operation):
    try:
        return operation()
    except TestSampleBatchUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Групповой запуск тестовых выборок не найден.",
        ) from exc


__all__ = ["register_test_sample_routes"]

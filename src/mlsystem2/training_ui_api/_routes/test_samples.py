"""HTTP-маршруты постоянных тестовых разметок."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from mlsystem2.training_ui_api._test_sample_annotations import (
    TestSampleRevisionConflict,
    merge_test_sample_annotations,
)

from mlsystem2.training_ui_api._test_samples import (
    TestSampleBatchUnavailable,
    TestSampleUnavailable,
    build_test_sample_download,
    build_test_samples_download,
    create_test_sample_batch,
    cancel_test_sample_batch,
    delete_test_sample_batch,
    move_test_sample_batch,
    save_test_sample_creation_settings,
    test_sample_batch_queue,
    create_test_sample,
    delete_test_sample,
    evaluate_test_sample_by_id,
    evaluate_test_sample_preview,
    latest_test_sample_batch,
    optimize_test_sample,
    optimize_test_sample_preview,
    reconcile_test_sample_evaluations,
    test_sample_batch_options,
    test_sample_batch_detail,
    test_sample_catalog,
    test_sample_cards,
    test_sample_class_index,
    test_sample_detail,
    test_sample_preview_path,
    test_sample_thumbnail_path,
    update_test_sample,
    update_test_sample_primary,
    update_test_sample_tile,
)
from mlsystem2.training_ui_api._service import (
    ensure_test_sample_batch_dataset_pseudo_markup_job,
    ensure_test_sample_pseudo_markup_job,
)
from mlsystem2.training_ui_api.contracts import (
    JobDetail,
    TestSampleAnnotationsMerge,
    TestSampleBatchCreate,
    TestSampleBatchMove,
    TestSampleBatchInfo,
    TestSampleBatchOptionsResponse,
    TestSampleCreationSettings,
    TestSampleBulkDownloadRequest,
    TestSampleCatalogResponse,
    TestSampleCard,
    TestSampleClassIndexResponse,
    TestSampleCreate,
    TestSampleDetail,
    TestSampleDownloadRequest,
    TestSampleDraftPreview,
    TestSampleEvaluationPreviewRequest,
    TestSampleOptimizeRequest,
    TestSamplePrimaryUpdate,
    TestSampleTileUpdate,
    TestSampleUpdate,
)

from .common import RouteContext


def register_test_sample_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/test-sample-batches", response_model=list[TestSampleBatchInfo])
    def get_creation_queue(
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> list[TestSampleBatchInfo]:
        return test_sample_batch_queue(db)

    @app.put("/api/v1/test-sample-batches/options/{dataset_key}/settings", response_model=TestSampleCreationSettings)
    def put_creation_settings(
        dataset_key: str, request: TestSampleCreationSettings,
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> TestSampleCreationSettings:
        saved = save_test_sample_creation_settings(db, dataset_key, request, ctx.config)
        db.commit()
        return saved

    @app.post("/api/v1/test-sample-batches/{batch_id}/move", response_model=TestSampleBatchInfo)
    def post_creation_move(
        batch_id: uuid.UUID, request: TestSampleBatchMove,
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchInfo:
        detail = _batch_or_404(lambda: move_test_sample_batch(db, batch_id, request.direction))
        db.commit()
        return detail

    @app.post("/api/v1/test-sample-batches/{batch_id}/cancel", response_model=TestSampleBatchInfo)
    def post_creation_cancel(
        batch_id: uuid.UUID, db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchInfo:
        detail = _batch_or_404(lambda: cancel_test_sample_batch(db, batch_id))
        db.commit()
        return detail

    @app.delete("/api/v1/test-sample-batches/{batch_id}", status_code=204)
    def delete_creation_job(
        batch_id: uuid.UUID, db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> Response:
        _batch_or_404(lambda: delete_test_sample_batch(db, batch_id))
        db.commit()
        return Response(status_code=204)

    @app.get(
        "/api/v1/test-sample-batches/options",
        response_model=TestSampleBatchOptionsResponse,
    )
    def get_test_sample_batch_options(
        class_key: str | None = None,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleBatchOptionsResponse:
        return test_sample_batch_options(db, ctx.config, class_key=class_key)

    @app.post(
        "/api/v1/test-sample-batches/options/{dataset_key}/pseudo-markup",
        response_model=JobDetail,
    )
    def post_test_sample_batch_dataset_pseudo_markup(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        detail = ensure_test_sample_batch_dataset_pseudo_markup_job(
            db,
            dataset_key,
            ctx.config,
        )
        db.commit()
        return detail

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
        return test_sample_catalog(db, ctx.config)

    @app.get("/api/v1/test-samples/classes", response_model=TestSampleClassIndexResponse)
    def get_test_sample_classes(
        include_empty: bool = False,
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> TestSampleClassIndexResponse:
        return test_sample_class_index(db, include_empty=include_empty)

    @app.get("/api/v1/test-samples/cards", response_model=list[TestSampleCard])
    def get_test_sample_cards(
        class_key: str | None = None,
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> list[TestSampleCard]:
        return test_sample_cards(db, class_key=class_key)

    @app.post("/api/v1/test-samples/classes/{class_key}/reconcile", status_code=204)
    def post_test_sample_class_reconcile(
        class_key: str,
        db: Session = Depends(ctx.get_db), _: str = Depends(ctx.authenticated),
    ) -> Response:
        reconcile_test_sample_evaluations(db, ctx.config, class_keys={class_key})
        db.commit()
        return Response(status_code=204)

    @app.post("/api/v1/test-samples/reconcile", response_model=TestSampleCatalogResponse)
    def post_test_samples_reconcile(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleCatalogResponse:
        reconcile_test_sample_evaluations(db, ctx.config)
        db.commit()
        return test_sample_catalog(db, ctx.config)

    @app.post("/api/v1/test-samples", response_model=TestSampleDetail)
    def post_test_sample(
        request: TestSampleCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        detail = create_test_sample(db, request, ctx.config)
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP выбранных сохранённых тестовых разметок.",
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    def download_test_samples(
        request: TestSampleBulkDownloadRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _sample_or_404(
            lambda: build_test_samples_download(
                db,
                request.sample_ids,
                ctx.config,
                include_previews=request.include_previews,
            )
        )
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
        return _sample_or_404(lambda: test_sample_detail(db, sample_id, ctx.config))

    @app.patch("/api/v1/test-samples/{sample_id}", response_model=TestSampleDetail)
    def patch_test_sample(
        sample_id: uuid.UUID,
        request: TestSampleUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        detail = _sample_or_404(lambda: update_test_sample(db, sample_id, request, ctx.config))
        db.commit()
        return detail

    @app.post("/api/v1/test-samples/{sample_id}/merge-annotations", response_model=TestSampleDetail)
    def post_test_sample_annotations_merge(
        sample_id: uuid.UUID,
        request: TestSampleAnnotationsMerge,
        db: Session = Depends(ctx.get_db),
        actor: str = Depends(ctx.authenticated),
    ) -> TestSampleDetail:
        try:
            with merge_test_sample_annotations(db, sample_id, request, ctx.config, actor=actor) as detail:
                db.commit()
                return detail
        except TestSampleRevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except TestSampleUnavailable as exc:
            raise HTTPException(status_code=404, detail="Тестовая разметка не найдена.") from exc

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
            lambda: update_test_sample_primary(db, sample_id, request, ctx.config)
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
            lambda: update_test_sample_tile(
                db,
                sample_id,
                tile_index,
                request,
                ctx.config,
            )
        )
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/{sample_id}/evaluate-preview",
        response_model=TestSampleDraftPreview,
    )
    def post_test_sample_evaluation_preview(
        sample_id: uuid.UUID,
        request: TestSampleEvaluationPreviewRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDraftPreview:
        return _sample_or_404(
            lambda: evaluate_test_sample_preview(db, sample_id, request, ctx.config)
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
        detail = _sample_or_404(lambda: evaluate_test_sample_by_id(db, sample_id, ctx.config))
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/{sample_id}/pseudo-markup",
        response_model=JobDetail,
    )
    def post_test_sample_pseudo_markup(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        job = _sample_or_404(
            lambda: ensure_test_sample_pseudo_markup_job(db, sample_id, ctx.config)
        )
        db.commit()
        return job

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
        detail = _sample_or_404(lambda: optimize_test_sample(db, sample_id, request, ctx.config))
        db.commit()
        return detail

    @app.post(
        "/api/v1/test-samples/{sample_id}/optimize-preview",
        response_model=TestSampleDraftPreview,
    )
    def post_test_sample_optimization_preview(
        sample_id: uuid.UUID,
        request: TestSampleOptimizeRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TestSampleDraftPreview:
        return _sample_or_404(
            lambda: optimize_test_sample_preview(db, sample_id, request, ctx.config)
        )

    @app.get(
        "/api/v1/test-samples/{sample_id}/tiles/{tile_index}/preview",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Постоянное PNG-превью тестового тайла.",
                "content": {"image/png": {"schema": {"type": "string", "format": "binary"}}},
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
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get(
        "/api/v1/test-samples/{sample_id}/tiles/{tile_index}/thumbnail",
        response_class=FileResponse,
        responses={
            200: {
                "description": "Компактная JPEG-миниатюра тестового тайла.",
                "content": {"image/jpeg": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    def get_test_sample_thumbnail(
        sample_id: uuid.UUID,
        tile_index: int,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        path = _sample_or_404(
            lambda: test_sample_thumbnail_path(db, sample_id, tile_index, ctx.config)
        )
        return FileResponse(
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @app.get(
        "/api/v1/test-samples/{sample_id}/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP включённых тайлов постоянной тестовой разметки.",
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    def download_test_sample(
        sample_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _sample_or_404(lambda: build_test_sample_download(db, sample_id, ctx.config))
        return FileResponse(
            artifact.path,
            filename=artifact.filename,
            media_type="application/zip",
            background=BackgroundTask(artifact.cleanup),
        )

    @app.post(
        "/api/v1/test-samples/{sample_id}/download",
        response_class=FileResponse,
        responses={
            200: {
                "description": "ZIP выбранных тайлов незаписанного черновика тестовой разметки.",
                "content": {"application/zip": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    def download_test_sample_draft(
        sample_id: uuid.UUID,
        request: TestSampleDownloadRequest,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> FileResponse:
        artifact = _sample_or_404(
            lambda: build_test_sample_download(
                db,
                sample_id,
                ctx.config,
                enabled_tile_indices=request.enabled_tile_indices,
                include_previews=request.include_previews,
            )
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
            detail="Тестовая разметка или её файл не найдены.",
        ) from exc


def _batch_or_404(operation):
    try:
        return operation()
    except TestSampleBatchUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Групповой запуск тестовых разметок не найден.",
        ) from exc


__all__ = ["register_test_sample_routes"]

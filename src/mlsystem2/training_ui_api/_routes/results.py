"""Result and pseudo-markup routes."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    clear_primary_training_result,
    create_pseudo_markup_job,
    dataset_results,
    delete_pseudo_markup_result,
    recalculate_dataset_test_f1,
    result_changes,
    result_classes,
    set_primary_training_result,
)
from mlsystem2.training_ui_api.contracts import (
    DatasetResultsResponse,
    JobDetail,
    PseudoMarkupResultInfo,
    PseudoMarkupViewInfo,
    ResultClassListResponse,
    ResultChangesResponse,
    TrainingResultInfo,
)

from .common import RouteContext
from mlsystem2.training_ui_api._pseudo_viewer import pseudo_markup_footprint, pseudo_markup_raster, pseudo_markup_view
from mlsystem2.training_ui_api._raster_http import raster_response, raster_revision


def register_result_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/results/pseudo-markup/{result_id}/view", response_model=PseudoMarkupViewInfo)
    def get_pseudo_markup_view(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> PseudoMarkupViewInfo:
        return pseudo_markup_view(db, ctx.config, result_id)

    @app.get("/api/v1/results/pseudo-markup/{result_id}/raster/{scene_id}")
    def get_pseudo_markup_raster(
        result_id: uuid.UUID, scene_id: str,
        v: str | None = None,
        range_header: str | None = Header(default=None, alias="Range"),
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> StreamingResponse:
        return raster_response(pseudo_markup_raster(db, ctx.config, result_id, scene_id), range_header, v)

    @app.get("/api/v1/results/pseudo-markup/{result_id}/footprint/{scene_id}")
    def get_pseudo_markup_footprint(
        result_id: uuid.UUID, scene_id: str,
        v: str | None = None,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JSONResponse:
        path = pseudo_markup_raster(db, ctx.config, result_id, scene_id)
        revision = raster_revision(path)
        if v is not None and v != revision:
            raise HTTPException(412, "Снимок изменился. Откройте просмотр заново.")
        return JSONResponse(pseudo_markup_footprint(path), headers={
            "ETag": f'"{revision}"', "Cache-Control": "private, no-store",
        })

    @app.get("/api/v1/results/classes", response_model=ResultClassListResponse)
    def get_result_classes(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> ResultClassListResponse:
        return result_classes(db, ctx.config)

    @app.get("/api/v1/results/changes", response_model=ResultChangesResponse)
    def get_result_changes(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> ResultChangesResponse:
        return result_changes(db, ctx.config)

    @app.get("/api/v1/results/datasets/{dataset_key}", response_model=DatasetResultsResponse)
    def get_dataset_results(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetResultsResponse:
        return dataset_results(db, dataset_key, ctx.config)

    @app.post("/api/v1/results/datasets/{dataset_key}/pseudo-markup", response_model=JobDetail)
    async def post_pseudo_markup(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
        source_dataset_key: str | None = Form(default=None, alias="dataset_key"),
        image_folder_key: str | None = Form(default=None),
        training_result_id: str | None = Form(default=None),
        scenes_txt: UploadFile | None = File(default=None),
    ) -> JobDetail:
        parsed_training_result_id = uuid.UUID(training_result_id) if training_result_id else None
        scenes_name = scenes_txt.filename if scenes_txt is not None and scenes_txt.filename else None
        scenes_bytes = await scenes_txt.read() if scenes_name is not None else None
        return create_pseudo_markup_job(
            db,
            class_key=dataset_key,
            dataset_key=source_dataset_key,
            image_folder_key=image_folder_key,
            training_result_id=parsed_training_result_id,
            scenes_name=scenes_name,
            scenes_content_type=scenes_txt.content_type if scenes_name is not None else None,
            scenes_bytes=scenes_bytes,
            config=ctx.config,
        )

    @app.post(
        "/api/v1/results/datasets/{dataset_key}/test-f1",
        response_model=DatasetResultsResponse,
    )
    def post_test_f1(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetResultsResponse:
        return recalculate_dataset_test_f1(db, dataset_key, ctx.config)

    @app.post(
        "/api/v1/results/training/{result_id}/primary",
        response_model=TrainingResultInfo,
    )
    def post_primary_training_result(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingResultInfo:
        return set_primary_training_result(db, result_id, ctx.config)

    @app.delete(
        "/api/v1/results/training/{result_id}/primary",
        response_model=TrainingResultInfo,
    )
    def delete_primary_training_result(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingResultInfo:
        return clear_primary_training_result(db, result_id, ctx.config)

    @app.delete("/api/v1/results/pseudo-markup/{result_id}", response_model=PseudoMarkupResultInfo)
    def delete_pseudo_markup(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> PseudoMarkupResultInfo:
        return delete_pseudo_markup_result(db, result_id, ctx.config)


__all__ = ["register_result_routes"]

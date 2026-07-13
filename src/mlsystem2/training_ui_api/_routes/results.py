"""Result and pseudo-markup routes."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI, File, Form, UploadFile
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    class_results,
    create_pseudo_markup_job,
    delete_pseudo_markup_result,
    recalculate_class_test_f1,
    result_changes,
    result_classes,
)
from mlsystem2.training_ui_api.contracts import (
    ClassResultsResponse,
    JobDetail,
    PseudoMarkupResultInfo,
    ResultClassListResponse,
    ResultChangesResponse,
)

from .common import RouteContext


def register_result_routes(app: FastAPI, ctx: RouteContext) -> None:
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
        return result_changes(db)

    @app.get("/api/v1/results/classes/{class_key}", response_model=ClassResultsResponse)
    def get_class_results(
        class_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> ClassResultsResponse:
        return class_results(db, class_key, ctx.config)

    @app.post("/api/v1/results/classes/{class_key}/pseudo-markup", response_model=JobDetail)
    async def post_pseudo_markup(
        class_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
        dataset_key: str | None = Form(default=None),
        image_folder_key: str | None = Form(default=None),
        training_result_id: str | None = Form(default=None),
        scenes_txt: UploadFile | None = File(default=None),
    ) -> JobDetail:
        parsed_training_result_id = uuid.UUID(training_result_id) if training_result_id else None
        scenes_name = scenes_txt.filename if scenes_txt is not None and scenes_txt.filename else None
        scenes_bytes = await scenes_txt.read() if scenes_name is not None else None
        return create_pseudo_markup_job(
            db,
            class_key=class_key,
            dataset_key=dataset_key,
            image_folder_key=image_folder_key,
            training_result_id=parsed_training_result_id,
            scenes_name=scenes_name,
            scenes_content_type=scenes_txt.content_type if scenes_name is not None else None,
            scenes_bytes=scenes_bytes,
            config=ctx.config,
        )

    @app.post(
        "/api/v1/results/classes/{class_key}/test-f1",
        response_model=ClassResultsResponse,
    )
    def post_test_f1(
        class_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> ClassResultsResponse:
        return recalculate_class_test_f1(db, class_key, ctx.config)

    @app.delete("/api/v1/results/pseudo-markup/{result_id}", response_model=PseudoMarkupResultInfo)
    def delete_pseudo_markup(
        result_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> PseudoMarkupResultInfo:
        return delete_pseudo_markup_result(db, result_id, ctx.config)


__all__ = ["register_result_routes"]

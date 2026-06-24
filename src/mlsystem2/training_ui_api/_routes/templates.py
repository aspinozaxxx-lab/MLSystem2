"""Training and inference template routes."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    apply_inference_template_field_to_all,
    apply_training_template_field_to_all,
    create_inference_template,
    create_training_template,
    delete_inference_template,
    delete_training_template,
    inference_template,
    inference_templates,
    training_template,
    training_templates,
    update_inference_template,
    update_inference_template_by_id,
    update_training_template,
    update_training_template_by_id,
)
from mlsystem2.training_ui_api.contracts import (
    InferenceTemplate,
    InferenceTemplateApplyField,
    InferenceTemplateCreate,
    InferenceTemplateListResponse,
    InferenceTemplateUpdate,
    TrainingTemplate,
    TrainingTemplateApplyField,
    TrainingTemplateCreate,
    TrainingTemplateListResponse,
    TrainingTemplateUpdate,
)

from .common import RouteContext


def register_template_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/training-templates", response_model=TrainingTemplateListResponse)
    def get_training_templates(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplateListResponse:
        return training_templates(db)

    @app.post("/api/v1/training-templates", response_model=TrainingTemplate)
    def post_training_template(
        request: TrainingTemplateCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplate:
        return create_training_template(db, request, ctx.config)

    @app.put("/api/v1/training-templates/by-id/{template_id}", response_model=TrainingTemplate)
    def put_training_template_by_id(
        template_id: uuid.UUID,
        request: TrainingTemplateUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplate:
        return update_training_template_by_id(db, template_id, request)

    @app.delete("/api/v1/training-templates/by-id/{template_id}", response_model=TrainingTemplate)
    def delete_training_template_by_id(
        template_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplate:
        return delete_training_template(db, template_id)

    @app.put(
        "/api/v1/training-templates/by-id/{template_id}/apply-field-to-all",
        response_model=TrainingTemplateListResponse,
    )
    def put_training_template_field_to_all(
        template_id: uuid.UUID,
        request: TrainingTemplateApplyField,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplateListResponse:
        return apply_training_template_field_to_all(db, template_id, request)

    @app.get("/api/v1/training-templates/{architecture}", response_model=TrainingTemplate)
    def get_training_template(
        architecture: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplate:
        return training_template(db, architecture)

    @app.put("/api/v1/training-templates/{architecture}", response_model=TrainingTemplate)
    def put_training_template(
        architecture: str,
        request: TrainingTemplateUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> TrainingTemplate:
        return update_training_template(db, architecture, request)

    @app.get("/api/v1/inference-templates", response_model=InferenceTemplateListResponse)
    def get_inference_templates(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplateListResponse:
        return inference_templates(db)

    @app.post("/api/v1/inference-templates", response_model=InferenceTemplate)
    def post_inference_template(
        request: InferenceTemplateCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplate:
        return create_inference_template(db, request, ctx.config)

    @app.put("/api/v1/inference-templates/by-id/{template_id}", response_model=InferenceTemplate)
    def put_inference_template_by_id(
        template_id: uuid.UUID,
        request: InferenceTemplateUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplate:
        return update_inference_template_by_id(db, template_id, request)

    @app.delete("/api/v1/inference-templates/by-id/{template_id}", response_model=InferenceTemplate)
    def delete_inference_template_by_id(
        template_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplate:
        return delete_inference_template(db, template_id)

    @app.put(
        "/api/v1/inference-templates/by-id/{template_id}/apply-field-to-all",
        response_model=InferenceTemplateListResponse,
    )
    def put_inference_template_field_to_all(
        template_id: uuid.UUID,
        request: InferenceTemplateApplyField,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplateListResponse:
        return apply_inference_template_field_to_all(db, template_id, request)

    @app.get("/api/v1/inference-templates/{architecture}", response_model=InferenceTemplate)
    def get_inference_template(
        architecture: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplate:
        return inference_template(db, architecture)

    @app.put("/api/v1/inference-templates/{architecture}", response_model=InferenceTemplate)
    def put_inference_template(
        architecture: str,
        request: InferenceTemplateUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> InferenceTemplate:
        return update_inference_template(db, architecture, request)


__all__ = ["register_template_routes"]

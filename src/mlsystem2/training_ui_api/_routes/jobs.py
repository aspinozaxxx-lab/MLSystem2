"""Job and queue routes."""

from __future__ import annotations

import uuid

from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    create_training_job,
    delete_job,
    job_detail,
    job_log,
    move_job,
    queue_count,
    queues,
    set_queue_enabled,
)
from mlsystem2.training_ui_api.contracts import (
    JobDetail,
    JobLogInfo,
    JobType,
    QueueEnabledUpdate,
    QueueCountInfo,
    QueueSnapshot,
    TrainingJobCreate,
)

from .common import RouteContext


def register_job_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.post("/api/v1/training-jobs", response_model=JobDetail)
    def post_training_job(
        request: TrainingJobCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        return create_training_job(db, request, ctx.config)

    @app.get("/api/v1/queues", response_model=QueueSnapshot)
    def get_queues(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> QueueSnapshot:
        return queues(db)

    @app.get("/api/v1/queues/count", response_model=QueueCountInfo)
    def get_queue_count(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> QueueCountInfo:
        return queue_count(db)

    @app.put("/api/v1/queues/training/enabled", response_model=QueueSnapshot)
    def put_training_enabled(
        request: QueueEnabledUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> QueueSnapshot:
        return set_queue_enabled(db, JobType.TRAINING, request, ctx.config)

    @app.put("/api/v1/queues/inference/enabled", response_model=QueueSnapshot)
    def put_inference_enabled(
        request: QueueEnabledUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> QueueSnapshot:
        return set_queue_enabled(db, JobType.INFERENCE, request, ctx.config)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobDetail)
    def get_job(
        job_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        return job_detail(db, job_id)

    @app.get("/api/v1/jobs/{job_id}/log", response_model=JobLogInfo)
    def get_job_log(
        job_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobLogInfo:
        return job_log(db, job_id, ctx.config)

    @app.delete("/api/v1/jobs/{job_id}", response_model=JobDetail)
    def delete_job_route(
        job_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        return delete_job(db, job_id)

    @app.post("/api/v1/jobs/{job_id}/move-up", response_model=JobDetail)
    def move_job_up(
        job_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        return move_job(db, job_id, direction=-1)

    @app.post("/api/v1/jobs/{job_id}/move-down", response_model=JobDetail)
    def move_job_down(
        job_id: uuid.UUID,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> JobDetail:
        return move_job(db, job_id, direction=1)


__all__ = ["register_job_routes"]

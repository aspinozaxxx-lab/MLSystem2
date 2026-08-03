"""Marshruty servernogo AOI-raspoznavaniya."""

from __future__ import annotations

import uuid

from fastapi import Body, Depends, FastAPI, Request, status
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._pseudolabel import (
    cancel_pseudolabel_job,
    create_pseudolabel_job,
    pseudolabel_classes,
    pseudolabel_job_info,
    pseudolabel_result,
)
from mlsystem2.training_ui_api.contracts import (
    PseudolabelAPIError,
    PseudolabelClassListResponse,
    PseudolabelJobCreate,
    PseudolabelJobInfo,
)

from .common import RouteContext


def register_pseudolabel_routes(app: FastAPI, context: RouteContext) -> None:
    """Zaregistrirovat marshruty AOI API."""

    def require_json(request: Request) -> None:
        """Otkazat ne-JSON telu do domennogo obrabotchika."""

        content_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
        if content_type not in {"application/json", "application/geo+json"}:
            raise PseudolabelAPIError(
                "UNSUPPORTED_CONTENT_TYPE",
                "Тело запроса должно иметь тип application/json.",
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            )

    @app.get(
        "/api/v1/pseudolabel/classes",
        response_model=PseudolabelClassListResponse,
        dependencies=[Depends(context.pseudolabel_authenticated)],
        tags=["pseudolabel"],
    )
    def list_classes(db: Session = Depends(context.get_db)) -> PseudolabelClassListResponse:
        """Vernut dostupnye klassy tekushchemu polzovatelyu."""

        return pseudolabel_classes(db, context.config)

    @app.post(
        "/api/v1/pseudolabel/jobs",
        response_model=PseudolabelJobInfo,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[
            Depends(context.pseudolabel_authenticated),
            Depends(require_json),
        ],
        tags=["pseudolabel"],
    )
    def create_job(
        payload: PseudolabelJobCreate = Body(...),
        db: Session = Depends(context.get_db),
    ) -> PseudolabelJobInfo:
        """Sozdat zafiksirovannoe AOI-zadanie."""

        return create_pseudolabel_job(db, payload, context.config)

    @app.get(
        "/api/v1/pseudolabel/jobs/{job_id}",
        response_model=PseudolabelJobInfo,
        dependencies=[Depends(context.pseudolabel_authenticated)],
        tags=["pseudolabel"],
    )
    def get_job(job_id: uuid.UUID, db: Session = Depends(context.get_db)) -> PseudolabelJobInfo:
        """Vernut publichnoe sostoyanie zadaniya."""

        return pseudolabel_job_info(db, job_id)

    @app.get(
        "/api/v1/pseudolabel/jobs/{job_id}/result",
        dependencies=[Depends(context.pseudolabel_authenticated)],
        tags=["pseudolabel"],
    )
    def get_result(job_id: uuid.UUID, db: Session = Depends(context.get_db)) -> dict[str, object]:
        """Vernut gotovyi GeoJSON bez servernyh putei."""

        return pseudolabel_result(db, job_id, context.config)

    @app.delete(
        "/api/v1/pseudolabel/jobs/{job_id}",
        response_model=PseudolabelJobInfo,
        dependencies=[Depends(context.pseudolabel_authenticated)],
        tags=["pseudolabel"],
    )
    def cancel_job(job_id: uuid.UUID, db: Session = Depends(context.get_db)) -> PseudolabelJobInfo:
        """Ostanovit aktivnoe zadanie."""

        return cancel_pseudolabel_job(db, job_id)


__all__ = ["register_pseudolabel_routes"]

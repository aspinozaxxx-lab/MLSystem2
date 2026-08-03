"""FastAPI application factory for the training UI API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ._auth import require_pseudolabel_user, require_user
from ._config import get_config
from ._database import Base, configure_schema, create_session_factory, session_scope
from ._dataset_catalog import synchronize_dataset_catalog
from ._markup_export import cleanup_expired_markup_exports
from ._routes.auth import register_auth_routes
from ._routes.automation import register_automation_routes
from ._routes.catalog import register_catalog_routes
from ._routes.common import RouteContext
from ._routes.export import register_export_routes
from ._routes.files import register_file_routes
from ._routes.frontend import register_frontend_routes
from ._routes.jobs import register_job_routes
from ._routes.pseudolabel import register_pseudolabel_routes
from ._routes.results import register_result_routes
from ._routes.templates import register_template_routes
from ._routes.test_samples import register_test_sample_routes
from ._service import ensure_seed_templates
from ._test_samples import (
    cleanup_test_sample_storage,
    reconcile_training_result_test_f1,
    recover_test_sample_batches,
    run_test_sample_batch_worker,
)
from ._worker import run_queue_worker
from .contracts import PseudolabelAPIError, TrainingUIAPIError


def create_app() -> FastAPI:
    config = get_config()
    configure_schema(None if config.database_url.startswith("sqlite") else config.database_schema)
    session_factory = create_session_factory(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        cleanup_expired_markup_exports(config, remove_incomplete=True)
        with session_factory() as session:
            synchronize_dataset_catalog(session, config)
            cleanup_test_sample_storage(session, config)
            recover_test_sample_batches(session)
            ensure_seed_templates(session)
            reconcile_training_result_test_f1(session, config)
            session.commit()
        worker_tasks: list[asyncio.Task[None]] = []
        if config.worker_enabled:
            worker_tasks = [
                asyncio.create_task(run_queue_worker(session_factory, config)),
                asyncio.create_task(run_test_sample_batch_worker(session_factory, config)),
            ]
        try:
            yield
        finally:
            for worker_task in worker_tasks:
                worker_task.cancel()
            for worker_task in worker_tasks:
                with suppress(asyncio.CancelledError):
                    await worker_task

    app = FastAPI(
        title="MLSystem2 Training UI API",
        description="Public FastAPI contract for the MLSystem2 training site.",
        version="0.1.0",
        lifespan=lifespan,
    )
    if config.cors_origin:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[config.cors_origin],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if config.database_url.startswith("sqlite"):
        Base.metadata.create_all(session_factory.kw["bind"])

    def get_db() -> Iterator[Session]:
        yield from session_scope(session_factory)

    def authenticated(request: Request) -> str:
        return require_user(request, config)

    def pseudolabel_authenticated(request: Request) -> str:
        """Proverit cookie ili otdelnyi Bearer-token QGIS."""

        return require_pseudolabel_user(request, config)

    route_context = RouteContext(
        config=config,
        get_db=get_db,
        authenticated=authenticated,
        pseudolabel_authenticated=pseudolabel_authenticated,
    )

    @app.exception_handler(PseudolabelAPIError)
    def pseudolabel_error_handler(_: Request, exc: PseudolabelAPIError):
        """Vernut stabilnyi JSON domennoi oshibki."""

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Ne menyat legacy-oshibki i strukturirovat tolko AOI API."""

        if request.url.path.startswith("/api/v1/pseudolabel/"):
            errors = [
                {
                    "field": ".".join(str(item) for item in error.get("loc", [])[1:]),
                    "type": str(error.get("type") or "validation_error"),
                }
                for error in exc.errors()
            ]
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content={
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Параметры запроса не прошли проверку.",
                        "details": {"errors": errors},
                    }
                },
            )
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(TrainingUIAPIError)
    def training_ui_error_handler(_: Request, exc: TrainingUIAPIError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "training_ui_api"}

    register_auth_routes(app, route_context)
    register_catalog_routes(app, route_context)
    register_export_routes(app, route_context)
    register_test_sample_routes(app, route_context)
    register_automation_routes(app, route_context)
    register_template_routes(app, route_context)
    register_job_routes(app, route_context)
    register_pseudolabel_routes(app, route_context)
    register_result_routes(app, route_context)
    register_file_routes(app, route_context)
    register_frontend_routes(app, config)
    return app


def main() -> None:
    import uvicorn

    config = get_config()
    uvicorn.run("mlsystem2.training_ui_api.api:create_app", host=config.host, port=config.port, factory=True)

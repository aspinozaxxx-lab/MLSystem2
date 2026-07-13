"""FastAPI application factory for the training UI API."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from ._auth import require_user
from ._config import get_config
from ._database import Base, configure_schema, create_session_factory, session_scope
from ._markup_export import cleanup_expired_markup_exports
from ._routes.auth import register_auth_routes
from ._routes.automation import register_automation_routes
from ._routes.catalog import register_catalog_routes
from ._routes.common import RouteContext
from ._routes.export import register_export_routes
from ._routes.files import register_file_routes
from ._routes.frontend import register_frontend_routes
from ._routes.jobs import register_job_routes
from ._routes.results import register_result_routes
from ._routes.templates import register_template_routes
from ._routes.test_samples import register_test_sample_routes
from ._service import ensure_seed_templates
from ._test_samples import cleanup_test_sample_storage
from ._worker import run_queue_worker
from .contracts import TrainingUIAPIError


def create_app() -> FastAPI:
    config = get_config()
    configure_schema(None if config.database_url.startswith("sqlite") else config.database_schema)
    session_factory = create_session_factory(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        cleanup_expired_markup_exports(config, remove_incomplete=True)
        with session_factory() as session:
            cleanup_test_sample_storage(session, config)
            ensure_seed_templates(session)
            session.commit()
        worker_task: asyncio.Task[None] | None = None
        if config.worker_enabled:
            worker_task = asyncio.create_task(run_queue_worker(session_factory, config))
        try:
            yield
        finally:
            if worker_task is not None:
                worker_task.cancel()
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

    route_context = RouteContext(config=config, get_db=get_db, authenticated=authenticated)

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
    register_result_routes(app, route_context)
    register_file_routes(app, route_context)
    register_frontend_routes(app, config)
    return app


def main() -> None:
    import uvicorn

    config = get_config()
    uvicorn.run("mlsystem2.training_ui_api.api:create_app", host=config.host, port=config.port, factory=True)

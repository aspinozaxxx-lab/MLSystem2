"""FastAPI-приложение training UI API."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from ._auth import current_user, login_response, logout_response, require_user, verify_credentials
from ._config import get_config
from ._database import Base, configure_schema, create_session_factory, session_scope
from ._models import StoredFileRow
from ._service import (
    app_links,
    class_results,
    classes,
    create_custom_dataset,
    create_mlflow_experiment,
    create_pseudo_markup_job,
    create_training_job,
    datasets,
    delete_job,
    ensure_seed_templates,
    job_detail,
    mlflow_experiments,
    models,
    move_job,
    queues,
    set_queue_enabled,
    stored_file,
    training_template,
    training_templates,
    update_training_template,
)
from .contracts import (
    AppLinksResponse,
    ClassListResponse,
    ClassResultsResponse,
    CustomDatasetInfo,
    DatasetListResponse,
    JobDetail,
    JobType,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ModelListResponse,
    QueueEnabledUpdate,
    QueueSnapshot,
    TrainingJobCreate,
    TrainingTemplate,
    TrainingTemplateListResponse,
    TrainingTemplateUpdate,
    TrainingUIAPIError,
)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str
    password: str


def create_app() -> FastAPI:
    config = get_config()
    configure_schema(None if config.database_url.startswith("sqlite") else config.database_schema)
    session_factory = create_session_factory(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        with session_factory() as session:
            ensure_seed_templates(session)
            session.commit()
        yield

    app = FastAPI(
        title="MLSystem2 Training UI API",
        description="Публичный FastAPI-контракт сайта обучения MLSystem2.",
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

    @app.exception_handler(TrainingUIAPIError)
    def training_ui_error_handler(_: Request, exc: TrainingUIAPIError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "training_ui_api"}

    @app.post("/api/v1/auth/login")
    def login(request: LoginRequest, response: Response) -> dict[str, str]:
        if not verify_credentials(request.username, request.password, config):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Неверный логин или пароль")
        login_response(response, request.username, config)
        return {"status": "ok"}

    @app.post("/api/v1/auth/logout")
    def logout(response: Response) -> dict[str, str]:
        logout_response(response, config)
        return {"status": "ok"}

    @app.get("/api/v1/auth/me")
    def me(request: Request) -> dict[str, str | bool | None]:
        user = current_user(request, config)
        return {"authenticated": user is not None, "username": user}

    @app.get("/auth/proxy-check", include_in_schema=False)
    def proxy_check(request: Request) -> Response:
        user = current_user(request, config)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Нужна авторизация")
        return Response(status_code=status.HTTP_204_NO_CONTENT, headers={"X-Remote-User": user})

    @app.get("/api/v1/app-links", response_model=AppLinksResponse)
    def get_app_links(_: str = Depends(authenticated)) -> AppLinksResponse:
        return app_links(config)

    @app.get("/api/v1/mlflow/experiments", response_model=list[MLflowExperimentInfo])
    def get_mlflow_experiments(_: str = Depends(authenticated)) -> list[MLflowExperimentInfo]:
        return mlflow_experiments(config)

    @app.post("/api/v1/mlflow/experiments", response_model=MLflowExperimentInfo)
    def post_mlflow_experiment(
        request: MLflowExperimentCreate,
        _: str = Depends(authenticated),
    ) -> MLflowExperimentInfo:
        return create_mlflow_experiment(request, config)

    @app.get("/api/v1/datasets", response_model=DatasetListResponse)
    def get_datasets(_: str = Depends(authenticated)) -> DatasetListResponse:
        return datasets(config)

    @app.get("/api/v1/classes", response_model=ClassListResponse)
    def get_classes(_: str = Depends(authenticated)) -> ClassListResponse:
        return classes(config)

    @app.post("/api/v1/custom-datasets", response_model=CustomDatasetInfo)
    async def post_custom_dataset(
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
        name: Annotated[str, Form()] = "Custom",
        scenes_txt: Annotated[UploadFile | None, File()] = None,
        annotation_geojson: Annotated[UploadFile | None, File()] = None,
    ) -> CustomDatasetInfo:
        if scenes_txt is None or annotation_geojson is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Нужны txt и geojson")
        return create_custom_dataset(
            db,
            name=name,
            scenes_name=scenes_txt.filename or "scenes.txt",
            scenes_content_type=scenes_txt.content_type,
            scenes_bytes=await scenes_txt.read(),
            annotation_name=annotation_geojson.filename or "annotation.geojson",
            annotation_content_type=annotation_geojson.content_type,
            annotation_bytes=await annotation_geojson.read(),
            config=config,
        )

    @app.get("/api/v1/models", response_model=ModelListResponse)
    def get_models(_: str = Depends(authenticated)) -> ModelListResponse:
        return models()

    @app.get("/api/v1/training-templates", response_model=TrainingTemplateListResponse)
    def get_training_templates(
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> TrainingTemplateListResponse:
        return training_templates(db)

    @app.get("/api/v1/training-templates/{architecture}", response_model=TrainingTemplate)
    def get_training_template(
        architecture: str,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> TrainingTemplate:
        return training_template(db, architecture)

    @app.put("/api/v1/training-templates/{architecture}", response_model=TrainingTemplate)
    def put_training_template(
        architecture: str,
        request: TrainingTemplateUpdate,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> TrainingTemplate:
        return update_training_template(db, architecture, request)

    @app.post("/api/v1/training-jobs", response_model=JobDetail)
    def post_training_job(
        request: TrainingJobCreate,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> JobDetail:
        return create_training_job(db, request, config)

    @app.get("/api/v1/queues", response_model=QueueSnapshot)
    def get_queues(
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> QueueSnapshot:
        return queues(db)

    @app.put("/api/v1/queues/training/enabled", response_model=QueueSnapshot)
    def put_training_enabled(
        request: QueueEnabledUpdate,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> QueueSnapshot:
        return set_queue_enabled(db, JobType.TRAINING, request, config)

    @app.put("/api/v1/queues/inference/enabled", response_model=QueueSnapshot)
    def put_inference_enabled(
        request: QueueEnabledUpdate,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> QueueSnapshot:
        return set_queue_enabled(db, JobType.INFERENCE, request, config)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobDetail)
    def get_job(
        job_id: uuid.UUID,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> JobDetail:
        return job_detail(db, job_id)

    @app.delete("/api/v1/jobs/{job_id}", response_model=JobDetail)
    def delete_job_route(
        job_id: uuid.UUID,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> JobDetail:
        return delete_job(db, job_id)

    @app.post("/api/v1/jobs/{job_id}/move-up", response_model=JobDetail)
    def move_job_up(
        job_id: uuid.UUID,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> JobDetail:
        return move_job(db, job_id, direction=-1)

    @app.post("/api/v1/jobs/{job_id}/move-down", response_model=JobDetail)
    def move_job_down(
        job_id: uuid.UUID,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> JobDetail:
        return move_job(db, job_id, direction=1)

    @app.get("/api/v1/results/classes", response_model=ClassListResponse)
    def get_result_classes(_: str = Depends(authenticated)) -> ClassListResponse:
        return classes(config)

    @app.get("/api/v1/results/classes/{class_key}", response_model=ClassResultsResponse)
    def get_class_results(
        class_key: str,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> ClassResultsResponse:
        return class_results(db, class_key, config)

    @app.post("/api/v1/results/classes/{class_key}/pseudo-markup", response_model=JobDetail)
    async def post_pseudo_markup(
        class_key: str,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
        dataset_key: Annotated[str | None, Form()] = None,
        training_result_id: Annotated[str | None, Form()] = None,
        scenes_txt: Annotated[UploadFile | None, File()] = None,
    ) -> JobDetail:
        parsed_training_result_id = uuid.UUID(training_result_id) if training_result_id else None
        scenes_bytes = await scenes_txt.read() if scenes_txt is not None else None
        return create_pseudo_markup_job(
            db,
            class_key=class_key,
            dataset_key=dataset_key,
            training_result_id=parsed_training_result_id,
            scenes_name=scenes_txt.filename if scenes_txt is not None else None,
            scenes_content_type=scenes_txt.content_type if scenes_txt is not None else None,
            scenes_bytes=scenes_bytes,
            config=config,
        )

    @app.get("/api/v1/files/{file_id}/download")
    def download_file(
        file_id: uuid.UUID,
        db: Session = Depends(get_db),
        _: str = Depends(authenticated),
    ) -> FileResponse:
        row: StoredFileRow = stored_file(db, file_id)
        path = row.path
        if not Path(path).is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден")
        return FileResponse(path, filename=row.original_name)

    index_path = config.frontend_dist / "index.html"
    assets_path = config.frontend_dist / "assets"
    if index_path.is_file():
        frontend_root = config.frontend_dist.resolve()

        def frontend_file(frontend_path: str) -> Path | None:
            candidate = (frontend_root / frontend_path).resolve()
            try:
                candidate.relative_to(frontend_root)
            except ValueError:
                return None
            return candidate if candidate.is_file() else None

        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="frontend-assets")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(index_path)

        @app.get("/{frontend_path:path}", include_in_schema=False)
        def frontend_fallback(frontend_path: str) -> FileResponse:
            if frontend_path.startswith(("api/", "docs", "redoc", "openapi.json")):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Файл не найден")
            if path := frontend_file(frontend_path):
                return FileResponse(path)
            return FileResponse(index_path)

    return app


def main() -> None:
    import uvicorn

    config = get_config()
    uvicorn.run("mlsystem2.training_ui_api.api:create_app", host=config.host, port=config.port, factory=True)

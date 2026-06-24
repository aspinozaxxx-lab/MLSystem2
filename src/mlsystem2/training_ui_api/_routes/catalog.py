"""Catalog, bootstrap, and MLflow metadata routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    app_links,
    bootstrap,
    classes,
    create_custom_dataset,
    create_mlflow_experiment,
    datasets,
    image_folders,
    mlflow_experiments,
    models,
)
from mlsystem2.training_ui_api.contracts import (
    AppLinksResponse,
    BootstrapInfo,
    ClassListResponse,
    CustomDatasetInfo,
    DatasetListResponse,
    ImageFolderListResponse,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ModelListResponse,
)

from .common import RouteContext


def register_catalog_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get("/api/v1/bootstrap", response_model=BootstrapInfo)
    def get_bootstrap(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> BootstrapInfo:
        return bootstrap(db, ctx.config)

    @app.get("/api/v1/app-links", response_model=AppLinksResponse)
    def get_app_links(_: str = Depends(ctx.authenticated)) -> AppLinksResponse:
        return app_links(ctx.config)

    @app.get("/api/v1/mlflow/experiments", response_model=list[MLflowExperimentInfo])
    def get_mlflow_experiments(_: str = Depends(ctx.authenticated)) -> list[MLflowExperimentInfo]:
        return mlflow_experiments(ctx.config)

    @app.post("/api/v1/mlflow/experiments", response_model=MLflowExperimentInfo)
    def post_mlflow_experiment(
        request: MLflowExperimentCreate,
        _: str = Depends(ctx.authenticated),
    ) -> MLflowExperimentInfo:
        return create_mlflow_experiment(request, ctx.config)

    @app.get("/api/v1/datasets", response_model=DatasetListResponse)
    def get_datasets(_: str = Depends(ctx.authenticated)) -> DatasetListResponse:
        return datasets(ctx.config)

    @app.get("/api/v1/image-folders", response_model=ImageFolderListResponse)
    def get_image_folders(_: str = Depends(ctx.authenticated)) -> ImageFolderListResponse:
        return image_folders(ctx.config)

    @app.get("/api/v1/classes", response_model=ClassListResponse)
    def get_classes(_: str = Depends(ctx.authenticated)) -> ClassListResponse:
        return classes(ctx.config)

    @app.get("/api/v1/models", response_model=ModelListResponse)
    def get_models(_: str = Depends(ctx.authenticated)) -> ModelListResponse:
        return models()

    @app.post("/api/v1/custom-datasets", response_model=CustomDatasetInfo)
    async def post_custom_dataset(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
        name: str = Form(default="Custom"),
        scenes_txt: UploadFile | None = File(default=None),
        annotation_geojson: UploadFile | None = File(default=None),
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
            config=ctx.config,
        )


__all__ = ["register_catalog_routes"]

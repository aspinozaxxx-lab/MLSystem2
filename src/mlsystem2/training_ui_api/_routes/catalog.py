"""Catalog, bootstrap, and MLflow metadata routes."""

from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._service import (
    app_links,
    bootstrap,
    classes,
    create_custom_dataset,
    create_dataset_class,
    create_managed_dataset,
    create_mlflow_experiment,
    dataset_catalog,
    datasets,
    image_folders,
    mlflow_experiments,
    models,
    set_primary_dataset,
    sync_dataset_catalog,
    update_dataset_class,
    update_managed_dataset,
)
from mlsystem2.training_ui_api.contracts import (
    AppLinksResponse,
    BootstrapInfo,
    ClassListResponse,
    CustomDatasetInfo,
    DatasetCatalogInfo,
    DatasetClassCreate,
    DatasetClassUpdate,
    DatasetListResponse,
    DatasetPrimaryDatasetUpdate,
    ImageFolderListResponse,
    MLflowExperimentCreate,
    MLflowExperimentInfo,
    ManagedDatasetCreate,
    ManagedDatasetUpdate,
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
    def get_datasets(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetListResponse:
        return datasets(db, ctx.config)

    @app.get("/api/v1/image-folders", response_model=ImageFolderListResponse)
    def get_image_folders(_: str = Depends(ctx.authenticated)) -> ImageFolderListResponse:
        return image_folders(ctx.config)

    @app.get("/api/v1/classes", response_model=ClassListResponse)
    def get_classes(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> ClassListResponse:
        return classes(db, ctx.config)

    @app.get("/api/v1/dataset-catalog", response_model=DatasetCatalogInfo)
    def get_dataset_catalog(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return dataset_catalog(db, ctx.config)

    @app.post("/api/v1/dataset-catalog/sync", response_model=DatasetCatalogInfo)
    def post_dataset_catalog_sync(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return sync_dataset_catalog(db, ctx.config)

    @app.post("/api/v1/dataset-classes", response_model=DatasetCatalogInfo)
    def post_dataset_class(
        request: DatasetClassCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return create_dataset_class(db, request, ctx.config)

    @app.patch("/api/v1/dataset-classes/{class_key}", response_model=DatasetCatalogInfo)
    def patch_dataset_class(
        class_key: str,
        request: DatasetClassUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return update_dataset_class(db, class_key, request, ctx.config)

    @app.put(
        "/api/v1/dataset-classes/{class_key}/primary-dataset",
        response_model=DatasetCatalogInfo,
    )
    def put_primary_dataset(
        class_key: str,
        request: DatasetPrimaryDatasetUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return set_primary_dataset(db, class_key, request, ctx.config)

    @app.post("/api/v1/managed-datasets", response_model=DatasetCatalogInfo)
    def post_managed_dataset(
        request: ManagedDatasetCreate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return create_managed_dataset(db, request, ctx.config)

    @app.patch("/api/v1/managed-datasets/{dataset_key}", response_model=DatasetCatalogInfo)
    def patch_managed_dataset(
        dataset_key: str,
        request: ManagedDatasetUpdate,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetCatalogInfo:
        return update_managed_dataset(db, dataset_key, request, ctx.config)

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

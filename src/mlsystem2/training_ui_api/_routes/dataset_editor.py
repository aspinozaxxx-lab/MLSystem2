"""HTTP-маршруты редактора per-image датасетов."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from mlsystem2.training_ui_api._dataset_editor import (
    DatasetEditorConflict,
    DatasetEditorGitError,
    add_editor_scenes,
    browse_editor_rasters,
    delete_editor_dataset,
    delete_editor_scene,
    discard_editor_drafts,
    editor_pseudo_job_info,
    editor_publication_info,
    editor_scene_detail,
    editor_scene_pseudo_markup,
    list_editor_datasets,
    list_editor_scenes,
    publish_editor_drafts,
    publish_editor_scenes,
    preview_editor_dataset_rebuild,
    rebuild_editor_dataset,
    resolve_editor_raster,
    save_editor_scene,
    save_editor_draft,
)
from mlsystem2.training_ui_api.contracts import (
    DatasetEditorAddScenesRequest,
    DatasetEditorDatasetListResponse,
    DatasetEditorDeleteSceneRequest,
    DatasetEditorDiscardDraftsResult,
    DatasetEditorDraftInfo,
    DatasetEditorMutationResult,
    DatasetEditorPublishRequest,
    DatasetEditorPublicationInfo,
    DatasetEditorPseudoMarkupInfo,
    DatasetEditorRasterBrowserResponse,
    DatasetEditorRebuildPreview,
    DatasetEditorRebuildRequest,
    DatasetEditorRebuildResult,
    DatasetEditorSaveSceneRequest,
    DatasetEditorSaveDraftRequest,
    DatasetEditorSceneDetail,
    DatasetEditorSceneListResponse,
)

from .common import RouteContext


_STREAM_CHUNK_SIZE = 1024 * 1024


def register_dataset_editor_routes(app: FastAPI, ctx: RouteContext) -> None:
    @app.get(
        "/api/v1/dataset-editor/datasets",
        response_model=DatasetEditorDatasetListResponse,
    )
    def datasets(
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorDatasetListResponse:
        return _git_call(list_editor_datasets, db, ctx.config)

    @app.get(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes",
        response_model=DatasetEditorSceneListResponse,
    )
    def scenes(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorSceneListResponse:
        return _git_call(
            list_editor_scenes,
            db,
            ctx.config,
            dataset_key,
            username=username,
        )

    @app.delete(
        "/api/v1/dataset-editor/datasets/{dataset_key}",
        response_model=DatasetEditorMutationResult,
    )
    def delete_dataset(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            delete_editor_dataset,
            db,
            ctx.config,
            dataset_key,
            username=username,
        )

    @app.get(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}",
        response_model=DatasetEditorSceneDetail,
    )
    def scene(
        dataset_key: str,
        annotation_name: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorSceneDetail:
        return _git_call(
            editor_scene_detail,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            username=username,
        )

    @app.put(
        "/api/v1/dataset-editor/datasets/{dataset_key}/drafts/{annotation_name}",
        response_model=DatasetEditorDraftInfo,
    )
    def save_draft(
        dataset_key: str,
        annotation_name: str,
        request: DatasetEditorSaveDraftRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorDraftInfo:
        return _git_call(
            save_editor_draft,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            base_revision=request.base_revision,
            geojson=request.geojson,
            username=username,
        )

    @app.delete(
        "/api/v1/dataset-editor/datasets/{dataset_key}/drafts/{annotation_name}",
        response_model=DatasetEditorDiscardDraftsResult,
    )
    def discard_draft(
        dataset_key: str,
        annotation_name: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorDiscardDraftsResult:
        return discard_editor_drafts(
            db,
            dataset_key,
            username=username,
            annotation_name=annotation_name,
        )

    @app.delete(
        "/api/v1/dataset-editor/datasets/{dataset_key}/drafts",
        response_model=DatasetEditorDiscardDraftsResult,
    )
    def discard_drafts(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorDiscardDraftsResult:
        return discard_editor_drafts(db, dataset_key, username=username)

    @app.post(
        "/api/v1/dataset-editor/datasets/{dataset_key}/drafts/publish",
        response_model=DatasetEditorMutationResult,
    )
    def publish_drafts(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            publish_editor_drafts,
            db,
            ctx.config,
            dataset_key,
            username=username,
        )

    @app.get(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}/pseudo-markup",
        response_model=DatasetEditorPseudoMarkupInfo,
    )
    def scene_pseudo_markup(
        dataset_key: str,
        annotation_name: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorPseudoMarkupInfo:
        return _git_call(
            editor_scene_pseudo_markup,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            ensure=False,
        )

    @app.post(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}/pseudo-markup",
        response_model=DatasetEditorPseudoMarkupInfo,
    )
    def ensure_scene_pseudo_markup(
        dataset_key: str,
        annotation_name: str,
        retry: bool = Query(default=False),
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorPseudoMarkupInfo:
        return _git_call(
            editor_scene_pseudo_markup,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            ensure=True,
            retry=retry,
        )

    @app.get(
        "/api/v1/dataset-editor/pseudo-markup/{job_id}",
        response_model=DatasetEditorPseudoMarkupInfo,
    )
    def pseudo_markup_job(
        job_id: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorPseudoMarkupInfo:
        from uuid import UUID

        try:
            parsed_job_id = UUID(job_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Задание псевдоразметки снимка не найдено",
            ) from exc
        return editor_pseudo_job_info(db, ctx.config, parsed_job_id)

    @app.get(
        "/api/v1/dataset-editor/datasets/{dataset_key}/rasters",
        response_model=DatasetEditorRasterBrowserResponse,
    )
    def rasters(
        dataset_key: str,
        folder: str = Query(default=""),
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorRasterBrowserResponse:
        return browse_editor_rasters(db, ctx.config, dataset_key, folder)

    @app.post(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes",
        response_model=DatasetEditorMutationResult,
    )
    def add_scenes(
        dataset_key: str,
        request: DatasetEditorAddScenesRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            add_editor_scenes,
            db,
            ctx.config,
            dataset_key,
            image_paths=request.image_paths,
            folder_path=request.folder_path,
            username=username,
        )

    @app.put(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes",
        response_model=DatasetEditorMutationResult,
    )
    def publish_scenes(
        dataset_key: str,
        request: DatasetEditorPublishRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            publish_editor_scenes,
            db,
            ctx.config,
            dataset_key,
            scenes=[
                (scene.annotation_name, scene.revision, scene.geojson)
                for scene in request.scenes
            ],
            username=username,
        )

    @app.put(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}",
        response_model=DatasetEditorMutationResult,
    )
    def save_scene(
        dataset_key: str,
        annotation_name: str,
        request: DatasetEditorSaveSceneRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            save_editor_scene,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            revision=request.revision,
            geojson=request.geojson,
            username=username,
        )

    @app.delete(
        "/api/v1/dataset-editor/datasets/{dataset_key}/scenes/{annotation_name}",
        response_model=DatasetEditorMutationResult,
    )
    def delete_scene(
        dataset_key: str,
        annotation_name: str,
        request: DatasetEditorDeleteSceneRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorMutationResult:
        return _git_call(
            delete_editor_scene,
            db,
            ctx.config,
            dataset_key,
            annotation_name,
            revision=request.revision,
            username=username,
        )

    @app.get(
        "/api/v1/dataset-editor/publication/{commit}",
        response_model=DatasetEditorPublicationInfo,
    )
    def publication(
        commit: str,
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorPublicationInfo:
        return _git_call(editor_publication_info, ctx.config, commit)

    @app.post(
        "/api/v1/dataset-editor/datasets/{dataset_key}/rebuild/preview",
        response_model=DatasetEditorRebuildPreview,
    )
    def rebuild_preview(
        dataset_key: str,
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> DatasetEditorRebuildPreview:
        return _git_call(
            preview_editor_dataset_rebuild,
            db,
            ctx.config,
            dataset_key,
        )

    @app.post(
        "/api/v1/dataset-editor/datasets/{dataset_key}/rebuild",
        response_model=DatasetEditorRebuildResult,
    )
    def rebuild(
        dataset_key: str,
        request: DatasetEditorRebuildRequest,
        db: Session = Depends(ctx.get_db),
        username: str = Depends(ctx.authenticated),
    ) -> DatasetEditorRebuildResult:
        return _git_call(
            rebuild_editor_dataset,
            db,
            ctx.config,
            dataset_key,
            preview_token=request.preview_token,
            mode=request.mode,
            username=username,
        )

    @app.get("/api/v1/dataset-editor/datasets/{dataset_key}/raster/{image_path:path}")
    def raster(
        dataset_key: str,
        image_path: str,
        range_header: str | None = Header(default=None, alias="Range"),
        db: Session = Depends(ctx.get_db),
        _: str = Depends(ctx.authenticated),
    ) -> StreamingResponse:
        path = resolve_editor_raster(db, ctx.config, dataset_key, image_path)
        return _raster_response(path, range_header)


def _git_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except DatasetEditorConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DatasetEditorGitError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


def _raster_response(path: Path, range_header: str | None) -> StreamingResponse:
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "image/tiff",
    }
    if range_header is None:
        headers["Content-Length"] = str(size)
        return StreamingResponse(
            _file_chunks(path, 0, size - 1),
            status_code=status.HTTP_200_OK,
            media_type="image/tiff",
            headers=headers,
        )
    start, end = _parse_range(range_header, size)
    headers.update(
        {
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
        }
    )
    return StreamingResponse(
        _file_chunks(path, start, end),
        status_code=status.HTTP_206_PARTIAL_CONTENT,
        media_type="image/tiff",
        headers=headers,
    )


def _parse_range(value: str, size: int) -> tuple[int, int]:
    if not value.startswith("bytes=") or "," in value or size <= 0:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}"},
        )
    raw_start, separator, raw_end = value[6:].partition("-")
    if not separator:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}"},
        )
    try:
        if raw_start:
            start = int(raw_start)
            end = int(raw_end) if raw_end else size - 1
        else:
            suffix_length = int(raw_end)
            if suffix_length <= 0:
                raise ValueError
            start = max(0, size - suffix_length)
            end = size - 1
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}"},
        ) from exc
    if start < 0 or start >= size or end < start:
        raise HTTPException(
            status_code=status.HTTP_416_REQUESTED_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}"},
        )
    return start, min(end, size - 1)


def _file_chunks(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as stream:
        stream.seek(start)
        while remaining > 0:
            chunk = stream.read(min(_STREAM_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


__all__ = ["register_dataset_editor_routes"]

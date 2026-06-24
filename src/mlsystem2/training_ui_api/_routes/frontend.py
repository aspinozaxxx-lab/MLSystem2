"""Static frontend routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mlsystem2.training_ui_api._config import TrainingUIAPIConfig


def register_frontend_routes(app: FastAPI, config: TrainingUIAPIConfig) -> None:
    index_path = config.frontend_dist / "index.html"
    assets_path = config.frontend_dist / "assets"
    if not index_path.is_file():
        return

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


__all__ = ["register_frontend_routes"]

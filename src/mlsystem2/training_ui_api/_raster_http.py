"""Потоковое чтение TIFF с поддержкой HTTP Range."""

from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

_STREAM_CHUNK_SIZE = 1024 * 1024


def raster_revision(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns:x}-{stat.st_ctime_ns:x}-{stat.st_size:x}"


def raster_response(path: Path, range_header: str | None, version: str | None = None) -> StreamingResponse:
    revision = raster_revision(path)
    if version is not None and version != revision:
        raise HTTPException(status_code=412, detail="Снимок изменился. Обновите страницу просмотра.")
    size = path.stat().st_size
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Type": "image/tiff",
        "ETag": f'"{revision}"',
        # Фрагменты хранит ограниченный кэш приложения после проверки входа.
        "Cache-Control": "private, no-store",
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

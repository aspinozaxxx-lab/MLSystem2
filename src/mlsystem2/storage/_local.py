"""Реализация хранилища на локальной файловой системе."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

from .contracts import ResolvedUri, StorageError, StorageObject


def resolve_local_uri(uri: str) -> ResolvedUri:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        path = Path(parsed.path)
    else:
        path = Path(uri)
    return ResolvedUri(uri=uri, scheme="local", path=str(path))


def exists(uri: str) -> bool:
    resolved = resolve_local_uri(uri)
    if resolved.path is None:
        return False
    return Path(resolved.path).exists()


def list_files(uri: str, *, suffixes: tuple[str, ...] | None = None) -> list[StorageObject]:
    resolved = resolve_local_uri(uri)
    if resolved.path is None:
        raise StorageError(f"Локальный URI не разрешился в путь: {uri}")

    root = Path(resolved.path)
    if not root.exists():
        return []

    candidates = [root] if root.is_file() else (path for path in root.rglob("*") if path.is_file())
    objects: list[StorageObject] = []
    for path in candidates:
        if suffixes is not None and not path.name.endswith(suffixes):
            continue
        stat = path.stat()
        objects.append(
            StorageObject(
                uri=str(path),
                name=path.name,
                size_bytes=stat.st_size,
                modified_at=None,
            )
        )
    return objects


def read_json(uri: str) -> dict[str, object]:
    resolved = resolve_local_uri(uri)
    if resolved.path is None:
        raise StorageError(f"Локальный URI не разрешился в путь: {uri}")
    path = Path(resolved.path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Не удалось прочитать JSON из {uri}") from exc
    if not isinstance(payload, dict):
        raise StorageError(f"JSON-данные должны быть объектом: {uri}")
    return payload


def write_json(uri: str, payload: dict[str, object]) -> None:
    resolved = resolve_local_uri(uri)
    if resolved.path is None:
        raise StorageError(f"Локальный URI не разрешился в путь: {uri}")
    path = Path(resolved.path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
    except OSError as exc:
        raise StorageError(f"Не удалось записать JSON в {uri}") from exc

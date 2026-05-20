"""Публичный фасад хранилища."""

from __future__ import annotations

from . import _local, _s3
from .contracts import ResolvedUri, StorageObject


def resolve_uri(uri: str) -> ResolvedUri:
    if uri.startswith("s3://"):
        return _s3.resolve_s3_uri(uri)
    return _local.resolve_local_uri(uri)


def exists(uri: str) -> bool:
    if uri.startswith("s3://"):
        return _s3.exists(uri)
    return _local.exists(uri)


def list_files(uri: str, *, suffixes: tuple[str, ...] | None = None) -> list[StorageObject]:
    if uri.startswith("s3://"):
        return _s3.list_files(uri, suffixes=suffixes)
    return _local.list_files(uri, suffixes=suffixes)


def read_json(uri: str) -> dict[str, object]:
    if uri.startswith("s3://"):
        return _s3.read_json(uri)
    return _local.read_json(uri)


def write_json(uri: str, payload: dict[str, object]) -> None:
    if uri.startswith("s3://"):
        _s3.write_json(uri, payload)
        return
    _local.write_json(uri, payload)


__all__ = ["resolve_uri", "exists", "list_files", "read_json", "write_json"]

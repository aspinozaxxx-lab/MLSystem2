"""Реализация S3-хранилища."""

from __future__ import annotations

import json
from urllib.parse import urlparse

from .contracts import ResolvedUri, StorageError, StorageObject


def resolve_s3_uri(uri: str) -> ResolvedUri:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise StorageError(f"Некорректный S3 URI: {uri}")
    return ResolvedUri(uri=uri, scheme="s3", bucket=parsed.netloc, key=parsed.path.lstrip("/"))


def _client():
    try:
        import boto3
    except ImportError as exc:
        raise StorageError("Для операций S3-хранилища требуется boto3") from exc
    return boto3.client("s3")


def exists(uri: str) -> bool:
    resolved = resolve_s3_uri(uri)
    client = _client()
    try:
        client.head_object(Bucket=resolved.bucket, Key=resolved.key)
    except Exception:
        return False
    return True


def list_files(uri: str, *, suffixes: tuple[str, ...] | None = None) -> list[StorageObject]:
    resolved = resolve_s3_uri(uri)
    client = _client()
    objects: list[StorageObject] = []
    token: str | None = None

    while True:
        kwargs: dict[str, object] = {"Bucket": resolved.bucket, "Prefix": resolved.key or ""}
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item["Key"]
            name = key.rsplit("/", 1)[-1]
            if suffixes is not None and not name.endswith(suffixes):
                continue
            objects.append(
                StorageObject(
                    uri=f"s3://{resolved.bucket}/{key}",
                    name=name,
                    size_bytes=item.get("Size"),
                    modified_at=str(item.get("LastModified")) if item.get("LastModified") else None,
                )
            )
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return objects


def read_json(uri: str) -> dict[str, object]:
    resolved = resolve_s3_uri(uri)
    client = _client()
    try:
        response = client.get_object(Bucket=resolved.bucket, Key=resolved.key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except Exception as exc:
        raise StorageError(f"Не удалось прочитать JSON из {uri}") from exc
    if not isinstance(payload, dict):
        raise StorageError(f"JSON-данные должны быть объектом: {uri}")
    return payload


def write_json(uri: str, payload: dict[str, object]) -> None:
    resolved = resolve_s3_uri(uri)
    client = _client()
    try:
        client.put_object(
            Bucket=resolved.bucket,
            Key=resolved.key,
            Body=json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception as exc:
        raise StorageError(f"Не удалось записать JSON в {uri}") from exc

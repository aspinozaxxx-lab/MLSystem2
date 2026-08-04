"""Подготовка внешних RGB-тайлов для одноразового AOI-инференса."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union


WEB_MERCATOR_LIMIT = 20037508.342789244
WEB_MERCATOR_WORLD = WEB_MERCATOR_LIMIT * 2.0
DEFAULT_TILE_SIZE = 256


class ExternalImageryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ExternalImageryResult:
    images_root: Path
    source_image_ids: tuple[str, ...]
    coverage_percent: float
    warnings: tuple[str, ...]
    attribution: str
    license_url: str
    download_sec: float


@dataclass(frozen=True)
class _TileRequest:
    url: str
    column: int
    row: int
    bounds_3857: tuple[float, float, float, float]


@dataclass(frozen=True)
class _TileGrid:
    requests: tuple[_TileRequest, ...]
    transform: Affine
    width: int
    height: int
    tile_size: int
    source_id: str


def prepare_external_imagery(
    config: dict[str, Any],
    aoi_wgs84: BaseGeometry,
    run_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> ExternalImageryResult:
    """Скачать нужные тайлы и записать временные геопривязанные RGB-мозаики."""

    started = time.perf_counter()
    settings = dict(config.get("source_settings") or {})
    source_id = str(config.get("source_id") or "external")
    source_kind = str(config.get("source_kind") or "")
    source_protocol = str(config.get("source_protocol") or "").lower()
    target_resolution = _positive_float(config.get("target_resolution_m"))
    if target_resolution is None:
        raise ExternalImageryError(
            "MODEL_RESOLUTION_UNAVAILABLE",
            "Не удалось определить целевое разрешение модели для внешнего источника.",
        )
    output_root = run_root / "external-imagery"
    output_root.mkdir(parents=True, exist_ok=True)
    if progress is not None:
        progress("downloading_imagery")

    providers: list[tuple[str, dict[str, Any], BaseGeometry]]
    attribution = str(config.get("source_attribution") or "")
    license_url = str(config.get("source_license_url") or "")
    if source_kind == "catalog" and source_id == "openaerialmap":
        providers, attribution = _openaerialmap_providers(
            settings,
            aoi_wgs84,
            target_resolution,
            attribution,
        )
    else:
        providers = [(source_id, settings, aoi_wgs84)]

    all_bounds: list[BaseGeometry] = []
    all_ids: list[str] = []
    warnings: list[str] = []
    for number, (provider_id, provider, provider_aoi) in enumerate(providers, start=1):
        protocol = str(provider.get("protocol") or source_protocol).lower()
        grid = (
            _wmts_grid(provider_id, provider, provider_aoi, target_resolution)
            if protocol == "wmts"
            else _slippy_grid(provider_id, provider, provider_aoi, target_resolution, protocol)
        )
        _ensure_disk_space(output_root, grid)
        output_path = output_root / f"{number:03d}_{_safe_name(provider_id)}.tif"
        downloaded_bounds, missing = _download_grid(
            grid,
            provider,
            output_path,
            workers=max(1, int(config.get("external_http_workers") or 8)),
        )
        if not downloaded_bounds:
            output_path.unlink(missing_ok=True)
            warnings.append(f"Источник {provider_id} не вернул ни одного тайла.")
            continue
        all_ids.append(provider_id)
        all_bounds.extend(downloaded_bounds)
        if missing:
            warnings.append(
                f"Источник {provider_id}: недоступно тайлов {missing} из {len(grid.requests)}."
            )

    if not all_ids:
        raise ExternalImageryError(
            "SOURCE_IMAGES_NOT_FOUND",
            "Внешний источник не содержит доступных снимков для зоны интереса.",
        )
    coverage = _coverage_percent(aoi_wgs84, all_bounds)
    if coverage <= 0:
        raise ExternalImageryError(
            "SOURCE_IMAGES_NOT_FOUND",
            "Зона интереса не покрывается загруженными внешними тайлами.",
        )
    if coverage < 99.999:
        warnings.append(f"Внешними снимками покрыто {coverage:.2f}% зоны интереса.")
    return ExternalImageryResult(
        images_root=output_root,
        source_image_ids=tuple(all_ids),
        coverage_percent=coverage,
        warnings=tuple(warnings),
        attribution=attribution,
        license_url=license_url,
        download_sec=time.perf_counter() - started,
    )


def _openaerialmap_providers(
    settings: dict[str, Any],
    aoi_wgs84: BaseGeometry,
    target_resolution: float,
    fallback_attribution: str,
) -> tuple[list[tuple[str, dict[str, Any], BaseGeometry]], str]:
    catalog_url = str(settings.get("catalog_url") or "https://api.openaerialmap.org/meta")
    min_x, min_y, max_x, max_y = aoi_wgs84.bounds
    url = _append_query(
        catalog_url,
        {"bbox": f"{min_x},{min_y},{max_x},{max_y}", "limit": "100"},
    )
    payload = json.loads(
        _request_bytes(
            url,
            headers={"User-Agent": "MLSystem2/0.6"},
            timeout=float(settings.get("request_timeout_seconds") or 60),
        ).decode("utf-8")
    )
    raw_results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(raw_results, list):
        raise ExternalImageryError(
            "EXTERNAL_CATALOG_INVALID",
            "Каталог OpenAerialMap вернул ответ неизвестного формата.",
        )
    ranked: list[tuple[float, float, str, dict[str, Any], BaseGeometry]] = []
    for item in raw_results:
        properties = item.get("properties") if isinstance(item, dict) else None
        if not isinstance(properties, dict):
            continue
        tms = str(properties.get("tms") or "").strip()
        if not tms:
            continue
        footprint = _catalog_geometry(item, properties)
        overlap = footprint.intersection(aoi_wgs84)
        if overlap.is_empty or overlap.area <= 0:
            continue
        gsd = _positive_float(properties.get("gsd")) or target_resolution
        distance = abs(math.log(max(gsd, 1e-9) / target_resolution))
        acquired = _datetime_score(properties.get("acquisition_start"))
        identifier = str(
            item.get("_id")
            or properties.get("uuid")
            or properties.get("title")
            or f"oam-{len(ranked) + 1}"
        )
        provider = {
            "protocol": "tms",
            "url_template": tms,
            "min_zoom": properties.get("min_zoom", 0),
            "max_zoom": properties.get("max_zoom", 24),
            "request_timeout_seconds": settings.get("request_timeout_seconds", 60),
            "auth": {"type": "none"},
        }
        ranked.append((distance, -acquired, identifier, provider, footprint))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    uncovered = aoi_wgs84
    selected: list[tuple[str, dict[str, Any], BaseGeometry]] = []
    attributions = [fallback_attribution] if fallback_attribution else []
    for _, _, identifier, provider, footprint in ranked:
        contribution = uncovered.intersection(footprint)
        if contribution.is_empty or contribution.area <= 0:
            continue
        selected.append((f"openaerialmap:{identifier}", provider, contribution))
        uncovered = uncovered.difference(footprint)
        if uncovered.is_empty or uncovered.area <= aoi_wgs84.area * 1e-6:
            break
    if not selected:
        raise ExternalImageryError(
            "SOURCE_IMAGES_NOT_FOUND",
            "OpenAerialMap не содержит снимков для зоны интереса.",
        )
    attributions.append("OpenAerialMap, лицензия CC BY 4.0")
    return selected, "; ".join(dict.fromkeys(value for value in attributions if value))


def _catalog_geometry(item: dict[str, Any], properties: dict[str, Any]) -> BaseGeometry:
    geometry = item.get("geometry") or properties.get("footprint")
    if isinstance(geometry, dict):
        try:
            return shape(geometry)
        except (TypeError, ValueError):
            pass
    bbox_value = item.get("bbox") or properties.get("bbox")
    if isinstance(bbox_value, (list, tuple)) and len(bbox_value) >= 4:
        return box(*map(float, bbox_value[:4]))
    return box(-180, -90, 180, 90)


def _slippy_grid(
    source_id: str,
    settings: dict[str, Any],
    aoi_wgs84: BaseGeometry,
    target_resolution: float,
    protocol: str,
) -> _TileGrid:
    template = os.path.expandvars(str(settings.get("url_template") or "").strip())
    if not template:
        raise ExternalImageryError(
            "IMAGERY_PROVIDER_INVALID",
            f"Для источника {source_id} не задан url_template.",
        )
    min_zoom = int(settings.get("min_zoom") or 0)
    max_zoom = int(settings.get("max_zoom") or 24)
    zoom = int(round(math.log2(156543.03392804097 / target_resolution)))
    zoom = min(max_zoom, max(min_zoom, zoom))
    tile_size = int(settings.get("tile_size") or DEFAULT_TILE_SIZE)
    expanded = _expanded_aoi_3857(aoi_wgs84, target_resolution, settings)
    min_x, min_y, max_x, max_y = expanded.bounds
    scale = 2**zoom
    x0 = max(0, min(scale - 1, int(math.floor((min_x + WEB_MERCATOR_LIMIT) / WEB_MERCATOR_WORLD * scale))))
    x1 = max(0, min(scale - 1, int(math.floor((max_x + WEB_MERCATOR_LIMIT) / WEB_MERCATOR_WORLD * scale))))
    y0 = max(0, min(scale - 1, int(math.floor((WEB_MERCATOR_LIMIT - max_y) / WEB_MERCATOR_WORLD * scale))))
    y1 = max(0, min(scale - 1, int(math.floor((WEB_MERCATOR_LIMIT - min_y) / WEB_MERCATOR_WORLD * scale))))
    resolution = WEB_MERCATOR_WORLD / (scale * tile_size)
    origin_x = -WEB_MERCATOR_LIMIT + x0 * tile_size * resolution
    origin_y = WEB_MERCATOR_LIMIT - y0 * tile_size * resolution
    requests: list[_TileRequest] = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            request_y = scale - 1 - y if protocol == "tms" else y
            url = template.format(z=zoom, x=x, y=request_y)
            left = -WEB_MERCATOR_LIMIT + x * tile_size * resolution
            top = WEB_MERCATOR_LIMIT - y * tile_size * resolution
            requests.append(
                _TileRequest(
                    url=url,
                    column=x - x0,
                    row=y - y0,
                    bounds_3857=(left, top - tile_size * resolution, left + tile_size * resolution, top),
                )
            )
    return _TileGrid(
        requests=tuple(requests),
        transform=Affine(resolution, 0, origin_x, 0, -resolution, origin_y),
        width=(x1 - x0 + 1) * tile_size,
        height=(y1 - y0 + 1) * tile_size,
        tile_size=tile_size,
        source_id=source_id,
    )


def _wmts_grid(
    source_id: str,
    settings: dict[str, Any],
    aoi_wgs84: BaseGeometry,
    target_resolution: float,
) -> _TileGrid:
    capabilities_url = os.path.expandvars(
        str(settings.get("capabilities_url") or "").strip()
    )
    if not capabilities_url:
        raise ExternalImageryError(
            "IMAGERY_PROVIDER_INVALID",
            f"Для WMTS-источника {source_id} не задан capabilities_url.",
        )
    headers, query = _authorization(settings)
    capabilities = _request_bytes(
        _append_query(
            capabilities_url,
            {
                **query,
                "SERVICE": "WMTS",
                "REQUEST": "GetCapabilities",
                "VERSION": "1.0.0",
            },
        ),
        headers=headers,
        timeout=float(settings.get("request_timeout_seconds") or 60),
    )
    root = ET.fromstring(capabilities)
    layer_id = str(settings.get("layer") or "").strip()
    matrix_set_id = str(settings.get("matrix_set") or "").strip()
    layer = _wmts_named_element(root, "Layer", layer_id)
    if layer is None:
        raise ExternalImageryError("WMTS_LAYER_NOT_FOUND", f"Слой WMTS {layer_id} не найден.")
    if not matrix_set_id:
        matrix_set_id = _descendant_text(layer, "TileMatrixSet") or ""
    matrix_set = _wmts_named_element(root, "TileMatrixSet", matrix_set_id)
    if matrix_set is None:
        raise ExternalImageryError(
            "WMTS_MATRIX_SET_NOT_FOUND",
            f"Набор матриц WMTS {matrix_set_id} не найден.",
        )
    crs_text = _descendant_text(matrix_set, "SupportedCRS") or "EPSG:3857"
    if "3857" not in crs_text and "900913" not in crs_text:
        raise ExternalImageryError(
            "WMTS_CRS_UNSUPPORTED",
            "Внешний WMTS сейчас должен предоставлять матрицу EPSG:3857.",
        )
    matrices: list[tuple[float, ET.Element]] = []
    for matrix in _children(matrix_set, "TileMatrix"):
        scale = _float_text(matrix, "ScaleDenominator")
        if scale is not None:
            matrices.append((scale * 0.00028, matrix))
    if not matrices:
        raise ExternalImageryError("WMTS_MATRIX_INVALID", "WMTS не содержит пригодных матриц.")
    resolution, matrix = min(matrices, key=lambda item: abs(math.log(item[0] / target_resolution)))
    matrix_id = _descendant_text(matrix, "Identifier") or ""
    top_left = (_descendant_text(matrix, "TopLeftCorner") or "").split()
    if len(top_left) != 2:
        raise ExternalImageryError("WMTS_MATRIX_INVALID", "WMTS не содержит TopLeftCorner.")
    origin_x, origin_y = map(float, top_left)
    tile_width = int(_float_text(matrix, "TileWidth") or DEFAULT_TILE_SIZE)
    tile_height = int(_float_text(matrix, "TileHeight") or DEFAULT_TILE_SIZE)
    matrix_width = int(_float_text(matrix, "MatrixWidth") or 0)
    matrix_height = int(_float_text(matrix, "MatrixHeight") or 0)
    expanded = _expanded_aoi_3857(aoi_wgs84, target_resolution, settings)
    min_x, min_y, max_x, max_y = expanded.bounds
    col0 = max(0, int(math.floor((min_x - origin_x) / (tile_width * resolution))))
    col1 = min(matrix_width - 1, int(math.floor((max_x - origin_x) / (tile_width * resolution))))
    row0 = max(0, int(math.floor((origin_y - max_y) / (tile_height * resolution))))
    row1 = min(matrix_height - 1, int(math.floor((origin_y - min_y) / (tile_height * resolution))))
    base_url = os.path.expandvars(str(settings.get("get_tile_url") or capabilities_url))
    style = str(settings.get("style") or "default")
    image_format = str(settings.get("format") or "image/png")
    requests: list[_TileRequest] = []
    for row in range(row0, row1 + 1):
        for column in range(col0, col1 + 1):
            url = _append_query(
                base_url,
                {
                    **query,
                    "SERVICE": "WMTS",
                    "REQUEST": "GetTile",
                    "VERSION": "1.0.0",
                    "LAYER": layer_id,
                    "STYLE": style,
                    "FORMAT": image_format,
                    "TILEMATRIXSET": matrix_set_id,
                    "TILEMATRIX": matrix_id,
                    "TILEROW": str(row),
                    "TILECOL": str(column),
                },
            )
            left = origin_x + column * tile_width * resolution
            top = origin_y - row * tile_height * resolution
            requests.append(
                _TileRequest(
                    url=url,
                    column=column - col0,
                    row=row - row0,
                    bounds_3857=(left, top - tile_height * resolution, left + tile_width * resolution, top),
                )
            )
    return _TileGrid(
        requests=tuple(requests),
        transform=Affine(
            resolution,
            0,
            origin_x + col0 * tile_width * resolution,
            0,
            -resolution,
            origin_y - row0 * tile_height * resolution,
        ),
        width=max(0, col1 - col0 + 1) * tile_width,
        height=max(0, row1 - row0 + 1) * tile_height,
        tile_size=tile_width,
        source_id=source_id,
    )


def _download_grid(
    grid: _TileGrid,
    settings: dict[str, Any],
    output_path: Path,
    *,
    workers: int,
) -> tuple[list[BaseGeometry], int]:
    if not grid.requests or grid.width <= 0 or grid.height <= 0:
        return [], 0
    headers, query = _authorization(settings)
    timeout = float(settings.get("request_timeout_seconds") or 60)
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 4,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": grid.transform,
        "tiled": True,
        "blockxsize": min(256, grid.tile_size),
        "blockysize": min(256, grid.tile_size),
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    downloaded: list[BaseGeometry] = []
    missing = 0

    def fetch(request: _TileRequest) -> tuple[_TileRequest, np.ndarray | None]:
        try:
            content = _request_bytes(
                _append_query(request.url, query),
                headers=headers,
                timeout=timeout,
                retries=4,
                missing_is_empty=True,
            )
        except ExternalImageryError:
            raise
        if not content:
            return request, None
        try:
            with Image.open(BytesIO(content)) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        except (OSError, ValueError):
            return request, None
        return request, np.moveaxis(rgba, 2, 0)

    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="imagery-http") as executor:
            futures = [executor.submit(fetch, request) for request in grid.requests]
            for future in as_completed(futures):
                request, rgba = future.result()
                if rgba is None:
                    missing += 1
                    continue
                height = min(grid.tile_size, rgba.shape[1])
                width = min(grid.tile_size, rgba.shape[2])
                dataset.write(
                    rgba[:, :height, :width],
                    window=Window(
                        request.column * grid.tile_size,
                        request.row * grid.tile_size,
                        width,
                        height,
                    ),
                )
                downloaded.append(box(*request.bounds_3857))
    return downloaded, missing


def _authorization(settings: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    auth = settings.get("auth")
    if not isinstance(auth, dict):
        auth = {"type": "none"}
    auth_type = str(auth.get("type") or "none").lower()
    headers = {"User-Agent": "MLSystem2/0.6"}
    query: dict[str, str] = {}
    if auth_type == "none":
        return headers, query
    if auth_type in {"query_api_key", "header_api_key", "bearer"}:
        value = _required_env(auth.get("env"), "ключ внешнего источника")
        if auth_type == "query_api_key":
            query[str(auth.get("name") or "api_key")] = value
        elif auth_type == "header_api_key":
            headers[str(auth.get("name") or "X-API-Key")] = value
        else:
            headers["Authorization"] = f"Bearer {value}"
        return headers, query
    if auth_type == "oauth2_client_credentials":
        client_id = _required_env(auth.get("client_id_env"), "OAuth2 client id")
        client_secret = _required_env(auth.get("client_secret_env"), "OAuth2 client secret")
        token_url = os.path.expandvars(str(auth.get("token_url") or ""))
        body = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if auth.get("scope"):
            body["scope"] = str(auth["scope"])
        token_payload = json.loads(
            _request_bytes(
                token_url,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
                data=urlencode(body).encode("utf-8"),
            ).decode("utf-8")
        )
        token = str(token_payload.get("access_token") or "")
        if not token:
            raise ExternalImageryError("AUTH_FAILED", "OAuth2-сервис не вернул access_token.")
        headers["Authorization"] = f"Bearer {token}"
        return headers, query
    raise ExternalImageryError("AUTH_INVALID", f"Неизвестный тип авторизации: {auth_type}.")


def _request_bytes(
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    retries: int = 3,
    missing_is_empty: bool = False,
    data: bytes | None = None,
) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=headers, data=data), timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404 and missing_is_empty:
                return b""
            if exc.code == 429 or 500 <= exc.code < 600:
                retry_after = exc.headers.get("Retry-After")
                delay = _retry_delay(retry_after, attempt)
                if attempt + 1 < retries:
                    time.sleep(delay)
                    continue
            raise ExternalImageryError(
                "EXTERNAL_HTTP_ERROR",
                f"Внешний сервис вернул HTTP {exc.code}.",
            ) from exc
        except (TimeoutError, URLError) as exc:
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
                continue
            raise ExternalImageryError(
                "EXTERNAL_HTTP_ERROR",
                "Внешний сервис не ответил в установленное время.",
            ) from exc
    return b""


def _expanded_aoi_3857(
    aoi_wgs84: BaseGeometry,
    target_resolution: float,
    settings: dict[str, Any],
) -> BaseGeometry:
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    geometry = shapely_transform(to_3857.transform, aoi_wgs84)
    context_pixels = int(settings.get("context_pixels") or 3072)
    return geometry.buffer(max(target_resolution * context_pixels, target_resolution))


def _coverage_percent(aoi_wgs84: BaseGeometry, bounds_3857: list[BaseGeometry]) -> float:
    if not bounds_3857:
        return 0.0
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    coverage = shapely_transform(to_wgs84.transform, unary_union(bounds_3857))
    if aoi_wgs84.area <= 0:
        return 0.0
    return round(min(100.0, max(0.0, coverage.intersection(aoi_wgs84).area / aoi_wgs84.area * 100)), 6)


def _ensure_disk_space(root: Path, grid: _TileGrid) -> None:
    estimated = max(1, grid.width) * max(1, grid.height) * 4
    free = shutil.disk_usage(root).free
    if free < estimated * 1.2:
        raise ExternalImageryError(
            "INSUFFICIENT_DISK_SPACE",
            f"Для внешних тайлов требуется примерно {estimated / 1024**3:.2f} ГиБ свободного места.",
        )


def _wmts_named_element(root: ET.Element, local_name: str, identifier: str) -> ET.Element | None:
    matches = [element for element in root.iter() if _local_name(element.tag) == local_name]
    if not identifier:
        return matches[0] if matches else None
    for element in matches:
        if _descendant_text(element, "Identifier") == identifier:
            return element
    return None


def _children(element: ET.Element, local_name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == local_name]


def _descendant_text(element: ET.Element, local_name: str) -> str | None:
    for child in element.iter():
        if _local_name(child.tag) == local_name and child.text:
            return child.text.strip()
    return None


def _float_text(element: ET.Element, local_name: str) -> float | None:
    value = _descendant_text(element, local_name)
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _append_query(url: str, values: dict[str, Any]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({str(key): str(value) for key, value in values.items() if value is not None})
    return urlunparse(parsed._replace(query=urlencode(query)))


def _required_env(name: object, description: str) -> str:
    env_name = str(name or "").strip()
    value = os.getenv(env_name, "") if env_name else ""
    if not value:
        raise ExternalImageryError(
            "AUTH_FAILED",
            f"Не задана переменная окружения для {description}: {env_name or '(не указана)' }.",
        )
    return value


def _retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return min(60.0, max(0.0, float(value)))
        except ValueError:
            pass
    return min(8.0, 0.5 * (2**attempt))


def _datetime_score(value: object) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)[:120]


__all__ = ["ExternalImageryError", "ExternalImageryResult", "prepare_external_imagery"]

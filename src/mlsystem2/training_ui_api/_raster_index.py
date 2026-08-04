"""Восстанавливаемый индекс метаданных исходных растров."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.warp import transform_bounds


INDEX_VERSION = 1


@dataclass(frozen=True)
class RasterIndexEntry:
    """Метаданные TIFF, достаточные для пространственного отбора без открытия файла."""

    relative_path: str
    size_bytes: int
    mtime_ns: int
    crs: str
    bounds: tuple[float, float, float, float]
    wgs84_bounds: tuple[float, float, float, float]
    band_count: int
    x_resolution: float
    y_resolution: float

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> RasterIndexEntry:
        return cls(
            relative_path=str(payload["relative_path"]),
            size_bytes=int(payload["size_bytes"]),
            mtime_ns=int(payload["mtime_ns"]),
            crs=str(payload["crs"]),
            bounds=tuple(float(value) for value in payload["bounds"]),
            wgs84_bounds=tuple(float(value) for value in payload["wgs84_bounds"]),
            band_count=int(payload["band_count"]),
            x_resolution=abs(float(payload["x_resolution"])),
            y_resolution=abs(float(payload["y_resolution"])),
        )


@dataclass(frozen=True)
class RasterIndex:
    entries: tuple[RasterIndexEntry, ...]
    warnings: tuple[str, ...]
    cache_state: str
    refreshed_files: int


def load_raster_index(
    images_root: Path,
    *,
    cache_path: Path | None = None,
    workers: int = 8,
) -> RasterIndex:
    """Обновить индекс по path/size/mtime и вернуть детерминированный снимок."""

    root = images_root.resolve()
    source_paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    cached = _read_cache(cache_path, root)
    current: dict[str, tuple[Path, int, int]] = {}
    for path in source_paths:
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        current[relative] = (path, int(stat.st_size), int(stat.st_mtime_ns))

    entries: dict[str, RasterIndexEntry] = {}
    refresh: list[tuple[str, Path, int, int]] = []
    for relative, (path, size_bytes, mtime_ns) in current.items():
        previous = cached.get(relative)
        if (
            previous is not None
            and previous.size_bytes == size_bytes
            and previous.mtime_ns == mtime_ns
        ):
            entries[relative] = previous
        else:
            refresh.append((relative, path, size_bytes, mtime_ns))

    warnings: list[str] = []
    if refresh:
        with ThreadPoolExecutor(max_workers=max(1, min(int(workers), len(refresh)))) as pool:
            for relative, entry, error in pool.map(_read_entry, refresh):
                if entry is not None:
                    entries[relative] = entry
                elif error:
                    warnings.append(f"Пропущен нечитаемый снимок {relative}: {error}.")

    ordered = tuple(entries[key] for key in sorted(entries, key=str.casefold))
    changed = bool(refresh) or set(cached) != set(current)
    cache_state = "rebuilt" if not cached else ("updated" if changed else "hit")
    if cache_path is not None and changed:
        _write_cache(cache_path, root, ordered)
    return RasterIndex(
        entries=ordered,
        warnings=tuple(warnings),
        cache_state=cache_state,
        refreshed_files=len(refresh),
    )


def projected_resolution_m(entry: RasterIndexEntry) -> float | None:
    """Оценить размер пикселя в метрах в Web Mercator."""

    try:
        left, bottom, right, top = transform_bounds(
            entry.crs,
            "EPSG:3857",
            *entry.bounds,
            densify_pts=21,
        )
        source_width = abs(entry.bounds[2] - entry.bounds[0]) / entry.x_resolution
        source_height = abs(entry.bounds[3] - entry.bounds[1]) / entry.y_resolution
        if source_width <= 0 or source_height <= 0:
            return None
        x_resolution = abs(right - left) / source_width
        y_resolution = abs(top - bottom) / source_height
        value = math.sqrt(x_resolution * y_resolution)
    except Exception:  # noqa: BLE001
        return None
    return value if math.isfinite(value) and value > 0 else None


def _read_entry(
    item: tuple[str, Path, int, int],
) -> tuple[str, RasterIndexEntry | None, str | None]:
    relative, path, size_bytes, mtime_ns = item
    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError("CRS не задана")
            bounds = tuple(float(value) for value in dataset.bounds)
            wgs84_bounds = tuple(
                float(value)
                for value in transform_bounds(
                    dataset.crs,
                    "EPSG:4326",
                    *dataset.bounds,
                    densify_pts=21,
                )
            )
            entry = RasterIndexEntry(
                relative_path=relative,
                size_bytes=size_bytes,
                mtime_ns=mtime_ns,
                crs=dataset.crs.to_wkt(),
                bounds=bounds,
                wgs84_bounds=wgs84_bounds,
                band_count=int(dataset.count),
                x_resolution=abs(float(dataset.res[0])),
                y_resolution=abs(float(dataset.res[1])),
            )
        return relative, entry, None
    except Exception as exc:  # noqa: BLE001
        return relative, None, str(exc)


def _read_cache(cache_path: Path | None, root: Path) -> dict[str, RasterIndexEntry]:
    if cache_path is None or not cache_path.is_file():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("version") != INDEX_VERSION or payload.get("images_root") != str(root):
            return {}
        result = {
            entry.relative_path: entry
            for entry in (RasterIndexEntry.from_json(item) for item in payload.get("entries", []))
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}
    return result


def _write_cache(cache_path: Path, root: Path, entries: tuple[RasterIndexEntry, ...]) -> None:
    payload = {
        "version": INDEX_VERSION,
        "images_root": str(root),
        "entries": [asdict(entry) for entry in entries],
    }
    temporary = cache_path.with_name(f".{cache_path.name}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache_path)
    except OSError:
        temporary.unlink(missing_ok=True)


__all__ = [
    "RasterIndex",
    "RasterIndexEntry",
    "load_raster_index",
    "projected_resolution_m",
]

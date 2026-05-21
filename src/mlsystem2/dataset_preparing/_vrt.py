"""Сборка VRT XML в памяти."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ._constants import TARGET_CRS
from ._raster_validation import RasterInfo


def build_vrt_xml(rasters: list[RasterInfo]) -> str:
    if not rasters:
        raise ValueError("Для построения VRT нужен хотя бы один снимок")

    gdalbuildvrt = _find_gdalbuildvrt()
    if gdalbuildvrt is None:
        raise RuntimeError("gdalbuildvrt не найден")

    with tempfile.TemporaryDirectory(prefix="mlsystem2_vrt_") as temp_dir:
        temp_path = Path(temp_dir)
        input_list_path = temp_path / "inputs.txt"
        output_path = temp_path / "mosaic.vrt"
        input_list_path.write_text(
            "\n".join(raster.path.resolve().as_posix() for raster in rasters) + "\n",
            encoding="utf-8",
        )
        command = [
            gdalbuildvrt,
            "-resolution",
            "highest",
            "-overwrite",
            "-input_file_list",
            input_list_path.as_posix(),
            output_path.as_posix(),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"gdalbuildvrt завершился с ошибкой: {message}")
        return output_path.read_text(encoding="utf-8")


def _find_gdalbuildvrt() -> str | None:
    executable = shutil.which("gdalbuildvrt")
    if executable is not None:
        return executable

    for candidate in (
        Path(r"C:\Program Files\QGIS 3.44.10\bin\gdalbuildvrt.exe"),
        Path(r"C:\Program Files\QGIS 3.42.0\bin\gdalbuildvrt.exe"),
        Path(r"C:\Program Files\QGIS 3.40.0\bin\gdalbuildvrt.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None

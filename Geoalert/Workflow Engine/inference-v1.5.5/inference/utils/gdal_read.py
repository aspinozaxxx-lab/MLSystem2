from pathlib import Path
from typing import Iterable, Optional
from osgeo import gdal
from rasterio.windows import Window
from we_queue_client.utils import log_time


@log_time(level="TRACE")
def read_part_from_minio_gdal(src_path: Path,
                              dst_path: Path,
                              window: Optional[Window]):
    """
    Read part of the raster directly from minio and write to the local path

    Use osgeo.gdal module as it is more robust than rasterio
    """
    gdal.UseExceptions()
    if window:
        gdal.Translate(str(dst_path), str(src_path), srcWin=window.flatten())
    else:
        gdal.Translate(str(dst_path), str(src_path))


def merge_files(src_paths: Iterable[Path],
                dst_path: Path):
    gdal.UseExceptions()
    vrt_options = gdal.BuildVRTOptions(resampleAlg='Bilinear')
    vrt_path = dst_path.with_suffix('.vrt')
    gdal.BuildVRT(str(vrt_path), [str(path) for path in src_paths], options=vrt_options)
    gdal.Translate(str(dst_path), str(vrt_path))
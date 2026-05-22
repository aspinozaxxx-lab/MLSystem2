import os

import rasterio
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio import windows as rio_windows
from rasterio.windows import Window
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry.base import BaseGeometry
from shapely.geometry import box, mapping
from loguru import logger
from pathlib import Path
import numpy as np

from ..errors import InvalidAOI

# If xy-oriented window of AOI covers fraction of image less than this threshold, read from s3 is windowed;
# otherwise file is copied as a whole
COPY_CUT_THRESHOLD = 0.7


def aoi_bbox_percent_cover(src: rasterio.DatasetReader,
                           aoi: BaseGeometry,
                           aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    Returns fraction of SRC image covered with the XY-axes oriented (in image coordinate system) bbox of the AOI.
    """
    vector_bounds = aoi.envelope
    bounds = rasterio.warp.transform_bounds(src.crs, aoi_crs, *src.bounds)
    raster_bounds = box(*bounds)

    fraction = (raster_bounds.intersection(vector_bounds).area/raster_bounds.area)
    logger.debug(f"AOI bbox covers {fraction} of image")
    return fraction


def aoi_fraction_cover(src: rasterio.DatasetReader,
                       aoi: BaseGeometry,
                       aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    Returns fraction of SRC image covered with the AOI.
    """
    bounds = rasterio.warp.transform_bounds(src.crs, aoi_crs, *src.bounds)
    raster_bounds = box(*bounds)

    fraction = (raster_bounds.intersection(aoi).area/raster_bounds.area)
    logger.debug(f"AOI covers {fraction} of image")

    return fraction

def should_read_window(src: rasterio.DatasetReader,
                       aoi: BaseGeometry,
                       aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    We want to read window with GDAL directly from s3 if the window (aoi bbox) is relatively small.
    It will save us some time as we do not read and copy excess information

    Although, if the window is big, it is faster to copy file with boto3 and handle it locally
    args:
        src_file: path to source raster
        aoi: shapely object with aoi
        use_aoi_bounds: if True, we compare the raster bounds with AOI bounds,
         to not cut if the AOI is small by area, but covers all the image, like diagonal line.
         If False, the AOI itself is used and the image is cut if the IOU (image, AOI) < threshold

    """
    if aoi_fraction_cover(src=src, aoi=aoi, aoi_crs=aoi_crs) <= 0:
        raise InvalidAOI(aoi=aoi, reason="AOI does not intersect the image")
    return aoi_bbox_percent_cover(src=src, aoi=aoi, aoi_crs=aoi_crs) < COPY_CUT_THRESHOLD

def file_too_big_for_gdal(window: Window, dtype):
    """
    Cache size should be 1.5 times bigger than the uncompressed band memory size
    """
    width = window.width
    height = window.height
    res = width*height*np.dtype(dtype).itemsize*1.5 > int(os.getenv("VSI_CACHE_SIZE"))
    if res:
        logger.debug(f"Window : {window}. Too big file for GDAL")
    return res

def get_window(src: rasterio.DatasetReader,
               aoi: BaseGeometry,
               aoi_crs: CRS = CRS.from_epsg(4326)):
    vector_bounds = rasterio.warp.transform_bounds(aoi_crs, src.crs, *aoi.bounds)
    window = rio_windows.from_bounds(*vector_bounds,
                                     transform=src.transform)
    window = window.intersection(rio_windows.Window(0, 0, src.width, src.height))
    window = rio_windows.Window.from_slices(*rio_windows.window_index(window))
    return window

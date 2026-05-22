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

# If fraction of file covered by AOI is less than this threshold, image is masked by AOI
# otherwise file is processed without changes
INFERENCE_CUT_THRESHOLD = 0.7
# If xy-oriented window of AOI covers fraction of image less than this threshold, read from s3 is windowed;
# otherwise file is copied as a whole
COPY_CUT_THRESHOLD = 0.7


def mask_dataset_by_aoi(src: rasterio.io.DatasetWriter,
                        aoi: BaseGeometry,
                        aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    Adds per-dataset nodata bitmask to the image to mark which pixels should not be processed

    Initial in-dataset bitmask is preserved (output mask is multiplication of AOI and initial mask)
    If dst_nodata is not None, out-of-mask pixels are reassigned with nodata value
    """
    initial_mask = src.read_masks(1)
    mask_geom = transform_geom(aoi_crs, src.crs, mapping(aoi))
    mask = geometry_mask([mask_geom], (src.height, src.width), transform=src.transform, invert=True).astype("uint8") * initial_mask
    src.write_mask(mask)


def mask_local_raster_by_aoi(filepath: Path,
                             aoi: BaseGeometry,
                             aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    Modify local file in place: cut to AOI
    """
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(filepath, 'r+', IGNORE_COG_LAYOUT_BREAK='YES') as src:
            if aoi and should_cut_file(src=src, aoi=aoi):
                logger.debug(f"Cutting file {filepath} to aoi")
                mask_dataset_by_aoi(src=src, aoi=aoi, aoi_crs=aoi_crs)
            else:
                logger.debug(f"No need to cut the file {filepath}")
                return


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


def should_cut_file(src: rasterio.DatasetReader,
                    aoi: BaseGeometry,
                    aoi_crs: CRS = CRS.from_epsg(4326)):
    """
    If we have copied the file as a whole, we can either process it as a whole, or cut excess
     to zero the data outside of AOI and NOT process it with the model

     We should do it if the aoi is relatively small and skip this step otherwise
    args:
        src_file: path to source raster
        aoi: shapely object with aoi
        use_aoi_bounds: if True, we compare the raster bounds with AOI bounds,
         to not cut if the AOI is small by area, but covers all the image, like diagonal line.
         If False, the AOI itself is used and the image is cut if the IOU (image, AOI) < threshold

    """
    if not aoi:
        return False
    return aoi_fraction_cover(src=src, aoi=aoi, aoi_crs=aoi_crs) < INFERENCE_CUT_THRESHOLD


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

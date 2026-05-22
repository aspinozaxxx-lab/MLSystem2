import os
import shutil
from pathlib import Path
from typing import Tuple, Union

import rasterio
import numpy as np
import hashlib

from fastapi import UploadFile
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from shapely.geometry import box

from PIL import Image

# todo: make it env?
STD_WIDTH = 3


def get_footprint(filename: str):
    """
    Footprint is a geometry that covers the whole image. This function does not takes nodata mask into account,
    and just generates a rectangle form image bounds
    Args:
        filename: path-like object, which can be accessed with rastreio.open
    Returns:
        Bounding box as shapely Polygon in lat-lon coordinates
    """
    with rasterio.open(filename) as src:
        bounds = transform_bounds(src.crs, CRS.from_epsg(4326), *src.bounds)
    return box(*bounds)


def get_sha1_checksum(file_path: Union[str, Path]):
    """
    returns string representation (digest) of md5 hash of file. Reads entire file into memory, as we do this anyway in other places
    """
    with open(file_path, "rb") as f:
        sha1_hash = hashlib.sha1(f.read())
    return sha1_hash.hexdigest()


def get_raster_metadata(file_path: Union[str, Path]):
    # this metadata should be saved in the database as a json/dict
    with rasterio.open(file_path) as src:
        meta = {
            'dtypes': src.dtypes,
            'width': src.width,
            'height': src.height,
            "nodata": src.nodata,
            "count": src.count,
            "crs": str(src.crs),
            "pixel_size": src.res
        }
    return meta


def get_file_description(file_path, filename: Union[str, Path]) -> Tuple[str, str, dict]:
    """
    Returns tuple: image filename (str), checksum (str), raster metadata (dict)
    """
    return filename, get_sha1_checksum(file_path=file_path), get_raster_metadata(file_path=file_path)


def preview_size(original_height: int, original_width: int, preview_size_limit: int):
    """ Calculates preview size preserving the ratio, with maximum size of preview_size_limit """
    if original_width <= preview_size_limit and original_height <= preview_size_limit:
        return original_height, original_width
    elif original_width > original_height:
        return int(preview_size_limit*original_height/original_width), preview_size_limit
    else:
        return preview_size_limit, int(preview_size_limit*original_width/original_height)


def to8bit(image):
    # mean +- 3*std color stretch to 0-255
    # image shape must be [c, h, w]
    # we assume that the image can be not-rgb, but width and height must be less than channels num
    assert image.ndim == 3 and image.shape[0] in [1, 3]
    channels_8bit = []
    for channel in image:
        mean, std, min_val, max_val = np.mean(channel), np.std(channel), np.min(channel), np.max(channel)
        lower = max(min_val, mean - STD_WIDTH*std)
        upper = min(max_val, mean + STD_WIDTH*std)
        ch_8bit = np.floor_divide(
            np.multiply((channel - lower), 255, dtype='float32'),
            (upper-lower)
        )
        # We clip it from 1 to leave 0 value for nodata
        ch_8bit = np.clip(np.around(ch_8bit, 0), 1, 255).astype('uint8')
        channels_8bit.append(ch_8bit)

    return np.stack(channels_8bit)


def generate_preview(input_file: Union[str, Path], preview_file: Union[str, Path], size: int):
    """
    args:
        input_file: local path to file, probably tempfile
        preview_file: local path to preview file to save data
    """
    # file_extension = os.path.splitext(filename)[1]
    input_file = Path(input_file)
    preview_file = Path(preview_file)

    with rasterio.open(input_file) as src:
        out_shape = preview_size(src.height, src.width, size)
        if src.count < 3:
            # if there are 1 or 2 channels, just read the first of them and save grayscale preview
            channels = [1]
        else:
            # read first 3 channels as RGB.
            # todo: add reading according to colorinterp or something else interesting
            channels = [1, 2, 3]
        preview = src.read(channels,
                           out_shape=(len(channels), *out_shape),
                           resampling=Resampling.bilinear)

        if src.dtypes[0] != 'uint8':
            preview = to8bit(preview)
        if preview.shape[0] == 1:
            preview = preview.squeeze(0)
        elif preview.shape[0] == 3:
            preview = preview.transpose(1, 2, 0)
        else:
            raise AssertionError('This should not happen! We read either 1 or 3 channels')
    Image.fromarray(preview).save(preview_file)


def save_upload_file_tmp(upload_file: UploadFile, file_path: Union[str, Path]):
    shutil.copyfileobj(upload_file.file, open(file_path, 'wb'))


def get_file_size(file_path: Union[str, Path]) -> int:
    return os.path.getsize(file_path)


def make_cog(filename, channels):
    # reproject to 3857
    # and resample to upper zoom
    # save as COG
    rasterio.warp.reproject()
    with rasterio.open(filename, 'r+') as src:
        src.driver = 'COG',
        src.compress ='ZSTD'
import rasterio
from loguru import logger
from typing import Tuple
from urllib.parse import urlparse
from ..errors.local import ImageMustBeTiff

def is_tiff(path: str):
    lower_name = path.lower()
    return lower_name.endswith('.tiff') or lower_name.endswith('.tif')


def is_folder(path: str):
    return path.endswith('/')


def read_profile_from_s3(storage, s3_url):
    """
    Read profile from s3 with rasterio
    Args:
        storage: queueClient Storage
        s3_url: url loooking like s3://bucket/path. It may point either to tiff file, or to directory of tif files
    Returns:
        rasterio profile object (dict) with file metadata
    """
    if not is_tiff(s3_url):
        raise ImageMustBeTiff(s3_url)
    with storage.open(s3_url) as src:
        profile = src.profile
    return profile
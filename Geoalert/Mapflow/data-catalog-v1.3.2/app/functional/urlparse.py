import os
from typing import Tuple
from urllib.parse import urlparse


def parse_image_url(image_url: str) -> Tuple[str, str, str]:
    """
    Parse filename, minio_bucket and minio_object from given url.
    url must be passed with scheme:
        e.g. in this form:  s3://mapflow-rasters/9764750d-6047-407e-a972-5ebd6844be8a/raster.tif
        not in this form: mapflow-rasters/9764750d-6047-407e-a972-5ebd6844be8a/raster.tif
    If url passed without scheme, urllib library will handle it as a path
    :return: (filename, minio_bucket, minio_object)
    """
    parsed_url = urlparse(image_url)
    minio_bucket_from_url = parsed_url.netloc
    minio_object_from_url = parsed_url.path[1:]
    filename = os.path.basename(image_url)
    return filename, minio_bucket_from_url, minio_object_from_url

import rasterio
from loguru import logger
from typing import Optional
from pathlib import Path
from contextlib import contextmanager
from shapely.geometry.base import BaseGeometry
from rasterio.session import AWSSession
from we_queue_client import Storage, InternalError
from rasterio.errors import WindowError
from data_validator_lib.functional.read_from_s3 import is_tiff


def is_subfolder(path: str, prefix: str):
    """
    True if path contains any subfolder delimeter after specified prefix.
    `/` in the beginning is stripped for cases when prefix does not contain trailing `/`
    """
    if not path.startswith(prefix):
        raise ValueError("Invalid prefix")
    postfix = path[len(prefix):].lstrip('/')
    return '/' in postfix


class SourceValidatorStorage(Storage):
    def __init__(self,
                 minio_url: str,
                 minio_access_key: , minio_secret_key: ,
                 aws_https: str = 'NO'):
        """
        Based on queue_client Storage

        Additional funcitonality: allows to read file data partially, without copying full file to the worker,
        and supports fallback to full copy + cut by AOI in case partial read fails

        todo: reuse queue-validator storage after it will be patched with GDAL support
        """

        super().__init__(minio_url=minio_url,
                         minio_access_key=,
                         minio_secret_key=

        self.endpoint = minio_url
        self.session = AWSSession(aws_access_key_id=,
                                  aws_secret_access_key=
        self.aws_https = aws_https

    @contextmanager
    def open(self, path):
        """
        contextmanager - wrapper around opening file at s3 with rasterio
        """
        try:
            with rasterio.env.Env(session=self.session,
                                  AWS_HTTPS=self.aws_https,
                                  GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                                  AWS_VIRTUAL_HOSTING=False,
                                  AWS_S3_ENDPOINT=self.endpoint):
                with rasterio.open(path) as src:
                    yield src
        finally:
            # returned resource is a contextmanager as well, so no need to teardown here
            pass


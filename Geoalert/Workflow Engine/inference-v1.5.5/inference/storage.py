import rasterio
from loguru import logger
from typing import Optional
from pathlib import Path
from contextlib import contextmanager
from shapely.geometry.base import BaseGeometry
from rasterio.session import AWSSession
from rasterio.errors import RasterioIOError
from we_queue_client import Storage, InternalError
from we_queue_client.utils import log_time
from rasterio.errors import WindowError

from .utils import (mask_local_raster_by_aoi,
                    get_window,
                    read_part_from_minio_gdal,
                    file_too_big_for_gdal,
                    should_read_window,
                    merge_files)
from .errors import InvalidAOI

from .message import InferenceArtifact


class InferenceStorage(Storage):
    def __init__(self,
                 minio_url: str,
                 minio_access_key: , minio_secret_key: ,
                 aws_https: str = 'NO',
                 vsi_cache_size=25_000_000):
        """
        Based on queue_client Storage

        Additional funcitonality: allows to read file data partially, without copying full file to the worker,
        and supports fallback to full copy + cut by AOI in case partial read fails

        TODO: big part of raster artifact functionality doubles the functionality of cog-builder. Will move it to
              queue-client as an optional feature
        """

        super().__init__(minio_url=minio_url,
                         minio_access_key=,
                         minio_secret_key=

        self.endpoint = minio_url
        self.session = AWSSession(aws_access_key_id=,
                                  aws_secret_access_key=
        self.aws_https = aws_https
        self.vsi_cache_size = vsi_cache_size

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
                                  AWS_S3_ENDPOINT=self.endpoint,
                                  VSI_CACHE=True,
                                  VSI_CACHE_SIZE=self.vsi_cache_size):
                with rasterio.open(path) as src:
                    yield src
        finally:
            # returned resouce is a contextmanager as well, so no need to teardown here
            pass

    @log_time(level="DEBUG", log_args=False, log_kwargs=False)
    def get_artifact(self,
                     artifact: InferenceArtifact,
                     workdir: Path,
                     aoi: Optional[BaseGeometry] = None):
        """
        Downloads the specified Artifact to workdir and cuts it by AOI if applicable
        """
        if artifact.is_raster and not artifact.is_folder \
                and aoi is not None:
            self.get_raster_artifact(artifact=artifact,
                                     workdir=workdir,
                                     aoi=aoi)
        elif artifact.is_raster and artifact.is_folder \
                and aoi is not None:
            self.get_folder_raster_artifact(artifact=artifact,
                                            workdir=workdir,
                                            aoi=aoi)
        elif not artifact.is_folder:
            # non-raster artifacts, or raster artifacts without aoi
            self.download(workdir=workdir,
                          artifact=artifact)
        else:
            raise InternalError("Cannot download non-raster artifact from folder")

    def get_raster_artifact(self,
                            artifact: InferenceArtifact,
                            workdir: Path,
                            aoi: BaseGeometry,
                            mask=True):
        """
        Tries to read the file directly from Minio (better way),
        in case of fail copies the whole file and cuts it by AOI locally
        """

        # gather info about the file at s3
        try:
            with self.open(artifact.path) as src:
                window = get_window(src, aoi)
                dtype = src.dtypes[0]
                # raises AOIError if file does not intersect with AOI
                read_window = should_read_window(src=src, aoi=aoi)
        except RasterioIOError as e:
            logger.opt(exception=True).warning(f"Could not get window with rasterio from minio: {e}. "
                                               f"Fallback to full copy.")
            read_window = False
            dtype = None
        if not read_window or file_too_big_for_gdal(window=window, dtype=dtype):
            # window is not returned if we do not need to cut image
            self.download(artifact=artifact, workdir=workdir)
        else:
            try:
                log_time(level="DEBUG",
                         log_kwargs=True)(read_part_from_minio_gdal)(src_path=artifact.gdal_path(),
                                                                     dst_path=workdir/artifact.name,
                                                                     window=window)
            except Exception as e:
                logger.opt(exception=True).warning("Could not read file with GDAL from minio. Fallback to full copy.")
                self.download(artifact=artifact, workdir=workdir)

        if mask:
            mask_local_raster_by_aoi(filepath=workdir/artifact.name, aoi=aoi)


    def get_folder_raster_artifact(self,
                                   artifact: InferenceArtifact,
                                   workdir: Path,
                                   aoi: Optional[BaseGeometry] = None):
        """
        Future implementation shdould be like:
            for file in intersecting_files:
            self.read_part_from_minio(artifact, file)
        """
        file_artfacts = self.list_artifacts(artifact=artifact, extensions=('tif', 'tiff'))
        downloaded_files = []
        for file in file_artfacts:
            try:
                self.get_raster_artifact(artifact=file,
                                         workdir=workdir,
                                         aoi=aoi,
                                         mask=False)
                downloaded_files.append(file)
            except (InvalidAOI, WindowError) as e:
                # Skip files!
                pass
        if not downloaded_files:
            raise InvalidAOI(aoi=aoi, reason="No files intersect with AOI")
        log_time(level='DEBUG',
                 log_kwargs=False,
                 log_args=False)(merge_files)([workdir/file.name for file in downloaded_files], workdir/artifact.name)
        log_time(level='DEBUG',
                 log_kwargs=False,
                 log_args=False)(mask_local_raster_by_aoi)(filepath=workdir / artifact.name, aoi=aoi)

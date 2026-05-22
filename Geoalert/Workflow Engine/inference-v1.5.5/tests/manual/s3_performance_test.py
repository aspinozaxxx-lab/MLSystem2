from loguru import logger
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from inference.message import InferenceArtifact
from inference.storage import InferenceStorage
from inference.utils import log_time, mask_local_raster_by_aoi, read_part_from_minio_gdal


class ProfilingStorage(InferenceStorage):

    def profile_get_artifact(self,
                             artifact: InferenceArtifact,
                             workdir: Path,
                             aoi: BaseGeometry,
                             window):
        logger.debug("Measuring time of raster artifact download")
        log_time(level='DEBUG',
                 log_args=False,
                 log_kwargs=False)(self.download)(workdir=workdir, artifact=artifact)
        log_time(level='DEBUG',
                 log_args=False,
                 log_kwargs=False)(mask_local_raster_by_aoi)(filepath=workdir / artifact.name,
                                                             aoi=aoi)

        (workdir / artifact.name).unlink()
        log_time(level='DEBUG',
                 log_args=False,
                 log_kwargs=False)(read_part_from_minio_gdal)(src_path=artifact.gdal_path(),
                                                              dst_path=workdir / artifact.name,
                                                              window=window)
        log_time(level='DEBUG',
                 log_args=False,
                 log_kwargs=False)(mask_local_raster_by_aoi)(filepath=workdir / artifact.name,
                                                             aoi=aoi)

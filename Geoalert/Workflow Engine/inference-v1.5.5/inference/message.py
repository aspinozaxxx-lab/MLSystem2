
import json
from typing import Optional, Sequence, Mapping, Any, Dict
from pydantic import BaseModel, validator, AnyUrl, validator
from we_queue_client import InputMessage, OutputMessage, Artifact
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from loguru import logger
from shapely.validation import explain_validity

from .errors import InvalidAOI

class InferenceArtifact(Artifact):

    @property
    def is_raster(self):
        # TODO [WE]: pass artifact type in the message
        lower_name = self.name.lower()
        return lower_name.endswith('.tiff') or lower_name.endswith('.tif')

    @property
    def is_folder(self):
        # TODO [WE]: pass artifact type in the message
        """
        If we want to download every file in the folder
        """
        return str(self.path).endswith('/')

    def gdal_path(self, streaming=True):
        """
        GDAL uses steaming protocol (single read data request) if the path starts with /vsis3_streaming/
        It is faster than multiple range-requests, but requires bigger cache
        """
        if not streaming:
            return self.path

        if self.path.startswith("/vsis3_streaming/"):
            return self.path
        elif self.path.startswith("/vsis3/"):
            return self.path.replace("/vsis3/", "/vsis3_streaming/", 1)
        elif self.path.startswith("s3://"):
            return self.path.replace("s3://", "/vsis3_streaming/", 1)
        else:
            raise ValueError("Unknown minio path prefix! Must be `s3://`. `/vsis3/` or `/vsis3_streaming/`")

class InferenceInputData(BaseModel):
    aoi: Optional[Mapping[str, Any]] = None
    pipeline: str
    source_data: Sequence[InferenceArtifact]
    blocks: Optional[Dict[str, bool]] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("aoi")
    def to_shape(cls, aoi) -> BaseGeometry:
        """
        Import from geojson-like mapping
        """
        try:
            aoi_shape = shape(aoi)
        except Exception as e:
            logger.exception("Could not create shapely object")
        else:
            if not aoi_shape.is_valid:
                # try to fix the AOI by buffer
                aoi_shape = aoi_shape.buffer(0)
                if not aoi_shape.is_valid:
                    raise InvalidAOI(aoi, reason=explain_validity(aoi_shape))
            return aoi_shape

    def __str__(self):
        # To print only necessary info
        return ','.join((str(self.path), self.name))


class InferenceOutputData(BaseModel):
    output_data: Sequence[InferenceArtifact]


class InferenceInputMessage(InputMessage):
    input: InferenceInputData
    output: InferenceOutputData


class InferenceOutputMessage(OutputMessage):
    pass
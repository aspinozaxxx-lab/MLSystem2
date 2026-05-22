from enum import Enum
from loguru import logger
from base64 import b64decode
from typing import Optional, Sequence, Mapping, Any, List
from pydantic import BaseModel, validator, AnyUrl, validator
from we_queue_client import InputMessage, OutputMessage, Artifact
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from ..functional.geometry import maybe_valid_geometry

class Compress(str, Enum):
    WEBP = "WEBP"
    JPEG = "JPEG"
    PNG = "PNG"
    LZW = "LZW"
    ZSTD = "ZSTD"
    NONE = "NONE"

class CogBuilderArtifact(Artifact):

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


class CogBuilderInputData(BaseModel):
    aoi: Optional[Mapping[str, Any]] = None
    channels: Optional[str] = None
    raster_source: str
    compress: Compress = Compress.WEBP
    std_width: Optional[float] = 3.

    class Config:
        arbitrary_types_allowed = True

    @validator("channels")
    def parse_channels_str(cls, channels_str) -> List[int]:
        return list(map(int, channels_str.split(",")))

    @validator("aoi")
    def validate_aoi(cls, aoi):
        if aoi is None:
            return None
        return maybe_valid_geometry(aoi)

    @property
    def input_artifact(self):
        return CogBuilderArtifact(path=self.raster_source, name="input.tif")


class CogBuilderOutputData(BaseModel):
    target_uri: str
    @property
    def output_artifact(self):
        return CogBuilderArtifact(path=self.target_uri, name="output.tif")

class CogBuilderInputMessage(InputMessage):
    input: CogBuilderInputData
    output: CogBuilderOutputData


class CogBuilderOutputMessage(OutputMessage):
    pass
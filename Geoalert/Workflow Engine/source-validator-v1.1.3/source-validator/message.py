import yaml
from loguru import logger
from base64 import b64decode
from typing import Optional, Sequence, Mapping, Any
from pydantic import BaseModel, validator, AnyUrl, validator
from we_queue_client import InputMessage, OutputMessage, Artifact
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from data_validator_lib.base.constants import AOI_KEY
from data_validator_lib.functional import is_tiff


class SourceValidatorInputData(BaseModel):
    aoi: Optional[Mapping[str, Any]] = None
    requirements: str
    request: Mapping[str, Any]

    class Config:
        arbitrary_types_allowed = True

    @validator("requirements")
    def decode_requirements(cls, requirements_encoded) -> Mapping[str, Any]:
        return yaml.safe_load(b64decode(requirements_encoded))

    @validator("request")
    def add_aoi(cls, request, values):
        aoi = values.get('aoi', None)
        if not aoi:
            logger.warning("AOI geometry should be provided in input message!")
        request.update({AOI_KEY: aoi})
        return request


class SourceValidatorInputMessage(InputMessage):
    input: SourceValidatorInputData


class SourceValidatorOutputMessage(OutputMessage):
    pass
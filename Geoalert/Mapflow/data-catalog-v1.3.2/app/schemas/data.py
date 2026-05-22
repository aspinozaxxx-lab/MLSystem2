from pydantic import BaseModel, validator
from pydantic.schema import UUID, datetime
from typing import Union, List, Any

from config import Config

SERVICE_ROOT_URL = Config.SERVICE_ROOT_URL


class PostDataParams(BaseModel):
    tags: Union[List[str], None] = None


class DataReturnSchema(BaseModel):
    # schema used as response model for get image/images requests
    id: UUID
    image_url: str
    preview_url_l: str
    preview_url_s: Union[str, None] = None
    uploaded_at: datetime
    file_size: int
    footprint: Any
    filename: str
    checksum: str
    meta_data: dict
    cog_link: Union[str, None]

    class Config:
        orm_mode = True

    @validator('preview_url_l')
    def _construct_large_preview_url(cls, v, values):
        return SERVICE_ROOT_URL + "/image/" + str(values['id']) + "/preview/l"

    @validator('preview_url_s')
    def _construct_small_preview_url(cls, v, values):
        return SERVICE_ROOT_URL + "/image/" + str(values['id']) + "/preview/s"

    @validator('footprint')
    def _convert_footprint_from_shapely_to_wkt(cls, v):
        return v.wkt


class DataReturnErrorSchema(BaseModel):
    message: str


class LinkImageSchema(BaseModel):
    url: str


class MemoryInfoOfUserSchema(BaseModel):
    memoryLimit: int
    memoryUsed: int
    memoryFree: int

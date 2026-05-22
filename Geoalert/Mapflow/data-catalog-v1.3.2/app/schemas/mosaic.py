import datetime
from pydantic import BaseModel
from pydantic.schema import UUID

from typing import Union, List


class MosaicBase(BaseModel):
    name: str
    tags: List[str]


class MosaicCreateRequestSchema(MosaicBase):
    pass


class MosaicCreateResponseSchema(BaseModel):
    id: UUID
    tags: Union[List[str], None]
    name: str
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class MosaicUpdateResponseSchema(MosaicCreateResponseSchema):
    pass


class MosaicUpdateRequestSchema(MosaicBase):
    pass


class MosaicReturnSchema(BaseModel):
    id: UUID
    rasterLayer: dict
    tags: Union[List[str], None]
    name: str
    created_at: datetime.datetime

    class Config:
        orm_mode = True


class MosaicReturnErrorSchema(BaseModel):
    message: str

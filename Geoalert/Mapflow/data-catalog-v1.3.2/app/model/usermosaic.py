import uuid

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.types import ARRAY

from sqlalchemy.dialects.postgresql import UUID

from model.base import Base


class UserMosaic(Base):
    __tablename__ = 'usermosaic'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    mosaic_id = Column(UUID(as_uuid=True), ForeignKey('mosaics.id'))

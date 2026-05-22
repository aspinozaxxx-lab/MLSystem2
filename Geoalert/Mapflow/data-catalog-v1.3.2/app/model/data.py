import uuid

from geoalchemy2 import Geometry
from sqlalchemy import Column, ForeignKey, Integer, String, TIMESTAMP
from sqlalchemy.orm import relationship

from sqlalchemy.dialects.postgresql import UUID, JSONB

from model.base import Base


class Data(Base):
    __tablename__ = 'images'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mosaic_id = Column(UUID(as_uuid=True), ForeignKey('mosaics.id'))
    image_url = Column(String)
    preview_url_l = Column(String)
    preview_url_s = Column(String)
    uploaded_at = Column(TIMESTAMP(timezone=True))
    footprint = Column(Geometry('Polygon'))
    filename = Column(String)
    checksum = Column(String)
    meta_data = Column(JSONB)
    file_size = Column(Integer)
    cog_link = Column(String)

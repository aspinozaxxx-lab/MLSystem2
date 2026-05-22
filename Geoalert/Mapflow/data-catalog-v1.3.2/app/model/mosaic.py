from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from model.base import Base


class Mosaic(Base):
    __tablename__ = 'mosaics'

    id = Column(UUID(as_uuid=True), primary_key=True)
    owner_id = Column(UUID(as_uuid=True))
    mosaic_url = Column(String)
    tags = Column(ARRAY(String))
    name = Column(String)
    created_at = Column(TIMESTAMP)
    cog_link = Column(String)

    # TODO Relationships
    # Mosaic is "parent" for users that have access to it (e.g. several user may use the same mosaic)
    # users_of = relationship("UserMosaic")

    # Mosaic is also "parent" for images (mosaic may consist of one or more images)
    # related_images = relationship("Data", backref='mosaic')

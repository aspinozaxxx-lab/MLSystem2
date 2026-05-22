from sqlalchemy import Boolean, Column, String, BigInteger
from sqlalchemy.orm import relationship

from sqlalchemy.dialects.postgresql import UUID

import uuid

from model.base import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    is_admin = Column(Boolean)
    login = Column(String)
    memory_used = Column(BigInteger)
    memory_limit = Column(BigInteger)

    # TODO: relationships
    # Relationships
    # User is "parent" for mosaic (e.g. one user may have several mosaics)
    # mosaics_of = relationship("UserMosaic")

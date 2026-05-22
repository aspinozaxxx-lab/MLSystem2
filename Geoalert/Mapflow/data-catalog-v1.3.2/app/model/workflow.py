import enum
from sqlalchemy import Column, Integer, Enum
from sqlalchemy.dialects.postgresql import UUID
from model.base import Base


class WorkflowStatus(enum.Enum):
    UNPROCESSED = 'UNPROCESSED'
    IN_PROGRESS = 'IN_PROGRESS'
    OK = 'OK'
    FAILED = 'FAILED'

    @classmethod
    def from_we_status(cls, we_status):
        if we_status == 'OK':
            return cls.OK
        elif we_status in ('FAILED', 'CANCELLED'):
            return cls.FAILED
        elif we_status in ('IN_PROGRESS', 'PAUSED'):
            return cls.IN_PROGRESS
        else:
            raise ValueError(f'Unknown WE status {we_status}')


class Workflow(Base):
    __tablename__ = 'workflow'

    image_id = Column(UUID(as_uuid=True), primary_key=True)
    we_id = Column(Integer)
    status = Column(Enum(WorkflowStatus), nullable=False)

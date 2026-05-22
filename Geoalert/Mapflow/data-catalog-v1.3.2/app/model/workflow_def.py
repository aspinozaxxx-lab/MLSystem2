from sqlalchemy import Column, Integer
from sqlalchemy.dialects.postgresql import TEXT

from model.base import Base


class WorkflowDef(Base):
    __tablename__ = 'workflow_def'

    id = Column(Integer, primary_key=True)
    yaml = Column(TEXT)

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.schema import MetaData

from config import Config

Base = declarative_base(metadata=MetaData(schema=Config.DB_SCHEMA))

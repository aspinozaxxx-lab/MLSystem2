from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Config

SQLALCHEMY_DATABASE_URL = Config.DB_STRING
dbschema=Config.DB_SCHEMA

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # not sure if will work for schema setting, if MetaData() does not work, will return to this method
    # from https://stackoverflow.com/questions/9298296/sqlalchemy-support-of-postgres-schemas
    # connect_args={'options': '-csearch_path={}'.format(dbschema)}
    # connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)

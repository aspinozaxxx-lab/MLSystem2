"""SQLAlchemy engine/session для training UI API."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import MetaData, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ._config import TrainingUIAPIConfig


class Base(DeclarativeBase):
    metadata = MetaData()


def configure_schema(schema: str | None) -> None:
    Base.metadata.schema = schema or None


def create_session_factory(config: TrainingUIAPIConfig) -> sessionmaker[Session]:
    if config.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    elif config.database_schema:
        connect_args = {"options": f"-csearch_path={config.database_schema},public"}
    else:
        connect_args = {}
    engine = create_engine(config.database_url, future=True, pool_pre_ping=True, connect_args=connect_args)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

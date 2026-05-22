import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# add your model's MetaData object here
# for 'autogenerate' support
from model.base import Base

# import models
from model.data import Data
from model.user import User
from model.mosaic import Mosaic
from model.usermosaic import UserMosaic
from model.workflow import Workflow
from model.workflow_def import WorkflowDef
from config import Config

target_metadata = Base.metadata

config = context.config
config.set_main_option('sqlalchemy.url',
                        Config.DB_STRING)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

import alembic.ddl.base as alembic_base

def format_table_name_with_schema(compiler, name, schema):
    from alembic.operations import schemaobj

    table = schemaobj.SchemaObjects().table(name, schema=schema)
    return compiler.preparer.format_table(table)

alembic_base.format_table_name = format_table_name_with_schema

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema=Config.DB_SCHEMA
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        conn = connection.execution_options(schema_translate_map={None: Config.DB_SCHEMA})
        context.configure(
            connection=conn,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema=Config.DB_SCHEMA
        )
        with context.begin_transaction():
            context.execute(f'SET search_path TO public,{Config.DB_SCHEMA}')
            conn.dialect.default_schema_name = Config.DB_SCHEMA
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

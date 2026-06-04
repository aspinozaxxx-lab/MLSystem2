from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

from mlsystem2.training_ui_api._database import Base, configure_schema
from mlsystem2.training_ui_api import _models  # noqa: F401


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _database_url() -> str:
    return os.getenv(
        "MLSYSTEM2_TRAINING_UI_DATABASE_URL",
        os.getenv(
            "TRAINING_UI_DATABASE_URL",
            config.get_main_option("sqlalchemy.url"),
        ),
    )


def _schema() -> str | None:
    value = os.getenv("MLSYSTEM2_TRAINING_UI_DATABASE_SCHEMA", "training_ui")
    return value or None


schema = _schema()
configure_schema(schema)
target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    del obj, compare_to
    if reflected and type_ == "table" and name == "alembic_version":
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema=schema,
        include_object=_include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    if schema:
        configuration["sqlalchemy.connect_args"] = {"options": f"-csearch_path={schema},public"}
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        if schema:
            quoted_schema = schema.replace('"', '""')
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{quoted_schema}"'))
            connection.commit()
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema=schema,
            include_object=_include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

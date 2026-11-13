"""Explicit, transactional Alembic migrations with adoption of the original schema."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, MetaData, Table, func, inspect, select

MIGRATIONS = Path(__file__).with_name("migrations")
INITIAL_REVISION = "001_parent_storage"
LEGACY_MIGRATION = "001_parent_storage.sql"
LEGACY_CHECKSUM = "0e165d5d007d94b19febf0e932d10c5c214296de47eab126c4dda50689ce28c9"


def migrate(engine: Engine, directory: Path = MIGRATIONS) -> list[str]:
    config = Config()
    config.set_main_option("script_location", str(directory.resolve()).replace("%", "%%"))
    scripts = ScriptDirectory.from_config(config)
    with engine.begin() as connection:
        # Serialize migration commands; ingestion still runs with the API stopped.
        connection.execute(select(func.pg_advisory_xact_lock(726403101)))
        config.attributes["connection"] = connection
        current = MigrationContext.configure(connection).get_current_revision()
        if current is None and _has_legacy_schema(connection):
            command.stamp(config, INITIAL_REVISION)
            current = INITIAL_REVISION
        pending = list(scripts.iterate_revisions("head", current))
        command.upgrade(config, "head")
        return [revision.revision for revision in reversed(pending)]


def _has_legacy_schema(connection: Connection) -> bool:
    inspector = inspect(connection)
    if not inspector.has_table("schema_migrations"):
        return False
    legacy = Table("schema_migrations", MetaData(), autoload_with=connection)
    applied = dict(connection.execute(select(legacy.c.name, legacy.c.checksum)).all())
    if not applied:
        return False
    if applied != {LEGACY_MIGRATION: LEGACY_CHECKSUM}:
        raise ValueError("Legacy migration has changed or contains unsupported migrations")
    if not all(inspector.has_table(name) for name in ("document_ingestions", "parent_chunks")):
        raise ValueError("Legacy migration is recorded but its storage tables are missing")
    return True

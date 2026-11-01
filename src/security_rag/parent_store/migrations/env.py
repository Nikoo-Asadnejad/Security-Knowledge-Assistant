"""Run migrations on the connection and transaction supplied by the CLI."""

from alembic import context

from security_rag.parent_store.schema import metadata

# The runner owns the outer transaction, including the migration advisory lock.
context.configure(connection=context.config.attributes["connection"], target_metadata=metadata)
with context.begin_transaction():
    context.run_migrations()

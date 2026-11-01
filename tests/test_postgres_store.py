"""Opt-in integration checks against a unique disposable schema, never production tables."""

import importlib
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    MetaData,
    Table,
    Text,
    create_engine,
    func,
    insert,
    inspect,
    or_,
    select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, TimeoutError
from sqlalchemy.schema import AddConstraint, CreateSchema, DropConstraint, DropSchema

from security_rag.config import Settings
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.ingestion.loader import parse_document
from security_rag.parent_store.migration_runner import (
    INITIAL_REVISION,
    LEGACY_CHECKSUM,
    LEGACY_MIGRATION,
)
from security_rag.parent_store.models import ParentReference, document_revision
from security_rag.parent_store.postgres_store import PostgresStore
from security_rag.parent_store.schema import parent_chunks


@pytest.fixture
def postgres_store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL for isolated PostgreSQL integration tests")
    database_url = make_url(url).set(drivername="postgresql+psycopg")
    schema = f"test_security_rag_{uuid4().hex}"
    admin = create_engine(database_url, hide_parameters=True)
    with admin.begin() as connection:
        connection.execute(CreateSchema(schema))
    engine = create_engine(
        database_url,
        pool_size=2,
        max_overflow=0,
        hide_parameters=True,
        connect_args={"options": f"-c search_path={schema}"},
    )
    try:
        yield PostgresStore(engine)
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        admin.dispose()


@pytest.fixture
def chunks(tokenizer, make_document):
    return MarkdownChunker(tokenizer).chunk(
        make_document(
            "### Code\n```python\n" + "print('alpha')\n" * 400 + "```\n\n"
            "### Mitigation\nalpha beta.\n"
        )
    )


def test_migrations_roundtrip_idempotency_and_lookup_after_restart(postgres_store, chunks):
    store = postgres_store
    assert store.migrate() == [INITIAL_REVISION]
    assert store.migrate() == []
    store.prepare_revision(chunks)
    store.prepare_revision(chunks)
    assert store.verify_parents(chunks, all_revisions=True)
    assert any(p.markdown_prefix or p.markdown_suffix for p in chunks.parents)
    revision = document_revision(chunks)
    state = store.ingestion_state(revision)
    assert state["status"] == "processing"
    assert state["expected_parents"] == len(chunks.parents)
    assert state["expected_children"] == len(chunks.children)
    store.finish_revision(chunks)

    # Dispose the pool: retrieval reconnects without source/tokenizer reconstruction.
    store.close()
    restarted = PostgresStore(store.engine)
    refs = [ParentReference.model_validate(c.model_dump()) for c in chunks.children[:2]]
    with patch.object(restarted, "get_parents", wraps=restarted.get_parents) as get_parents:
        actual = restarted.resolve_parents(refs)
    get_parents.assert_called_once()
    assert actual == {p.id: p for p in chunks.parents if p.id in {r.parent_id for r in refs}}
    wrong = refs[0].model_copy(update={"document_hash": "0" * 64})
    assert restarted.resolve_parents([wrong]) == {}
    assert restarted.get_parents([uuid4()]) == {}
    assert restarted.get_parents([]) == {}
    assert restarted.ingestion_state(revision)["status"] == "completed"
    assert store.statistics()[0]["expected_parents"] == len(chunks.parents)


def test_parent_transaction_rolls_back_state_and_every_parent(postgres_store, chunks):
    store = postgres_store
    store.migrate()
    with store.engine.begin() as connection:
        parents = Table("parent_chunks", MetaData(), autoload_with=connection)
        constraint = CheckConstraint(parents.c.parent_index != 1, name="interrupt_parent_write")
        parents.append_constraint(constraint)
        connection.execute(AddConstraint(constraint))
    with pytest.raises(IntegrityError):
        store.prepare_revision(chunks)
    assert store.ingestion_state(document_revision(chunks)) is None
    assert store.get_parents([p.id for p in chunks.parents]) == {}


def test_migration_failure_rolls_back_schema_and_version(postgres_store):
    store = postgres_store
    with (
        patch("alembic.op.create_index", side_effect=RuntimeError("migration interrupted")),
        pytest.raises(RuntimeError, match="migration interrupted"),
    ):
        store.migrate()
    assert inspect(store.engine).get_table_names() == []
    assert store.migrate() == [INITIAL_REVISION]
    assert store.migrate() == []


def test_settings_engine_enforces_pool_limit_and_statement_timeout(postgres_store):
    settings = Settings(
        _env_file=None,
        database_url=postgres_store.engine.url.render_as_string(hide_password=False),
        database_pool_size=1,
        database_timeout_seconds=0.05,
    )
    with closing(PostgresStore.from_settings(settings)) as store:
        with store.engine.connect() as connection:
            assert connection.scalar(select(func.current_setting("statement_timeout"))) == "50ms"
            # A second caller must time out rather than create an overflow connection.
            with pytest.raises(TimeoutError):
                store.engine.connect()
        with store.engine.connect() as connection:
            assert connection.scalar(select(1)) == 1


def create_legacy_schema(engine, checksum=LEGACY_CHECKSUM):
    """The frozen initial revision has the same tables as the original SQL migration."""
    initial = importlib.import_module(
        "security_rag.parent_store.migrations.versions.001_parent_storage"
    )
    legacy = Table(
        "schema_migrations",
        MetaData(),
        Column("name", Text, primary_key=True),
        Column("checksum", Text, nullable=False),
        Column("applied_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            initial.upgrade()
        legacy.create(connection)
        connection.execute(insert(legacy).values(name=LEGACY_MIGRATION, checksum=checksum))


def test_adopts_legacy_database_without_recreating_or_losing_parents(postgres_store, chunks):
    store = postgres_store
    create_legacy_schema(store.engine)
    store.prepare_revision(chunks)
    store.finish_revision(chunks)
    before = store.statistics()
    assert store.migrate() == []
    assert store.migrate() == []
    assert store.statistics() == before
    assert store.verify_parents(chunks, all_revisions=True)
    with store.engine.connect() as connection:
        assert MigrationContext.configure(connection).get_current_revision() == INITIAL_REVISION


def test_rejects_changed_legacy_migration_without_stamping(postgres_store):
    store = postgres_store
    create_legacy_schema(store.engine, checksum="0" * 64)
    with pytest.raises(ValueError, match="Legacy migration has changed"):
        store.migrate()
    assert not inspect(store.engine).has_table("alembic_version")


@pytest.mark.parametrize("title", ["Prompt Injection", "Updated Title"])
def test_cleanup_keeps_other_documents_and_only_new_revision(
    postgres_store, tokenizer, make_document, title
):
    store = postgres_store
    store.migrate()
    chunker = MarkdownChunker(tokenizer)
    old = chunker.chunk(make_document("### Test\nold alpha.\n"))
    new = chunker.chunk(make_document("### Test\nnew beta.\n", title=title))
    other = chunker.chunk(
        parse_document(
            Path("LLM02_SensitiveInformationDisclosure.md"),
            b"## LLM02:2026 Sensitive Information Disclosure\n\n### Test\nother gamma.\n",
        )
    )
    store.prepare_revision(other)
    store.finish_revision(other)
    store.prepare_revision(old)
    store.finish_revision(old)
    store.prepare_revision(new)
    assert not store.verify_parents(new, all_revisions=True)
    store.finish_revision(new)
    assert store.get_parents([p.id for p in old.parents]) == {}
    assert store.verify_parents(new, all_revisions=True)
    assert store.verify_parents(other, all_revisions=True)
    assert len(store.statistics()) == 2
    assert store.ingestion_state(document_revision(new))["status"] == "completed"


def test_verification_can_ignore_malformed_obsolete_parents(
    postgres_store, tokenizer, make_document
):
    store = postgres_store
    store.migrate()
    chunker = MarkdownChunker(tokenizer)
    old = chunker.chunk(make_document("### Test\nold alpha.\n"))
    new = chunker.chunk(make_document("### Test\nnew beta.\n"))
    store.prepare_revision(old)
    store.prepare_revision(new)
    with store.engine.begin() as connection:
        connection.execute(
            parent_chunks.update()
            .where(parent_chunks.c.id.in_([parent.id for parent in old.parents]))
            .values(parent_hash="0" * 64)
        )
    assert not store.verify_parents(new, all_revisions=True)
    assert store.verify_parents(new, all_revisions=True, allow_extra=True)
    store.finish_revision(new)
    assert store.verify_parents(new, all_revisions=True)


def test_real_postgres_and_qdrant_pipeline_recovers_from_partial_child_upserts(
    postgres_store, tokenizer, make_document
):
    from qdrant_client import QdrantClient

    from security_rag.ingestion.pipeline import IngestionPipeline
    from security_rag.vector_store.qdrant_store import QdrantStore

    store = postgres_store
    store.migrate()
    document = make_document("### Test\nalpha.\n\n### Other\nbeta.\n")
    chunker = MarkdownChunker(tokenizer)
    chunks = chunker.chunk(document)

    class LocalEmbedder:
        def embed(self, children):
            return [[1.0] + [0.0] * 383 for _ in children]

    with closing(QdrantClient(":memory:")) as client:
        children = QdrantStore(client, "postgres_integration", 1)
        children.ensure_collection()
        pipeline = IngestionPipeline(chunker, store, children, LocalEmbedder(), batch_size=1)
        upsert = children.upsert

        def interrupt(batch, vectors):
            upsert(batch, vectors)
            raise RuntimeError("interrupted after successful child write")

        with patch.object(children, "upsert", side_effect=interrupt), pytest.raises(RuntimeError):
            pipeline.ingest_document(document)
        assert store.ingestion_state(document_revision(chunks))["status"] == "processing"
        assert store.verify_parents(chunks)
        assert client.count("postgres_integration").count == 1
        assert not pipeline.ingest_document(document).skipped
        assert pipeline.ingest_document(document).skipped
        assert children.verify_children(chunks, all_revisions=True)


def test_failed_completion_rolls_back_parent_cleanup_and_can_retry(
    postgres_store, tokenizer, make_document
):
    store = postgres_store
    store.migrate()
    chunker = MarkdownChunker(tokenizer)
    old = chunker.chunk(make_document("### Test\nold alpha.\n"))
    new = chunker.chunk(make_document("### Test\nnew beta.\n"))
    store.prepare_revision(old)
    store.finish_revision(old)
    store.prepare_revision(new)
    with store.engine.begin() as connection:
        states = Table("document_ingestions", MetaData(), autoload_with=connection)
        constraint = CheckConstraint(
            or_(
                states.c.document_hash != new.parents[0].document_hash,
                states.c.status != "completed",
            ),
            name="interrupt_completion",
        )
        states.append_constraint(constraint)
        connection.execute(AddConstraint(constraint))
    with pytest.raises(IntegrityError):
        store.finish_revision(new)
    assert store.get_parents([p.id for p in old.parents])
    assert store.ingestion_state(document_revision(old))["status"] == "completed"
    assert store.ingestion_state(document_revision(new))["status"] == "processing"
    with store.engine.begin() as connection:
        connection.execute(DropConstraint(constraint))
    store.finish_revision(new)
    assert store.verify_parents(new, all_revisions=True)
    assert store.ingestion_state(document_revision(new))["status"] == "completed"

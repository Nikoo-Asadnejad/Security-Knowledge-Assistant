"""Opt-in integration checks against a unique disposable schema, never production tables."""

import importlib
import os
from contextlib import closing
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
    select,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError, TimeoutError
from sqlalchemy.schema import AddConstraint, CreateSchema, DropConstraint, DropSchema

from security_rag.config import Settings
from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.parent_store.migration_runner import (
    INITIAL_REVISION,
    LEGACY_CHECKSUM,
    LEGACY_MIGRATION,
)
from security_rag.parent_store.models import ParentReference, document_revision
from security_rag.parent_store.postgres_store import PostgresStore


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
    assert store.migrate() == [INITIAL_REVISION, "002_sparse_indexing"]
    assert store.migrate() == []
    store.prepare_revision(chunks)
    store.prepare_revision(chunks)
    assert store.verify_parents(chunks)
    assert any(p.markdown_prefix or p.markdown_suffix for p in chunks.parents)
    revision = document_revision(chunks)
    state = store.ingestion_state(revision)
    assert state["status"] == "processing"
    assert state["expected_parents"] == len(chunks.parents)
    assert state["expected_children"] == len(chunks.children)
    store.finish_revision(chunks, "a" * 64)

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


def test_live_initial_ingestion_stores_both_vectors_and_parent_context(
    postgres_store, tokenizer, make_document, sparse_encoder
):
    from qdrant_client import QdrantClient, models

    from security_rag.ingestion.pipeline import IngestionPipeline
    from security_rag.vector_store.qdrant_store import QdrantStore

    url = os.environ.get("TEST_QDRANT_URL")
    if not url:
        pytest.skip("Set TEST_QDRANT_URL for the dual-store acceptance test")
    parents = postgres_store
    parents.migrate()
    parents.save_sparse_config(sparse_encoder.config)
    source = make_document("### Mitigation\nCWE-79 alpha.\n\n#### Details\nLLM01 beta.\n")
    chunks = MarkdownChunker(tokenizer).chunk(source)
    name = f"test_initial_{uuid4().hex}"

    class LocalEmbedder:
        def embed(self, children):
            return [[1.0] + [0.0] * 383 for _ in children]

    with closing(QdrantClient(url=url, timeout=10)) as client:
        store = QdrantStore(client, name, batch_size=1)
        store.ensure_collection()
        try:
            pipeline = IngestionPipeline(
                MarkdownChunker(tokenizer),
                parents,
                store,
                LocalEmbedder(),
                sparse_encoder,
                batch_size=1,
            )
            report = pipeline.ingest_document(source)
            config_hash = sparse_encoder.config.fingerprint
            assert report.children == len(chunks.children)
            assert store.verify_children(chunks, config_hash)
            info = client.get_collection(name)
            assert info.config.params.sparse_vectors["bm25"].modifier == models.Modifier.IDF
            assert info.payload_schema["identifiers"].data_type == models.PayloadSchemaType.KEYWORD
            assert info.payload_schema["document_id"].data_type == models.PayloadSchemaType.KEYWORD
            query = sparse_encoder.encode(["CWE-79"], query=True)[0]
            for code, expected in [("CWE-79", 1), ("CWE-790", 0)]:
                hits = client.query_points(
                    name,
                    query=query,
                    using="bm25",
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="identifiers", match=models.MatchValue(value=code)
                            )
                        ]
                    ),
                ).points
                assert len(hits) == expected
            parents.close()
            restarted = PostgresStore(parents.engine)
            state = restarted.ingestion_state(document_revision(chunks))
            assert state["status"] == state["sparse_status"] == "completed"
            assert state["sparse_config_hash"] == config_hash
            assert restarted.sparse_config() == sparse_encoder.config
            assert restarted.verify_parents(chunks)
            pipeline.ingest_document(source)
            assert store.statistics()["children"] == len(chunks.children)
        finally:
            client.delete_collection(name)


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
    assert store.migrate() == [INITIAL_REVISION, "002_sparse_indexing"]
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
    # Populate the old schema using its original column definitions.
    with store.engine.begin() as connection:
        states = Table("document_ingestions", MetaData(), autoload_with=connection)
        parents = Table("parent_chunks", MetaData(), autoload_with=connection)
        connection.execute(
            insert(states).values(
                **document_revision(chunks).model_dump(),
                status="completed",
                expected_parents=len(chunks.parents),
                expected_children=len(chunks.children),
            )
        )
        connection.execute(
            insert(parents),
            [{**parent.model_dump(mode="json"), "id": parent.id} for parent in chunks.parents],
        )
    assert store.migrate() == ["002_sparse_indexing"]
    store.prepare_revision(chunks)
    store.finish_revision(chunks, "a" * 64)
    before = store.statistics()
    assert store.migrate() == []
    assert store.migrate() == []
    assert store.statistics() == before
    assert store.verify_parents(chunks)
    with store.engine.connect() as connection:
        assert (
            MigrationContext.configure(connection).get_current_revision() == "002_sparse_indexing"
        )


def test_rejects_changed_legacy_migration_without_stamping(postgres_store):
    store = postgres_store
    create_legacy_schema(store.engine, checksum="0" * 64)
    with pytest.raises(ValueError, match="Legacy migration has changed"):
        store.migrate()
    assert not inspect(store.engine).has_table("alembic_version")


@pytest.mark.parametrize("change", ["source", "title", "config"])
def test_document_updates_are_rejected_before_parent_or_state_changes(
    postgres_store, tokenizer, make_document, change
):
    from security_rag.ingestion.chunk_models import ChunkingConfig

    store = postgres_store
    store.migrate()
    chunker = MarkdownChunker(tokenizer)
    source = make_document("### Test\nold alpha.\n")
    old = chunker.chunk(source)
    store.prepare_revision(old)
    store.finish_revision(old, "a" * 64)
    before = store.statistics()
    if change == "config":
        chunker = MarkdownChunker(tokenizer, ChunkingConfig(child_overlap_tokens=30))
    else:
        source = make_document(
            "### Test\nnew beta.\n",
            title="Updated Title" if change == "title" else "Prompt Injection",
        )
    new = chunker.chunk(source)
    with pytest.raises(ValueError, match="Phase 8"):
        store.prepare_revision(new)
    assert store.statistics() == before
    assert store.verify_parents(old)
    assert store.get_parents([p.id for p in new.parents]) == {}


def test_real_postgres_and_qdrant_pipeline_recovers_from_partial_child_upserts(
    postgres_store, tokenizer, make_document, sparse_encoder
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
        pipeline = IngestionPipeline(
            chunker, store, children, LocalEmbedder(), sparse_encoder, batch_size=1
        )
        upsert = children.upsert

        def interrupt(*args):
            upsert(*args)
            raise RuntimeError("interrupted after successful child write")

        with patch.object(children, "upsert", side_effect=interrupt), pytest.raises(RuntimeError):
            pipeline.ingest_document(document)
        assert store.ingestion_state(document_revision(chunks))["status"] == "processing"
        assert store.verify_parents(chunks)
        assert client.count("postgres_integration").count == 1
        pipeline.ingest_document(document)
        assert children.verify_children(chunks, sparse_encoder.config.fingerprint)


def test_failed_completion_keeps_processing_state_until_verified_retry(
    postgres_store, tokenizer, make_document
):
    store = postgres_store
    store.migrate()
    chunks = MarkdownChunker(tokenizer).chunk(make_document("### Test\nalpha beta.\n"))
    store.prepare_revision(chunks)
    with store.engine.begin() as connection:
        states = Table("document_ingestions", MetaData(), autoload_with=connection)
        constraint = CheckConstraint(states.c.status != "completed", name="interrupt_completion")
        states.append_constraint(constraint)
        connection.execute(AddConstraint(constraint))
    with pytest.raises(IntegrityError):
        store.finish_revision(chunks, "a" * 64)
    assert store.verify_parents(chunks)
    assert store.ingestion_state(document_revision(chunks))["status"] == "processing"
    with store.engine.begin() as connection:
        connection.execute(DropConstraint(constraint))
    store.finish_revision(chunks, "a" * 64)
    assert store.ingestion_state(document_revision(chunks))["status"] == "completed"

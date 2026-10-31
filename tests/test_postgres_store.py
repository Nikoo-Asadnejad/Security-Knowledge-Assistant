"""Opt-in integration checks against a unique disposable schema, never production tables."""

import os
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from security_rag.ingestion.chunker import MarkdownChunker
from security_rag.parent_store.models import ParentReference, document_revision
from security_rag.parent_store.postgres_store import PostgresStore

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


@pytest.fixture
def postgres_store():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL for isolated PostgreSQL integration tests")
    schema = f"test_security_rag_{uuid4().hex}"
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    pool = ConnectionPool(
        url,
        min_size=1,
        max_size=2,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row, "options": f"-c search_path={schema}"},
    )
    pool.open(wait=True)
    try:
        yield PostgresStore(pool)
    finally:
        pool.close()
        with psycopg.connect(url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


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
    assert store.migrate(MIGRATIONS) == ["001_parent_storage.sql"]
    assert store.migrate(MIGRATIONS) == []
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

    # New pool and store: persisted parents are available without source/tokenizer reconstruction.
    pool = ConnectionPool(
        store.pool.conninfo, min_size=1, max_size=1, open=False, kwargs=store.pool.kwargs
    )
    pool.open(wait=True)
    restarted = PostgresStore(pool)
    try:
        refs = [ParentReference.model_validate(c.model_dump()) for c in chunks.children[:2]]
        with pytest.MonkeyPatch.context() as mp:
            calls = []
            get_parents = restarted.get_parents

            def capture(ids):
                calls.append(ids)
                return get_parents(ids)

            mp.setattr(restarted, "get_parents", capture)
            actual = restarted.resolve_parents(refs)
        assert len(calls) == 1
        assert actual == {p.id: p for p in chunks.parents if p.id in {r.parent_id for r in refs}}
        wrong = refs[0].model_copy(update={"document_hash": "0" * 64})
        assert restarted.resolve_parents([wrong]) == {}
        assert restarted.get_parents([uuid4()]) == {}
        assert restarted.get_parents([]) == {}
        assert restarted.ingestion_state(revision)["status"] == "completed"
    finally:
        restarted.close()


def test_parent_transaction_rolls_back_state_and_every_parent(postgres_store, chunks):
    store = postgres_store
    store.migrate(MIGRATIONS)
    with store.pool.connection() as conn:
        conn.execute("ALTER TABLE parent_chunks ADD CHECK (parent_index <> 1)")
    with pytest.raises(psycopg.errors.CheckViolation):
        store.prepare_revision(chunks)
    assert store.ingestion_state(document_revision(chunks)) is None
    assert store.get_parents([p.id for p in chunks.parents]) == {}


def test_migration_failure_rolls_back_and_changed_migration_is_rejected(postgres_store, tmp_path):
    store = postgres_store
    (tmp_path / "001_parent_storage.sql").write_bytes(
        (MIGRATIONS / "001_parent_storage.sql").read_bytes()
    )
    (tmp_path / "002_bad.sql").write_text("CREATE TABLE broken (")
    with pytest.raises(psycopg.errors.SyntaxError):
        store.migrate(tmp_path)
    with store.pool.connection() as conn:
        assert (
            conn.execute("SELECT to_regclass('schema_migrations') AS name").fetchone()["name"]
            is None
        )
    (tmp_path / "002_bad.sql").unlink()
    assert store.migrate(tmp_path) == ["001_parent_storage.sql"]
    (tmp_path / "001_parent_storage.sql").write_text("SELECT 1;")
    with pytest.raises(ValueError, match="migration has changed"):
        store.migrate(tmp_path)


@pytest.mark.parametrize("title", ["Prompt Injection", "Updated Title"])
def test_cleanup_keeps_other_documents_and_only_new_revision(
    postgres_store, tokenizer, make_document, title
):
    store = postgres_store
    store.migrate(MIGRATIONS)
    chunker = MarkdownChunker(tokenizer)
    old = chunker.chunk(make_document("### Test\nold alpha.\n"))
    new = chunker.chunk(make_document("### Test\nnew beta.\n", title=title))
    store.prepare_revision(old)
    store.finish_revision(old)
    store.prepare_revision(new)
    assert not store.verify_parents(new, all_revisions=True)
    store.finish_revision(new)
    assert store.get_parents([p.id for p in old.parents]) == {}
    assert store.verify_parents(new, all_revisions=True)
    assert len(store.statistics()) == 1
    assert store.ingestion_state(document_revision(new))["status"] == "completed"


def test_real_postgres_and_qdrant_pipeline_recovers_from_partial_child_upserts(
    postgres_store, tokenizer, make_document
):
    from unittest.mock import patch

    from qdrant_client import QdrantClient

    from security_rag.ingestion.pipeline import IngestionPipeline
    from security_rag.vector_store.qdrant_store import QdrantStore

    store = postgres_store
    store.migrate(MIGRATIONS)
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
    store.migrate(MIGRATIONS)
    chunker = MarkdownChunker(tokenizer)
    old = chunker.chunk(make_document("### Test\nold alpha.\n"))
    new = chunker.chunk(make_document("### Test\nnew beta.\n"))
    store.prepare_revision(old)
    store.finish_revision(old)
    store.prepare_revision(new)
    with store.pool.connection() as conn:
        conn.execute(
            """
            CREATE FUNCTION fail_completion() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                IF NEW.status = 'completed' THEN
                    RAISE EXCEPTION 'completion interrupted';
                END IF;
                RETURN NEW;
            END $$;
            CREATE TRIGGER interrupt_completion BEFORE UPDATE ON document_ingestions
                FOR EACH ROW EXECUTE FUNCTION fail_completion();
        """,
            prepare=False,
        )
    with pytest.raises(psycopg.errors.RaiseException):
        store.finish_revision(new)
    assert store.get_parents([p.id for p in old.parents])
    assert store.ingestion_state(document_revision(old))["status"] == "completed"
    assert store.ingestion_state(document_revision(new))["status"] == "processing"
    with store.pool.connection() as conn:
        conn.execute("DROP TRIGGER interrupt_completion ON document_ingestions")
    store.finish_revision(new)
    assert store.verify_parents(new, all_revisions=True)
    assert store.ingestion_state(document_revision(new))["status"] == "completed"

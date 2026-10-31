"""Durable parent records and ingestion progress using explicit transactions."""

import hashlib
from pathlib import Path
from uuid import UUID

from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import ValidationError

from security_rag.config import Settings
from security_rag.ingestion.chunk_models import ChunkedDocument, ParentChunk
from security_rag.parent_store.models import ParentReference, Revision, document_revision

REVISION_WHERE = "document_id = %s AND document_hash = %s AND chunking_config_hash = %s"


class PostgresStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool

    @classmethod
    def from_settings(cls, settings: Settings) -> "PostgresStore":
        url = settings.database_url.get_secret_value()
        if not url:
            raise ValueError("Set DATABASE_URL before using PostgreSQL")
        pool = ConnectionPool(
            url,
            min_size=1,
            max_size=settings.database_pool_size,
            timeout=settings.database_timeout_seconds,
            max_waiting=settings.database_pool_size,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "connect_timeout": max(1, int(settings.database_timeout_seconds)),
                "options": f"-c statement_timeout={int(settings.database_timeout_seconds * 1000)}",
            },
            open=False,
        )
        try:
            pool.open(wait=True, timeout=settings.database_timeout_seconds)
        except Exception:
            pool.close()
            raise
        return cls(pool)

    def close(self) -> None:
        self.pool.close()

    def migrate(self, directory: Path = Path("migrations")) -> list[str]:
        paths = sorted(directory.glob("[0-9]*_*.sql"))
        if not paths:
            raise ValueError(f"No SQL migrations found in {directory}")
        applied = []
        with self.pool.connection() as conn, conn.transaction():
            # Serialize explicit migration commands; ingestion still runs with the API stopped.
            conn.execute("SELECT pg_advisory_xact_lock(726403101)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "name text PRIMARY KEY, checksum text NOT NULL, "
                "applied_at timestamptz NOT NULL DEFAULT now())"
            )
            existing = {
                row["name"]: row["checksum"]
                for row in conn.execute("SELECT name, checksum FROM schema_migrations")
            }
            for path in paths:
                content = path.read_bytes()
                checksum = hashlib.sha256(content).hexdigest()
                if path.name in existing:
                    if existing[path.name] != checksum:
                        raise ValueError(f"Applied migration has changed: {path.name}")
                    continue
                conn.execute(content.decode("utf-8"), prepare=False)
                conn.execute(
                    "INSERT INTO schema_migrations (name, checksum) VALUES (%s, %s)",
                    (path.name, checksum),
                )
                applied.append(path.name)
        return applied

    def ingestion_state(self, revision: Revision) -> dict | None:
        with self.pool.connection() as conn, conn.transaction():
            return conn.execute(
                f"SELECT * FROM document_ingestions WHERE {REVISION_WHERE}", revision.key
            ).fetchone()

    def prepare_revision(self, chunks: ChunkedDocument) -> None:
        revision = document_revision(chunks)
        fields = list(ParentChunk.model_fields)
        statement = sql.SQL(
            "INSERT INTO parent_chunks ({fields}) VALUES ({values}) "
            "ON CONFLICT (id) DO UPDATE SET {updates}"
        ).format(
            fields=sql.SQL(", ").join(map(sql.Identifier, fields)),
            values=sql.SQL(", ").join(sql.Placeholder() for _ in fields),
            updates=sql.SQL(", ").join(
                sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(field))
                for field in fields
                if field != "id"
            ),
        )
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO document_ingestions "
                "(document_id, document_hash, chunking_config_hash, status, "
                "expected_parents, expected_children) VALUES (%s, %s, %s, 'processing', %s, %s) "
                "ON CONFLICT (document_id, document_hash, chunking_config_hash) DO UPDATE SET "
                "status = 'processing', expected_parents = EXCLUDED.expected_parents, "
                "expected_children = EXCLUDED.expected_children, updated_at = now()",
                (*revision.key, len(chunks.parents), len(chunks.children)),
            )
            rows = []
            for parent in chunks.parents:
                record = parent.model_dump(mode="json")
                record["id"] = parent.id
                rows.append(tuple(record[field] for field in fields))
            with conn.cursor() as cursor:
                cursor.executemany(statement, rows)

    def get_parents(self, ids: list[UUID]) -> dict[UUID, ParentChunk]:
        if not ids:
            return {}
        with self.pool.connection() as conn, conn.transaction():
            rows = conn.execute("SELECT * FROM parent_chunks WHERE id = ANY(%s)", (ids,))
            return {row["id"]: ParentChunk.model_validate(row) for row in rows}

    def resolve_parents(self, references: list[ParentReference]) -> dict[UUID, ParentChunk]:
        """Fetch only requested IDs, in one query, excluding mismatched revisions."""
        parents = self.get_parents(list({ref.parent_id for ref in references}))
        return {
            ref.parent_id: parents[ref.parent_id]
            for ref in references
            if ref.parent_id in parents and Revision.of(parents[ref.parent_id]).key == ref.key
        }

    def verify_parents(
        self, chunks: ChunkedDocument, *, all_revisions: bool = False, allow_extra: bool = False
    ) -> bool:
        revision = document_revision(chunks)
        where = "(document_id = %s OR file_name = %s)" if all_revisions else REVISION_WHERE
        params = (
            (revision.document_id, chunks.parents[0].file_name) if all_revisions else revision.key
        )
        expected = {parent.id: parent for parent in chunks.parents}
        with self.pool.connection() as conn, conn.transaction():
            if allow_extra:
                # Unexpected records are removed after child cleanup. Validate only
                # the expected IDs here, so even malformed obsolete rows can be repaired.
                where += " AND id = ANY(%s)"
                params = (*params, list(expected))
            rows = conn.execute(f"SELECT * FROM parent_chunks WHERE {where}", params)
            try:
                actual = {row["id"]: ParentChunk.model_validate(row) for row in rows}
            except ValidationError:
                return False
        return actual == expected

    def finish_revision(self, chunks: ChunkedDocument) -> None:
        """Call only after Qdrant obsolete-child removal and verification succeed."""
        revision = document_revision(chunks)
        with self.pool.connection() as conn, conn.transaction():
            # The manifest filename remains stable even if a revised title changes its slug ID.
            identity = (revision.document_id, chunks.parents[0].file_name)
            document_ids = {revision.document_id} | {
                row["document_id"]
                for row in conn.execute(
                    "SELECT DISTINCT document_id FROM parent_chunks "
                    "WHERE document_id = %s OR file_name = %s",
                    identity,
                )
            }
            conn.execute(
                "DELETE FROM parent_chunks WHERE (document_id = %s OR file_name = %s) "
                "AND NOT (id = ANY(%s))",
                (*identity, [parent.id for parent in chunks.parents]),
            )
            conn.execute(
                f"DELETE FROM document_ingestions WHERE document_id = ANY(%s) "
                f"AND NOT ({REVISION_WHERE})",
                (list(document_ids), *revision.key),
            )
            result = conn.execute(
                f"UPDATE document_ingestions SET status = 'completed', updated_at = now() "
                f"WHERE {REVISION_WHERE}",
                revision.key,
            )
            if result.rowcount != 1:
                raise ValueError("Cannot complete a revision without ingestion state")

    def statistics(self) -> list[dict]:
        with self.pool.connection() as conn, conn.transaction():
            return conn.execute(
                "SELECT document_id, document_hash, chunking_config_hash, status, "
                "expected_parents, expected_children, updated_at FROM document_ingestions "
                "ORDER BY document_id, updated_at"
            ).fetchall()

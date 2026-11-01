"""Durable parent records using SQLAlchemy Core and explicit transactions."""

from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import Engine, Table, and_, create_engine, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import make_url
from sqlalchemy.sql.elements import ColumnElement

from security_rag.config import Settings
from security_rag.ingestion.chunk_models import ChunkedDocument, ParentChunk
from security_rag.parent_store.migration_runner import MIGRATIONS, migrate
from security_rag.parent_store.models import ParentReference, Revision, document_revision
from security_rag.parent_store.schema import REVISION_COLUMNS, document_ingestions, parent_chunks


def _revision_where(table: Table, revision: Revision) -> ColumnElement[bool]:
    return and_(
        *(
            table.c[name] == value
            for name, value in zip(REVISION_COLUMNS, revision.key, strict=True)
        )
    )


class PostgresStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def from_settings(cls, settings: Settings) -> "PostgresStore":
        url = settings.database_url.get_secret_value()
        if not url:
            raise ValueError("Set DATABASE_URL before using PostgreSQL")
        database_url = make_url(url)
        if database_url.drivername not in {"postgres", "postgresql", "postgresql+psycopg"}:
            raise ValueError("DATABASE_URL must use PostgreSQL with the Psycopg driver")
        timeout_ms = max(1, int(settings.database_timeout_seconds * 1000))
        engine = create_engine(
            database_url.set(drivername="postgresql+psycopg"),
            pool_size=settings.database_pool_size,
            max_overflow=0,
            pool_timeout=settings.database_timeout_seconds,
            pool_pre_ping=True,
            hide_parameters=True,
            connect_args={
                "connect_timeout": max(1, int(settings.database_timeout_seconds)),
                "options": f"-c statement_timeout={timeout_ms}",
            },
        )
        try:
            with engine.connect() as connection:
                connection.execute(select(1))
        except Exception:
            engine.dispose()
            raise
        return cls(engine)

    def close(self) -> None:
        self.engine.dispose()

    def migrate(self, directory: Path = MIGRATIONS) -> list[str]:
        return migrate(self.engine, directory)

    def ingestion_state(self, revision: Revision) -> dict | None:
        statement = select(document_ingestions).where(
            _revision_where(document_ingestions, revision)
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
            return dict(row) if row is not None else None

    def prepare_revision(self, chunks: ChunkedDocument) -> None:
        revision = document_revision(chunks)
        state = insert(document_ingestions).values(
            **revision.model_dump(),
            status="processing",
            expected_parents=len(chunks.parents),
            expected_children=len(chunks.children),
        )
        state = state.on_conflict_do_update(
            index_elements=list(REVISION_COLUMNS),
            set_={
                "status": "processing",
                "expected_parents": state.excluded.expected_parents,
                "expected_children": state.excluded.expected_children,
                "updated_at": func.now(),
            },
        )
        parents = insert(parent_chunks)
        parents = parents.on_conflict_do_update(
            index_elements=[parent_chunks.c.id],
            set_={
                column.name: parents.excluded[column.name]
                for column in parent_chunks.c
                if column.name != "id"
            },
        )
        rows = []
        for parent in chunks.parents:
            record = parent.model_dump(mode="json")
            record["id"] = parent.id
            rows.append(record)
        with self.engine.begin() as connection:
            connection.execute(state)
            connection.execute(parents, rows)

    def get_parents(self, ids: list[UUID]) -> dict[UUID, ParentChunk]:
        if not ids:
            return {}
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(parent_chunks).where(parent_chunks.c.id.in_(ids))
            ).mappings()
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
        where = (
            or_(
                parent_chunks.c.document_id == revision.document_id,
                parent_chunks.c.file_name == chunks.parents[0].file_name,
            )
            if all_revisions
            else _revision_where(parent_chunks, revision)
        )
        expected = {parent.id: parent for parent in chunks.parents}
        if allow_extra:
            # Obsolete records are removed after child cleanup. Only validate expected IDs.
            where = and_(where, parent_chunks.c.id.in_(expected))
        with self.engine.connect() as connection:
            rows = connection.execute(select(parent_chunks).where(where)).mappings()
            try:
                actual = {row["id"]: ParentChunk.model_validate(row) for row in rows}
            except ValidationError:
                return False
        return actual == expected

    def finish_revision(self, chunks: ChunkedDocument) -> None:
        """Call only after Qdrant obsolete-child removal and verification succeed."""
        revision = document_revision(chunks)
        # The manifest filename remains stable even if a revised title changes its slug ID.
        identity = or_(
            parent_chunks.c.document_id == revision.document_id,
            parent_chunks.c.file_name == chunks.parents[0].file_name,
        )
        with self.engine.begin() as connection:
            document_ids = {revision.document_id} | set(
                connection.execute(
                    select(parent_chunks.c.document_id).where(identity).distinct()
                ).scalars()
            )
            connection.execute(
                delete(parent_chunks).where(
                    identity, parent_chunks.c.id.not_in([parent.id for parent in chunks.parents])
                )
            )
            connection.execute(
                delete(document_ingestions).where(
                    document_ingestions.c.document_id.in_(document_ids),
                    ~_revision_where(document_ingestions, revision),
                )
            )
            result = connection.execute(
                update(document_ingestions)
                .where(_revision_where(document_ingestions, revision))
                .values(status="completed", updated_at=func.now())
            )
            if result.rowcount != 1:
                raise ValueError("Cannot complete a revision without ingestion state")

    def statistics(self) -> list[dict]:
        statement = select(document_ingestions).order_by(
            document_ingestions.c.document_id, document_ingestions.c.updated_at
        )
        with self.engine.connect() as connection:
            return [dict(row) for row in connection.execute(statement).mappings()]

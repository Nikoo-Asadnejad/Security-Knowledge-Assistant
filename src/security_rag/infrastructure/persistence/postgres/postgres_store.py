"""Durable parent records using SQLAlchemy Core and explicit transactions."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import Connection, Engine, Table, and_, create_engine, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import make_url
from sqlalchemy.sql.elements import ColumnElement

from security_rag.config import Settings
from security_rag.core.application.ingest import IngestionBusy
from security_rag.core.domain.chunks import ChunkedDocument, ParentChunk
from security_rag.core.domain.revisions import Revision, document_revision
from security_rag.core.domain.sparse_config import SparseConfig
from security_rag.infrastructure.persistence.postgres.migration_runner import MIGRATIONS, migrate
from security_rag.infrastructure.persistence.postgres.schema import (
    REVISION_COLUMNS,
    document_ingestions,
    parent_chunks,
    sparse_configuration,
)

INGESTION_LOCK = 726403102


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
    def from_settings(cls, settings: Settings, *, check_connection: bool = True) -> "PostgresStore":
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
        if check_connection:
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

    @contextmanager
    def single_writer(self) -> Iterator[None]:
        # Detach this dedicated session so a pool of size one can still run writes.
        # Closing the physical session releases its advisory lock, including on failure.
        with self.engine.connect() as session:
            session.detach()
            acquired = session.scalar(select(func.pg_try_advisory_lock(INGESTION_LOCK)))
            session.commit()
            if not acquired:
                raise IngestionBusy("Another ingestion or search is already running")
            yield

    @contextmanager
    def search_snapshot(self) -> Iterator[tuple[SparseConfig | None, list[dict]]]:
        with self.engine.begin() as connection:
            acquired = connection.scalar(
                select(func.pg_try_advisory_xact_lock_shared(INGESTION_LOCK))
            )
            if not acquired:
                raise IngestionBusy("An ingestion is already running")
            yield self._sparse_config(connection), self._statistics(connection)

    def ingestion_state(self, revision: Revision) -> dict | None:
        statement = select(document_ingestions).where(
            _revision_where(document_ingestions, revision)
        )
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
            return dict(row) if row is not None else None

    def prepare_revision(
        self, chunks: ChunkedDocument, *, check_existing: Callable[[tuple[Revision, ...]], None]
    ) -> None:
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
                "sparse_status": "processing",
                "sparse_config_hash": None,
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
            existing = connection.execute(
                select(*(document_ingestions.c[name] for name in REVISION_COLUMNS)).where(
                    or_(
                        document_ingestions.c.document_id == revision.document_id,
                        document_ingestions.c.document_id.in_(
                            select(parent_chunks.c.document_id).where(
                                parent_chunks.c.file_name == chunks.parents[0].file_name
                            )
                        ),
                    )
                )
            ).mappings()
            check_existing(tuple(Revision.model_validate(row) for row in existing))
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

    def verify_parents(self, chunks: ChunkedDocument) -> bool:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(parent_chunks).where(
                    _revision_where(parent_chunks, document_revision(chunks))
                )
            ).mappings()
            try:
                actual = {row["id"]: ParentChunk.model_validate(row) for row in rows}
            except ValidationError:
                return False
        return actual == {parent.id: parent for parent in chunks.parents}

    def finish_revision(self, chunks: ChunkedDocument, sparse_config_hash: str) -> None:
        """Complete only after parents and both child vectors have been verified."""
        with self.engine.begin() as connection:
            result = connection.execute(
                update(document_ingestions)
                .where(_revision_where(document_ingestions, document_revision(chunks)))
                .values(
                    status="completed",
                    sparse_status="completed",
                    sparse_config_hash=sparse_config_hash,
                    updated_at=func.now(),
                )
            )
            if result.rowcount != 1:
                raise ValueError("Cannot complete a revision without ingestion state")

    def sparse_config(self) -> SparseConfig | None:
        with self.engine.connect() as connection:
            return self._sparse_config(connection)

    @staticmethod
    def _sparse_config(connection: Connection) -> SparseConfig | None:
        value = connection.scalar(select(sparse_configuration.c.config))
        return SparseConfig.model_validate(value) if value is not None else None

    def save_sparse_config(self, config: SparseConfig) -> SparseConfig:
        with self.engine.begin() as connection:
            connection.execute(
                insert(sparse_configuration)
                .values(id=1, config=config.model_dump(mode="json"))
                .on_conflict_do_nothing(index_elements=[sparse_configuration.c.id])
            )
            value = connection.execute(select(sparse_configuration.c.config)).scalar_one()
            return SparseConfig.model_validate(value)

    def statistics(self) -> list[dict]:
        with self.engine.connect() as connection:
            return self._statistics(connection)

    @staticmethod
    def _statistics(connection: Connection) -> list[dict]:
        statement = select(document_ingestions).order_by(
            document_ingestions.c.document_id, document_ingestions.c.updated_at
        )
        return [dict(row) for row in connection.execute(statement).mappings()]

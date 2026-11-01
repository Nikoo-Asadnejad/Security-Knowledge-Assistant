"""SQLAlchemy storage definitions, independent of the validated chunk models."""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Table,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY

metadata = MetaData()
REVISION_COLUMNS = ("document_id", "document_hash", "chunking_config_hash")

document_ingestions = Table(
    "document_ingestions",
    metadata,
    *(Column(name, Text, primary_key=True) for name in REVISION_COLUMNS),
    Column("status", Text, nullable=False),
    Column("expected_parents", Integer, nullable=False),
    Column("expected_children", Integer, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
for expression in (
    document_ingestions.c.status.in_(("processing", "completed")),
    document_ingestions.c.expected_parents > 0,
    document_ingestions.c.expected_children > 0,
):
    document_ingestions.append_constraint(CheckConstraint(expression))

parent_chunks = Table(
    "parent_chunks",
    metadata,
    Column("id", Uuid, primary_key=True),
    *(
        Column(name, Text, nullable=False)
        for name in (
            *REVISION_COLUMNS,
            "document_code",
            "title",
            "file_name",
            "version",
            "source_path",
            "source_url",
            "section",
        )
    ),
    Column("heading_path", ARRAY(Text), nullable=False),
    Column("text", Text, nullable=False),
    Column("parent_index", Integer, nullable=False),
    Column("source_start", Integer, nullable=False),
    Column("source_end", Integer, nullable=False),
    Column("parent_hash", Text, nullable=False),
    Column("markdown_prefix", Text, nullable=False),
    Column("markdown_suffix", Text, nullable=False),
    ForeignKeyConstraint(
        REVISION_COLUMNS, [f"document_ingestions.{name}" for name in REVISION_COLUMNS]
    ),
)
for expression in (
    func.length(parent_chunks.c.text) > 0,
    parent_chunks.c.parent_index >= 0,
    parent_chunks.c.source_start >= 0,
    parent_chunks.c.source_end > parent_chunks.c.source_start,
):
    parent_chunks.append_constraint(CheckConstraint(expression))
Index("parent_chunks_revision_idx", *(parent_chunks.c[name] for name in REVISION_COLUMNS))

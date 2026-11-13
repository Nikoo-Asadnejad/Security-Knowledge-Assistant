"""Create durable parents and ingestion state.

Keep these definitions frozen; subsequent schema changes belong in new revisions.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "001_parent_storage"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_ingestions",
        sa.Column("document_id", sa.Text, primary_key=True),
        sa.Column("document_hash", sa.Text, primary_key=True),
        sa.Column("chunking_config_hash", sa.Text, primary_key=True),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("expected_parents", sa.Integer, nullable=False),
        sa.Column("expected_children", sa.Integer, nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(sa.column("status").in_(("processing", "completed"))),
        sa.CheckConstraint(sa.column("expected_parents") > 0),
        sa.CheckConstraint(sa.column("expected_children") > 0),
    )
    op.create_table(
        "parent_chunks",
        sa.Column("id", sa.Uuid, primary_key=True),
        *(
            sa.Column(name, sa.Text, nullable=False)
            for name in (
                "document_id",
                "document_hash",
                "chunking_config_hash",
                "document_code",
                "title",
                "file_name",
                "version",
                "source_path",
                "source_url",
                "section",
            )
        ),
        sa.Column("heading_path", ARRAY(sa.Text), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("parent_index", sa.Integer, nullable=False),
        sa.Column("source_start", sa.Integer, nullable=False),
        sa.Column("source_end", sa.Integer, nullable=False),
        sa.Column("parent_hash", sa.Text, nullable=False),
        sa.Column("markdown_prefix", sa.Text, nullable=False),
        sa.Column("markdown_suffix", sa.Text, nullable=False),
        sa.CheckConstraint(sa.func.length(sa.column("text")) > 0),
        sa.CheckConstraint(sa.column("parent_index") >= 0),
        sa.CheckConstraint(sa.column("source_start") >= 0),
        sa.CheckConstraint(sa.column("source_end") > sa.column("source_start")),
        sa.ForeignKeyConstraint(
            ["document_id", "document_hash", "chunking_config_hash"],
            [
                "document_ingestions.document_id",
                "document_ingestions.document_hash",
                "document_ingestions.chunking_config_hash",
            ],
        ),
    )
    op.create_index(
        "parent_chunks_revision_idx",
        "parent_chunks",
        ["document_id", "document_hash", "chunking_config_hash"],
    )


def downgrade() -> None:
    op.drop_table("parent_chunks")
    op.drop_table("document_ingestions")

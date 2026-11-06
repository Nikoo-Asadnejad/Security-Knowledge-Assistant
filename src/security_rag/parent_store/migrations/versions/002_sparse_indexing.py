"""Track sparse coverage separately from dense/chunk revisions."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "002_sparse_indexing"
down_revision = "001_parent_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_ingestions", sa.Column("sparse_config_hash", sa.Text))
    op.add_column(
        "document_ingestions",
        sa.Column("sparse_status", sa.Text, nullable=False, server_default="processing"),
    )
    op.create_check_constraint(
        "sparse_status_valid",
        "document_ingestions",
        "sparse_status IN ('processing', 'completed')",
    )
    op.create_check_constraint(
        "sparse_completed_hash",
        "document_ingestions",
        "sparse_status != 'completed' OR sparse_config_hash IS NOT NULL",
    )
    op.create_table(
        "sparse_configuration",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("config", JSONB, nullable=False),
        sa.CheckConstraint("id = 1"),
    )


def downgrade() -> None:
    op.drop_table("sparse_configuration")
    op.drop_constraint("sparse_completed_hash", "document_ingestions")
    op.drop_constraint("sparse_status_valid", "document_ingestions")
    op.drop_column("document_ingestions", "sparse_status")
    op.drop_column("document_ingestions", "sparse_config_hash")

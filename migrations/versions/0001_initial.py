"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.String(64), sa.ForeignKey("conversations.id")),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id", sa.String(64), sa.ForeignKey("conversations.id"), nullable=True
        ),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("query_class", sa.String(32), nullable=False),
        sa.Column("requested_decision", sa.String(32), nullable=False),
        sa.Column("decision", sa.String(32), nullable=False),
        sa.Column("final_answer", sa.Text, nullable=False, server_default=""),
        sa.Column("confidence", sa.String(16), nullable=False, server_default="medium"),
        sa.Column("dissent_notes", sa.Text, nullable=False, server_default=""),
        sa.Column("disagreement", sa.Float, nullable=False, server_default="0"),
        sa.Column("degraded", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("timeout_partial", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "run_stages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.id")),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("detail", sa.JSON, nullable=True),
    )
    op.create_table(
        "run_proposers",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.id")),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("ok", sa.Boolean, nullable=False),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
    )


def downgrade() -> None:
    op.drop_table("run_proposers")
    op.drop_table("run_stages")
    op.drop_table("runs")
    op.drop_table("messages")
    op.drop_table("conversations")

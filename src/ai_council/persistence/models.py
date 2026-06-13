"""SQLAlchemy ORM schema. Portable types (JSON, not JSONB) so the same models
and migration apply to both PostgreSQL (production) and SQLite (local/tests)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON as SAJSON
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # correlation id
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True
    )
    query: Mapped[str] = mapped_column(Text)
    query_class: Mapped[str] = mapped_column(String(32))
    requested_decision: Mapped[str] = mapped_column(String(32))
    decision: Mapped[str] = mapped_column(String(32))
    final_answer: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[str] = mapped_column(String(16), default="medium")
    dissent_notes: Mapped[str] = mapped_column(Text, default="")
    disagreement: Mapped[float] = mapped_column(Float, default=0.0)
    degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    timeout_partial: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    stages: Mapped[list[RunStage]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunStage.seq"
    )
    proposers: Mapped[list[RunProposer]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class RunStage(Base):
    __tablename__ = "run_stages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    seq: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict[str, Any]] = mapped_column(SAJSON, default=dict)

    run: Mapped[Run] = relationship(back_populates="stages")


class RunProposer(Base):
    __tablename__ = "run_proposers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"))
    model: Mapped[str] = mapped_column(String(128))
    ok: Mapped[bool] = mapped_column(Boolean)
    text: Mapped[str] = mapped_column(Text, default="")
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    run: Mapped[Run] = relationship(back_populates="proposers")

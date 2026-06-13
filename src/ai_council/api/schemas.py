"""Request/response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    force_council: bool = False
    force_single: bool = False


class ChatResponse(BaseModel):
    correlation_id: str
    final_answer: str
    decision: str
    query_class: str
    confidence: str
    dissent_notes: str
    contributing_sources: list[str]
    disagreement: float
    degraded: bool
    timeout_partial: bool
    cost_usd: float
    latency_ms: float

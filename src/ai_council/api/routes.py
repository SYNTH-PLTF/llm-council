"""Public API routes. The /v1/chat endpoint drives the orchestrator."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request

from ai_council.api.schemas import ChatRequest, ChatResponse
from ai_council.council.orchestrator import Orchestrator, RunContext, RunResult

router = APIRouter(tags=["chat"])


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("orchestrator is not configured")
    return orchestrator


def _to_response(result: RunResult) -> ChatResponse:
    return ChatResponse(
        correlation_id=result.correlation_id,
        final_answer=result.final_answer,
        decision=result.decision,
        query_class=result.query_class,
        confidence=result.confidence,
        dissent_notes=result.dissent_notes,
        contributing_sources=result.contributing_sources,
        disagreement=result.disagreement,
        degraded=result.degraded,
        timeout_partial=result.timeout_partial,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
    )


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    cid = getattr(request.state, "correlation_id", None) or uuid.uuid4().hex
    ctx = RunContext(
        correlation_id=cid,
        query=payload.query,
        force_council=payload.force_council,
        force_single=payload.force_single,
    )
    result = await orchestrator.run(ctx)
    return _to_response(result)

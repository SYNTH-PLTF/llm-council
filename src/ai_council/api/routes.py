"""Public API routes: /v1/chat with rate limiting, idempotency, and budgets."""

from __future__ import annotations

import math
import uuid
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Request

from ai_council.api.auth import require_api_key
from ai_council.api.schemas import ChatRequest, ChatResponse
from ai_council.cache.redis_cache import RedisCache
from ai_council.council.orchestrator import Orchestrator, RunContext, RunResult
from ai_council.settings import get_config

router = APIRouter(tags=["chat"])


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("orchestrator is not configured")
    return orchestrator


def get_cache(request: Request) -> RedisCache | None:
    return getattr(request.app.state, "cache", None)


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
        flags=result.flags,
        stages=[stage.name for stage in result.stages],
        proposer_models=result.proposer_models,
    )


@router.post("/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    cache: RedisCache | None = Depends(get_cache),
    _api_key: str = Depends(require_api_key),
) -> ChatResponse:
    config = get_config()
    api_key = request.headers.get("X-API-Key", "anonymous")

    if cache is not None:
        allowed, retry_after = await cache.allow(
            api_key,
            rate_per_s=config.rate_limit.requests_per_second,
            capacity=config.rate_limit.burst,
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(math.ceil(retry_after))},
            )

    idempotency_key = request.headers.get("Idempotency-Key")
    if cache is not None and idempotency_key:
        stored = await cache.get_idempotent(idempotency_key)
        if stored is not None:
            return ChatResponse.model_validate_json(stored)

    today = date.today().isoformat()
    daily_remaining: float | None = None
    if cache is not None:
        spent = await cache.get_spend(api_key, day=today)
        daily_remaining = max(0.0, config.router.budgets.per_user_daily_usd - spent)

    cid = getattr(request.state, "correlation_id", None) or uuid.uuid4().hex
    ctx = RunContext(
        correlation_id=cid,
        query=payload.query,
        force_council=payload.force_council,
        force_single=payload.force_single,
        request_budget_usd=config.router.budgets.per_request_usd,
        user_daily_remaining_usd=daily_remaining,
    )
    result = await orchestrator.run(ctx)
    response = _to_response(result)

    if cache is not None:
        if idempotency_key:
            await cache.set_idempotent(idempotency_key, response.model_dump_json())
        await cache.add_spend(api_key, result.cost_usd, day=today)
    return response

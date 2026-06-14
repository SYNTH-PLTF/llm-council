"""Prometheus metrics for infrastructure monitoring.

These are deliberately separate from semantic quality (which lives in traces and
eval scores): latency, error rate, tokens, and cost for operational dashboards.
A dedicated registry avoids clobbering the global default and keeps tests clean.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "ai_council_requests_total", "Chat requests", ["decision"], registry=REGISTRY
)
ERRORS = Counter("ai_council_errors_total", "Errors", ["kind"], registry=REGISTRY)
LLM_CALLS = Counter(
    "ai_council_llm_calls_total", "LLM calls", ["model", "outcome"], registry=REGISTRY
)
LLM_TOKENS = Counter(
    "ai_council_llm_tokens_total", "LLM tokens", ["model", "kind"], registry=REGISTRY
)
COST = Counter("ai_council_cost_usd_total", "Total LLM cost (USD)", registry=REGISTRY)
REQUEST_LATENCY = Histogram(
    "ai_council_request_latency_seconds", "Request latency", ["decision"], registry=REGISTRY
)
STAGE_LATENCY = Histogram(
    "ai_council_stage_latency_seconds", "Stage latency", ["stage"], registry=REGISTRY
)


def record_llm_call(
    model: str,
    outcome: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    LLM_CALLS.labels(model=model, outcome=outcome).inc()
    if prompt_tokens:
        LLM_TOKENS.labels(model=model, kind="prompt").inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKENS.labels(model=model, kind="completion").inc(completion_tokens)
    if cost_usd:
        COST.inc(cost_usd)


def record_request(decision: str, latency_s: float) -> None:
    REQUESTS.labels(decision=decision).inc()
    REQUEST_LATENCY.labels(decision=decision).observe(latency_s)


def record_stage(stage: str, latency_s: float) -> None:
    STAGE_LATENCY.labels(stage=stage).observe(latency_s)


def record_error(kind: str) -> None:
    ERRORS.labels(kind=kind).inc()


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

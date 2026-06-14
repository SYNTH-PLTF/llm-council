"""Phase 9 verification: nested tracing spans, Prometheus metrics, /metrics."""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ai_council.api.app import create_app
from ai_council.council.orchestrator import Orchestrator, RunContext
from ai_council.gateway.client import LLMGateway
from ai_council.observability.metrics import render
from ai_council.observability.tracing import RecordingSpan, RecordingTracer
from ai_council.settings import AppConfig

LONG = (
    "please analyze in careful detail the long term economic and social "
    "consequences of this multifaceted policy decision and its many tradeoffs"
)
MODELS = ["P1", "P2", "P3", "CH"]


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {m: {"env_prefix": m} for m in MODELS},
            "router": {"router_model": "P1", "budgets": {"latency_budget_s": 30.0}},
            "council": {
                "proposers": ["P1", "P2", "P3"],
                "quorum": 2,
                "chairman": {"model": "CH"},
                "ranking": {"orderings_per_judge": 2},
                "debate": {"enabled": False},
            },
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
            "pricing_usd_per_1k_tokens": {m: {"input": 1.0, "output": 1.0} for m in MODELS},
        }
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def _council_completion() -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        short = str(kwargs["model"]).split("/")[-1]
        if "Rank ALL candidates" in content:
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}))
        if "final_answer" in content:
            return _resp(json.dumps({"final_answer": "SYNTH", "confidence": "high"}))
        return _resp(f"answer-{short}")

    return fn


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in MODELS:
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


def _find(node: RecordingSpan, name: str) -> RecordingSpan | None:
    if node.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found is not None:
            return found
    return None


async def test_recording_tracer_nests_spans() -> None:
    tracer = RecordingTracer()
    gw = LLMGateway(_cfg(), completion_fn=_council_completion(), sleep_fn=_no_sleep)
    orch = Orchestrator(_cfg(), gw, tracer=tracer, rng=random.Random(0))
    await orch.run(RunContext(correlation_id="t", query=LONG, force_council=True))

    names = set(tracer.names())
    assert {"request", "triage", "proposers", "ranking", "chairman", "llm"} <= names
    assert len(tracer.roots) == 1
    assert tracer.roots[0].name == "request"
    proposers = _find(tracer.roots[0], "proposers")
    assert proposers is not None
    assert any(c.name == "llm" for c in proposers.children)  # gateway spans nest under stage


async def test_prometheus_metrics_recorded() -> None:
    gw = LLMGateway(_cfg(), completion_fn=_council_completion(), sleep_fn=_no_sleep)
    orch = Orchestrator(_cfg(), gw, rng=random.Random(0))
    await orch.run(RunContext(correlation_id="t2", query=LONG, force_council=True))

    text = render()[0].decode()
    assert "ai_council_requests_total" in text
    assert 'decision="council"' in text
    assert "ai_council_llm_calls_total" in text


def test_metrics_endpoint_serves_prometheus() -> None:
    client = TestClient(create_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "ai_council_cost_usd_total" in resp.text

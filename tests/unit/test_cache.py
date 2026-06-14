"""Phase 10 verification: cache, idempotency, rate limiting, budgets (fakeredis)."""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest
from fakeredis import FakeAsyncRedis
from fastapi.testclient import TestClient

from ai_council.api.app import create_app
from ai_council.cache.redis_cache import RedisCache
from ai_council.council.orchestrator import Orchestrator, RunContext, RunResult
from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage
from ai_council.settings import AppConfig

LONG = (
    "please analyze in careful detail the long term economic and social "
    "consequences of this multifaceted policy decision and its many tradeoffs"
)


async def _no_sleep(_delay: float) -> None:
    return None


def _resp(text: str, *, ct: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=ct, total_tokens=10 + ct),
    )


# --- primitives -------------------------------------------------------------


async def test_cache_and_spend_primitives() -> None:
    cache = RedisCache(FakeAsyncRedis())
    assert await cache.get_exact("M", "hello", {"t": 0}) is None
    await cache.set_exact("M", "hello", {"t": 0}, "answer")
    assert await cache.get_exact("M", "hello", {"t": 0}) == "answer"
    assert await cache.get_exact("M", "  HELLO ", {"t": 0}) == "answer"  # normalized key

    await cache.set_idempotent("k", "stored")
    assert await cache.get_idempotent("k") == "stored"

    assert await cache.get_spend("u", day="2026-06-14") == 0.0
    await cache.add_spend("u", 0.5, day="2026-06-14")
    await cache.add_spend("u", 0.25, day="2026-06-14")
    assert await cache.get_spend("u", day="2026-06-14") == pytest.approx(0.75)


async def test_token_bucket_denies_then_refills() -> None:
    clock = {"t": 0.0}
    cache = RedisCache(FakeAsyncRedis(), now=lambda: clock["t"])
    assert (await cache.allow("k", rate_per_s=1.0, capacity=2))[0] is True
    assert (await cache.allow("k", rate_per_s=1.0, capacity=2))[0] is True
    allowed, retry = await cache.allow("k", rate_per_s=1.0, capacity=2)
    assert allowed is False
    assert retry > 0
    clock["t"] = 2.0  # refill
    assert (await cache.allow("k", rate_per_s=1.0, capacity=2))[0] is True


# --- exact cache short-circuits the provider --------------------------------


async def test_exact_cache_hit_avoids_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MA_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MA_API_KEY", "k")
    calls = {"n": 0}

    async def fn(**_: Any) -> Any:
        calls["n"] += 1
        return _resp("ANSWER")

    cfg = AppConfig.model_validate(
        {
            "models": {"A": {"env_prefix": "MA"}},
            "router": {"router_model": "A"},
            "council": {"proposers": ["A"], "chairman": {"model": "A"}},
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
            "cache": {"exact_match": True},
        }
    )
    gw = LLMGateway(cfg, completion_fn=fn, sleep_fn=_no_sleep, cache=RedisCache(FakeAsyncRedis()))
    msgs = [ChatMessage(role="user", content="hello")]
    first = await gw.complete("A", msgs)
    second = await gw.complete("A", msgs)
    assert first.text == "ANSWER"
    assert second.text == "ANSWER"
    assert second.from_cache is True
    assert calls["n"] == 1  # provider called once, second served from cache


# --- API: rate limit + idempotency ------------------------------------------


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: RunContext) -> RunResult:
        self.calls += 1
        return RunResult(correlation_id=ctx.correlation_id, decision="council", final_answer="ok")


def _client(orch: Any) -> TestClient:
    app = create_app()
    app.state.orchestrator = orch
    app.state.cache = RedisCache(FakeAsyncRedis())
    return TestClient(app)


def test_rate_limit_returns_429_with_retry_after() -> None:
    client = _client(_FakeOrchestrator())
    last_429 = None
    for _ in range(30):  # default burst is 20
        resp = client.post("/v1/chat", json={"query": "hello there friend"})
        if resp.status_code == 429:
            last_429 = resp
    assert last_429 is not None
    assert any(k.lower() == "retry-after" for k in last_429.headers)


def test_idempotent_replay_skips_orchestrator() -> None:
    orch = _FakeOrchestrator()
    client = _client(orch)
    headers = {"Idempotency-Key": "abc-123"}
    first = client.post("/v1/chat", json={"query": "hello there friend"}, headers=headers)
    second = client.post("/v1/chat", json={"query": "hello there friend"}, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["final_answer"] == second.json()["final_answer"]
    assert orch.calls == 1  # second request served from the idempotency store


# --- orchestrator mid-run budget re-check -----------------------------------


def _council_completion() -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        short = str(kwargs["model"]).split("/")[-1]
        if "Rank ALL candidates" in content:
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}), ct=100_000)
        if "final_answer" in content:
            return _resp(json.dumps({"final_answer": "SYNTH"}), ct=100_000)
        return _resp(f"answer-{short}", ct=100_000)  # heavy proposer output

    return fn


def _budget_cfg() -> AppConfig:
    models = ["P1", "P2", "P3", "CH"]
    return AppConfig.model_validate(
        {
            "models": {m: {"env_prefix": m} for m in models},
            "router": {"router_model": "P1", "budgets": {"latency_budget_s": 30.0}},
            "council": {
                "proposers": ["P1", "P2", "P3"],
                "quorum": 2,
                "chairman": {"model": "CH"},
                "ranking": {"orderings_per_judge": 2},
                "debate": {"enabled": False},
            },
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
            "pricing_usd_per_1k_tokens": {m: {"input": 1.0, "output": 1.0} for m in models},
        }
    )


async def test_budget_recheck_skips_chairman(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("P1", "P2", "P3", "CH"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "k")
    cfg = _budget_cfg()
    gw = LLMGateway(cfg, completion_fn=_council_completion(), sleep_fn=_no_sleep)
    orch = Orchestrator(cfg, gw, rng=random.Random(0))
    # Budget passes the router estimate gate but the heavy actual cost trips the
    # orchestrator's pre-chairman re-check.
    result = await orch.run(
        RunContext(correlation_id="t", query=LONG, force_council=True, request_budget_usd=20.0)
    )
    assert result.decision == "council"
    assert result.degraded is True
    assert result.final_answer.startswith("answer-")  # top candidate, chairman skipped
    assert not any(s.name == "chairman" for s in result.stages)
    assert any(s.name == "budget_capped" for s in result.stages)

"""Phase 2 verification: rule overrides, classifier fallback, budget gating."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.gateway.client import LLMGateway
from ai_council.router.triage import LLMRouter, RouteRequest
from ai_council.settings import AppConfig

LONG_QUERY = (
    "please analyze the long term geopolitical and economic consequences "
    "of this complex multi factor scenario in careful detail"
)


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {"A": {"env_prefix": "MA"}, "B": {"env_prefix": "MB"}},
            "router": {
                "router_model": "A",
                "rules": {"force_single_if_tokens_lt": 12, "force_council_on_user_flag": True},
            },
            "council": {"proposers": ["A"], "chairman": {"model": "A"}},
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
            "pricing_usd_per_1k_tokens": {"A": {"input": 1.0, "output": 2.0}},
        }
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MA_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("MA_API_KEY", "test-key")


def _router(completion_fn: Any) -> LLMRouter:
    gw = LLMGateway(_cfg(), completion_fn=completion_fn, sleep_fn=_no_sleep)
    return LLMRouter(_cfg(), gw, prompt="Classify: {{query}}")


async def test_force_single_skips_classifier() -> None:
    calls = {"n": 0}

    async def counted(**_: Any) -> Any:
        calls["n"] += 1
        return _resp('{"query_class": "high_stakes"}')

    out = await _router(counted).route(RouteRequest(query=LONG_QUERY, force_single=True))
    assert out.decision == "single_model"
    assert out.source == "rules"
    assert calls["n"] == 0  # classifier never consulted


async def test_force_council_overrides() -> None:
    async def counted(**_: Any) -> Any:
        return _resp('{"query_class": "trivial"}')

    out = await _router(counted).route(RouteRequest(query=LONG_QUERY, force_council=True))
    assert out.decision == "council"
    assert out.source == "rules"


async def test_short_query_routes_single() -> None:
    async def boom(**_: Any) -> Any:
        raise AssertionError("classifier should not be called for short queries")

    out = await _router(boom).route(RouteRequest(query="hi there"))
    assert out.decision == "single_model"
    assert out.query_class == "trivial"
    assert out.source == "rules"


async def test_classifier_high_stakes_to_council() -> None:
    async def classify(**_: Any) -> Any:
        return _resp('{"query_class": "high_stakes", "decision": "council", "reason": "hard"}')

    out = await _router(classify).route(RouteRequest(query=LONG_QUERY))
    assert out.query_class == "high_stakes"
    assert out.decision == "council"
    assert out.source == "classifier"


async def test_classifier_garbage_falls_back_safely() -> None:
    async def garbage(**_: Any) -> Any:
        return _resp("I am not JSON at all, sorry!")

    out = await _router(garbage).route(RouteRequest(query=LONG_QUERY))
    assert out.query_class == "standard"
    assert out.decision == "single_model"
    assert out.source == "fallback"


async def test_budget_gate_downgrades_council() -> None:
    async def classify(**_: Any) -> Any:
        return _resp('{"query_class": "high_stakes"}')

    out = await _router(classify).route(
        RouteRequest(query=LONG_QUERY, budget_remaining_usd=0.5)
    )
    assert out.decision == "single_model"
    assert out.source == "policy"
    assert "budget" in out.reason

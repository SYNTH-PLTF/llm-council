"""Phase 7 verification: end-to-end orchestration across every route path.

Uses the REAL router/ranker/chairman/voting logic with a single mocked gateway
that branches on prompt-content markers, so each path is exercised genuinely.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.orchestrator import Orchestrator, RunContext
from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import TransientError
from ai_council.settings import AppConfig

LONG = (
    "please analyze in careful detail the long term economic and social "
    "consequences of this multifaceted policy decision and its many tradeoffs"
)
MODELS = ["P1", "P2", "P3", "CH"]


def _cfg(
    *, debate_enabled: bool = True, debate_threshold: float = 0.4, latency_budget: float = 30.0
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {m: {"env_prefix": m} for m in MODELS},
            "router": {
                "router_model": "P1",
                "budgets": {
                    "latency_budget_s": latency_budget,
                    "per_request_usd": 100.0,
                    "per_user_daily_usd": 100.0,
                },
            },
            "council": {
                "proposers": ["P1", "P2", "P3"],
                "quorum": 2,
                "chairman": {"model": "CH", "max_output_tokens": 256},
                "ranking": {"orderings_per_judge": 2},
                "debate": {
                    "enabled": debate_enabled,
                    "threshold": debate_threshold,
                    "max_rounds": 1,
                },
            },
            "class_routing": {
                "trivial": "single_model",
                "standard": "single_model",
                "high_stakes": "council",
                "verifiable_reasoning": "council_with_voting",
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


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in MODELS:
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


def _completion(
    *,
    triage_class: str = "high_stakes",
    answer: Callable[[str], str] = lambda short: f"answer-{short}",
    verdict: dict[str, Any] | None = None,
    fail_generation: frozenset[str] = frozenset(),
    chairman_sleep: float = 0.0,
) -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        short = str(kwargs["model"]).split("/")[-1]
        if "query_class" in content:  # triage prompt
            return _resp(json.dumps({"query_class": triage_class, "decision": "council"}))
        if "Rank ALL candidates" in content:  # ranker judge prompt
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}))
        if "final_answer" in content:  # chairman prompt
            if chairman_sleep:
                await asyncio.sleep(chairman_sleep)
            return _resp(json.dumps(verdict or {"final_answer": "SYNTHESIS", "confidence": "high"}))
        if short in fail_generation:  # proposer / single-model generation
            raise TransientError(f"{short} down")
        return _resp(answer(short))

    return fn


def _orch(completion: Any, cfg: AppConfig) -> Orchestrator:
    gw = LLMGateway(cfg, completion_fn=completion, sleep_fn=_no_sleep)
    return Orchestrator(cfg, gw, rng=random.Random(0))


def _ctx(query: str = LONG, **kw: Any) -> RunContext:
    return RunContext(correlation_id="t", query=query, **kw)


async def test_single_model_path() -> None:
    orch = _orch(_completion(answer=lambda short: "SHORT-ANSWER"), _cfg())
    result = await orch.run(_ctx(query="hi there"))  # short query -> rule -> single
    assert result.decision == "single_model"
    assert result.query_class == "trivial"
    assert result.final_answer == "SHORT-ANSWER"


async def test_council_consensus_path() -> None:
    orch = _orch(_completion(triage_class="high_stakes"), _cfg(debate_enabled=False))
    result = await orch.run(_ctx())
    assert result.decision == "council"
    assert result.final_answer == "SYNTHESIS"
    assert result.confidence == "high"
    assert {s.name for s in result.stages} >= {"triage", "proposers", "ranking", "chairman"}
    assert result.cost_usd > 0  # cost captured across every stage


async def test_council_debate_path() -> None:
    orch = _orch(_completion(triage_class="high_stakes"), _cfg(debate_threshold=-1.0))
    result = await orch.run(_ctx())
    assert result.decision == "council"
    assert any(s.name == "debate" for s in result.stages)


async def test_council_vote_path() -> None:
    answers = {"P1": "answer: 42", "P2": "answer: 42", "P3": "answer: 7"}
    orch = _orch(
        _completion(triage_class="verifiable_reasoning", answer=lambda short: answers[short]),
        _cfg(),
    )
    result = await orch.run(_ctx())
    assert result.decision == "council_with_voting"
    assert "42" in result.final_answer
    names = {s.name for s in result.stages}
    assert "voting" in names
    assert "ranking" not in names  # verifiable path never invokes the LLM ranker
    assert "chairman" not in names


async def test_degraded_path() -> None:
    orch = _orch(
        _completion(fail_generation=frozenset({"P2", "P3"}), answer=lambda short: "KEPT"),
        _cfg(),
    )
    result = await orch.run(_ctx(force_council=True))
    assert result.degraded is True
    assert result.decision == "single_model"
    assert result.final_answer == "KEPT"


async def test_budget_timeout_returns_partial() -> None:
    orch = _orch(
        _completion(triage_class="high_stakes", chairman_sleep=0.5),
        _cfg(debate_enabled=False, latency_budget=0.1),
    )
    result = await orch.run(_ctx())
    assert result.timeout_partial is True
    # best result so far is a top-ranked proposer answer (chairman never finished)
    assert result.final_answer.startswith("answer-")

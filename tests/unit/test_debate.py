"""Phase 5 verification: debate gating, convergence, and the hard round cap."""

from __future__ import annotations

import random
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.debate import DebateResult, converged, run_debate, should_debate
from ai_council.council.ranking import Candidate, RankingResult
from ai_council.gateway.client import LLMGateway
from ai_council.settings import AppConfig


def _cfg(*, max_rounds: int = 1, enabled: bool = True, threshold: float = 0.4) -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {"P1": {"env_prefix": "P1"}, "P2": {"env_prefix": "P2"}},
            "router": {"router_model": "P1"},
            "council": {
                "proposers": ["P1", "P2"],
                "chairman": {"model": "P1"},
                "debate": {"enabled": enabled, "threshold": threshold, "max_rounds": max_rounds},
            },
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
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
    for prefix in ("P1", "P2"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


def _gw(text_fn: Callable[[str], str]) -> LLMGateway:
    async def fn(**kwargs: Any) -> Any:
        short = str(kwargs["model"]).split("/")[-1]
        return _resp(text_fn(short))

    return LLMGateway(_cfg(), completion_fn=fn, sleep_fn=_no_sleep)


async def _rerank(cands: list[Candidate]) -> RankingResult:
    return RankingResult(ordering=[c.id for c in cands])


def test_should_debate_rules() -> None:
    cfg = _cfg(threshold=0.4)
    assert should_debate(cfg, "council", 0.5) is True
    assert should_debate(cfg, "council", 0.3) is False  # below threshold
    assert should_debate(cfg, "council_with_voting", 0.9) is False  # voting path
    assert should_debate(_cfg(enabled=False), "council", 0.9) is False


def test_converged_detects_stability() -> None:
    base = [Candidate(id="P1", text="hello world"), Candidate(id="P2", text="foo")]
    same = [Candidate(id="P1", text="hello   world"), Candidate(id="P2", text="FOO")]
    diff = [Candidate(id="P1", text="changed"), Candidate(id="P2", text="foo")]
    assert converged(base, same) is True
    assert converged(base, diff) is False


async def test_run_debate_respects_round_cap() -> None:
    cands = [Candidate(id="P1", text="orig-P1"), Candidate(id="P2", text="orig-P2")]
    ranking = RankingResult(ordering=["P1", "P2"])
    gw = _gw(lambda short: f"new-{short}")  # always changes -> never converges
    res = await run_debate(
        gw, _cfg(max_rounds=1), "q", cands, ranking, rerank=_rerank, rng=random.Random(0)
    )
    assert isinstance(res, DebateResult)
    assert res.rounds == 1
    assert res.converged is False
    assert {c.text for c in res.candidates} == {"new-P1", "new-P2"}


async def test_run_debate_stops_early_on_convergence() -> None:
    cands = [Candidate(id="P1", text="orig-P1"), Candidate(id="P2", text="orig-P2")]
    ranking = RankingResult(ordering=["P1", "P2"])
    gw = _gw(lambda short: f"orig-{short}")  # returns originals -> converges
    res = await run_debate(
        gw, _cfg(max_rounds=3), "q", cands, ranking, rerank=_rerank, rng=random.Random(0)
    )
    assert res.rounds == 1  # stopped after round 1 despite a cap of 3
    assert res.converged is True

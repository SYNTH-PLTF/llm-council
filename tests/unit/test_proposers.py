"""Phase 3 verification: parallel proposers, quorum, graceful degradation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.proposers import run_stage1
from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage, TransientError
from ai_council.settings import AppConfig

MSG = [ChatMessage(role="user", content="q")]


def _cfg(quorum: int = 2) -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {
                "A": {"env_prefix": "MA"},
                "B": {"env_prefix": "MB"},
                "C": {"env_prefix": "MC"},
            },
            "router": {"router_model": "A"},
            "council": {"proposers": ["A", "B", "C"], "quorum": quorum, "chairman": {"model": "A"}},
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
            "pricing_usd_per_1k_tokens": {
                "A": {"input": 1.0, "output": 1.0},
                "B": {"input": 1.0, "output": 1.0},
                "C": {"input": 1.0, "output": 1.0},
            },
        }
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _completion(fail: set[str]) -> Any:
    async def fn(**kwargs: Any) -> Any:
        short = str(kwargs["model"]).split("/")[-1]
        if short in fail:
            raise TransientError(f"{short} down")
        choice = SimpleNamespace(
            message=SimpleNamespace(content=f"ans-{short}"), finish_reason="stop"
        )
        return SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    return fn


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("MA", "MB", "MC"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


def _gw(fail: set[str]) -> LLMGateway:
    return LLMGateway(_cfg(), completion_fn=_completion(fail), sleep_fn=_no_sleep)


async def test_all_succeed() -> None:
    res = await run_stage1(_gw(set()), _cfg(), MSG)
    assert res.degraded is False
    assert res.single is None
    assert res.round.met is True
    assert len(res.round.successes) == 3
    assert all(o.cost_usd == pytest.approx(0.02) for o in res.round.successes)
    assert {o.text for o in res.round.successes} == {"ans-A", "ans-B", "ans-C"}


async def test_one_fails_quorum_met() -> None:
    res = await run_stage1(_gw({"C"}), _cfg(), MSG)
    assert res.degraded is False
    assert res.round.met is True
    assert len(res.round.successes) == 2
    assert len(res.round.failures) == 1
    assert res.round.failures[0].model == "C"
    assert res.round.failures[0].ok is False


async def test_two_fail_below_quorum_degrades_to_single() -> None:
    res = await run_stage1(_gw({"B", "C"}), _cfg(), MSG)
    assert res.degraded is True
    assert res.round.met is False
    assert res.single is not None
    assert res.single.model == "A"
    assert res.single.ok is True
    assert res.single.text == "ans-A"


async def test_all_fail_degrades_gracefully_without_raising() -> None:
    res = await run_stage1(_gw({"A", "B", "C"}), _cfg(), MSG)
    assert res.degraded is True
    assert res.single is not None
    assert res.single.ok is False  # nothing succeeded, but no exception escaped

"""Phase 6 verification: structured synthesis, repair-retry, streaming."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.chairman import Chairman, ChairmanError, ChairmanVerdict, stream_text
from ai_council.council.ranking import Candidate
from ai_council.gateway.client import LLMGateway
from ai_council.settings import AppConfig

TOP = [Candidate(id="a", text="answer one"), Candidate(id="b", text="answer two")]


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {"C1": {"env_prefix": "C1"}},
            "router": {"router_model": "C1"},
            "council": {"proposers": ["C1"], "chairman": {"model": "C1", "max_output_tokens": 512}},
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
    monkeypatch.setenv("C1_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("C1_API_KEY", "test-key")


def _chair(completion: Any) -> Chairman:
    gw = LLMGateway(_cfg(), completion_fn=completion, sleep_fn=_no_sleep)
    return Chairman(_cfg(), gw, prompt="Q:{{query}} C:{{top_k_candidates}}")


async def test_synthesize_valid() -> None:
    payload = json.dumps(
        {
            "final_answer": "the answer",
            "confidence": "high",
            "dissent_notes": "none",
            "contributing_sources": ["1"],
        }
    )

    async def ok(**_: Any) -> Any:
        return _resp(payload)

    verdict = await _chair(ok).synthesize("q", TOP)
    assert isinstance(verdict, ChairmanVerdict)
    assert verdict.final_answer == "the answer"
    assert verdict.confidence == "high"
    assert verdict.contributing_sources == ["1"]


async def test_synthesize_repairs_then_succeeds() -> None:
    calls = {"n": 0}
    good = json.dumps({"final_answer": "ok"})

    async def flaky(**_: Any) -> Any:
        calls["n"] += 1
        return _resp("not json at all" if calls["n"] == 1 else good)

    verdict = await _chair(flaky).synthesize("q", TOP)
    assert verdict.final_answer == "ok"
    assert verdict.confidence == "medium"  # schema default
    assert calls["n"] == 2  # one repair attempt


async def test_synthesize_fails_typed_after_repair() -> None:
    calls = {"n": 0}

    async def bad(**_: Any) -> Any:
        calls["n"] += 1
        return _resp("still not json")

    with pytest.raises(ChairmanError):
        await _chair(bad).synthesize("q", TOP)
    assert calls["n"] == 2  # repaired exactly once, then gave up


async def test_stream_text_chunks() -> None:
    chunks = [c async for c in stream_text("hello world", chunk_size=4)]
    assert "".join(chunks) == "hello world"
    assert len(chunks) == 3

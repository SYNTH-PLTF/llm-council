"""Phase 1 verification: gateway retry / fallback / breaker / cost, fully
mocked - no network, no litellm import."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.gateway.client import CircuitBreaker, LLMGateway
from ai_council.gateway.models import (
    AllAttemptsFailed,
    ChatMessage,
    CircuitOpenError,
    NonRetryableError,
    TransientError,
)
from ai_council.settings import AppConfig

MSG = [ChatMessage(role="user", content="hi")]


def _cfg(*, max_retries: int = 1, with_fallback: bool = True, fail_threshold: int = 2) -> AppConfig:
    model_a: dict[str, Any] = {"env_prefix": "MA", "litellm_provider": "openai"}
    if with_fallback:
        model_a["fallback"] = "B"
    return AppConfig.model_validate(
        {
            "models": {
                "A": model_a,
                "B": {"env_prefix": "MB", "litellm_provider": "openai"},
            },
            "router": {"router_model": "A"},
            "council": {"proposers": ["A"], "chairman": {"model": "A"}},
            "gateway": {
                "max_retries": max_retries,
                "backoff_base_s": 0.0,
                "circuit_breaker": {"fail_threshold": fail_threshold, "reset_after_s": 100},
            },
            "pricing_usd_per_1k_tokens": {
                "A": {"input": 1.0, "output": 2.0},
                "B": {"input": 0.0, "output": 0.0},
            },
        }
    )


def _resp(text: str = "hi", pt: int = 10, ct: int = 5, finish: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
    )


async def _no_sleep(_delay: float) -> None:
    return None


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("MA", "MB"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


async def test_success_and_cost() -> None:
    async def ok(**_: Any) -> Any:
        return _resp(text="answer", pt=10, ct=5)

    gw = LLMGateway(_cfg(), completion_fn=ok, sleep_fn=_no_sleep)
    result = await gw.complete("A", MSG)

    assert result.text == "answer"
    assert result.finish_reason == "stop"
    assert result.attempts == 1
    assert result.used_fallback is False
    assert result.usage.total_tokens == 15
    # 10/1000 * 1.0 + 5/1000 * 2.0 = 0.01 + 0.01 = 0.02
    assert result.cost_usd == pytest.approx(0.02)


async def test_timeout_then_retry_succeeds() -> None:
    calls = {"n": 0}

    async def flaky(**_: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("slow")
        return _resp(text="recovered")

    gw = LLMGateway(_cfg(max_retries=2), completion_fn=flaky, sleep_fn=_no_sleep)
    result = await gw.complete("A", MSG)

    assert result.text == "recovered"
    assert result.attempts == 2
    assert calls["n"] == 2


async def test_4xx_is_not_retried() -> None:
    calls = {"n": 0}

    async def bad(**_: Any) -> Any:
        calls["n"] += 1
        raise _StatusError("bad request", 400)

    gw = LLMGateway(_cfg(max_retries=3, with_fallback=False), completion_fn=bad, sleep_fn=_no_sleep)
    with pytest.raises(NonRetryableError):
        await gw.complete("A", MSG)
    assert calls["n"] == 1  # never retried


async def test_exhausted_primary_falls_back() -> None:
    async def primary_fails(**kwargs: Any) -> Any:
        if str(kwargs["model"]).endswith("/A"):
            raise TransientError("primary down")
        return _resp(text="from-B")

    gw = LLMGateway(_cfg(max_retries=1), completion_fn=primary_fails, sleep_fn=_no_sleep)
    result = await gw.complete("A", MSG)

    assert result.text == "from-B"
    assert result.used_fallback is True


async def test_fallback_also_fails_raises_typed() -> None:
    async def all_fail(**_: Any) -> Any:
        raise TransientError("everything down")

    gw = LLMGateway(_cfg(max_retries=1), completion_fn=all_fail, sleep_fn=_no_sleep)
    with pytest.raises(AllAttemptsFailed):
        await gw.complete("A", MSG)


async def test_circuit_opens_and_fast_fails() -> None:
    async def always_down(**_: Any) -> Any:
        raise TransientError("down")

    gw = LLMGateway(
        _cfg(max_retries=0, with_fallback=False, fail_threshold=2),
        completion_fn=always_down,
        sleep_fn=_no_sleep,
    )
    for _ in range(2):
        with pytest.raises(AllAttemptsFailed):
            await gw.complete("A", MSG)
    # Breaker is now open -> next call fast-fails without calling the provider.
    with pytest.raises(CircuitOpenError):
        await gw.complete("A", MSG)


async def test_streaming_yields_pieces() -> None:
    async def stream_fn(**_: Any) -> Any:
        async def gen() -> Any:
            for piece in ("he", "llo"):
                choice = SimpleNamespace(delta=SimpleNamespace(content=piece))
                yield SimpleNamespace(choices=[choice])

        return gen()

    gw = LLMGateway(_cfg(), completion_fn=stream_fn, sleep_fn=_no_sleep)
    chunks = [c async for c in gw.stream("A", MSG)]
    assert "".join(chunks) == "hello"


def test_circuit_breaker_half_open_after_cooldown() -> None:
    clock = {"now": 0.0}
    cb = CircuitBreaker(fail_threshold=2, reset_after_s=10.0, clock=lambda: clock["now"])

    cb.record_failure()
    assert cb.allow() is True
    cb.record_failure()
    assert cb.state == "open"
    assert cb.allow() is False

    clock["now"] = 10.0
    assert cb.allow() is True
    assert cb.state == "half_open"
    cb.record_success()
    assert cb.state == "closed"


class _StatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code

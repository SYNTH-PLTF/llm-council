"""LiteLLM-backed gateway.

Responsibilities, all centralized so no caller has to think about them:
  * resolve each model's OpenAI-compatible endpoint (Target URI + key) from env
  * per-call timeout
  * bounded retries with jittered exponential backoff, on TRANSIENT errors only
    (429 / 5xx / timeout / connection); 4xx validation errors are never retried
  * per-model fallback
  * per-provider circuit breaker (open after N consecutive failures, half-open
    probe after a cooldown)
  * capture usage + cost + latency + finish_reason into a typed ProviderResult

The provider call is injectable (``completion_fn``) so tests never hit a network
and never import litellm.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar, Token
from typing import Any, Literal

from ai_council.gateway.models import (
    AllAttemptsFailed,
    ChatMessage,
    CircuitOpenError,
    GatewayError,
    NonRetryableError,
    ProviderResult,
    TransientError,
    Usage,
)
from ai_council.settings import AppConfig, ModelSpec
from ai_council.telemetry.logging import get_logger

log = get_logger("gateway")

CompletionFn = Callable[..., Awaitable[Any]]
SleepFn = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]

_run_cost: ContextVar[list[float] | None] = ContextVar("ai_council_run_cost", default=None)


class cost_capture:
    """Accumulate the cost_usd of every gateway call made in this async context.

    Works across asyncio.gather and asyncio.wait_for: child tasks copy the
    context and share the same accumulator list, so concurrent stage calls all
    add into one per-run total.
    """

    def __init__(self) -> None:
        self._token: Token[list[float] | None] | None = None
        self.costs: list[float] = []

    def __enter__(self) -> cost_capture:
        self._token = _run_cost.set(self.costs)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._token is not None:
            _run_cost.reset(self._token)

    @property
    def total(self) -> float:
        return sum(self.costs)


_TRANSIENT_NAMES = {
    "RateLimitError",
    "Timeout",
    "APITimeoutError",
    "APIConnectionError",
    "ServiceUnavailableError",
    "InternalServerError",
}
_FATAL_NAMES = {
    "BadRequestError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "UnprocessableEntityError",
    "ContentPolicyViolationError",
}


def classify_error(exc: BaseException) -> Literal["transient", "fatal"]:
    """Decide whether an exception from the provider is worth retrying."""
    if isinstance(exc, NonRetryableError):
        return "fatal"
    if isinstance(exc, (TransientError, TimeoutError)):
        return "transient"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (408, 409, 429) or status >= 500:
            return "transient"
        if 400 <= status < 500:
            return "fatal"
    name = type(exc).__name__
    if name in _TRANSIENT_NAMES:
        return "transient"
    if name in _FATAL_NAMES:
        return "fatal"
    return "fatal"


class _Exhausted(Exception):
    """Internal: one candidate model used up its retries on transient errors."""

    def __init__(self, last: Exception) -> None:
        super().__init__(str(last))
        self.last = last


class CircuitBreaker:
    """Per-provider breaker: closed -> open (after N fails) -> half_open probe."""

    def __init__(
        self, fail_threshold: int, reset_after_s: float, clock: Clock = time.monotonic
    ) -> None:
        self._threshold = max(1, fail_threshold)
        self._reset_after = reset_after_s
        self._clock = clock
        self.failures = 0
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.state == "open":
            opened = self.opened_at if self.opened_at is not None else self._clock()
            if self._clock() - opened >= self._reset_after:
                self.state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.state = "closed"
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.state == "half_open" or self.failures >= self._threshold:
            self.state = "open"
            self.opened_at = self._clock()


class LLMGateway:
    def __init__(
        self,
        config: AppConfig,
        *,
        completion_fn: CompletionFn | None = None,
        sleep_fn: SleepFn | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self._config = config
        self._completion = completion_fn or _litellm_acompletion
        self._sleep = sleep_fn or asyncio.sleep
        self._clock = clock
        self._cb_cfg = config.gateway.circuit_breaker
        self._breakers: dict[str, CircuitBreaker] = {}

    def _breaker(self, model: str) -> CircuitBreaker:
        breaker = self._breakers.get(model)
        if breaker is None:
            breaker = CircuitBreaker(
                self._cb_cfg.fail_threshold, self._cb_cfg.reset_after_s, self._clock
            )
            self._breakers[model] = breaker
        return breaker

    def _resolve_endpoint(self, spec: ModelSpec) -> tuple[str, str, str]:
        prefix = spec.env_prefix
        base = os.environ.get(f"{prefix}_BASE_URL")
        key = os.environ.get(f"{prefix}_API_KEY")
        deployment = os.environ.get(f"{prefix}_MODEL") or ""
        if not base or not key:
            raise GatewayError(
                f"missing endpoint env for '{prefix}': "
                f"set {prefix}_BASE_URL and {prefix}_API_KEY"
            )
        return base, key, deployment

    def _cost(self, model: str, usage: Usage) -> float:
        price = self._config.pricing_usd_per_1k_tokens.get(model)
        if price is None:
            return 0.0
        return (
            (usage.prompt_tokens / 1000.0) * price.input
            + (usage.completion_tokens / 1000.0) * price.output
        )

    async def complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ProviderResult:
        spec = self._config.require_model(model)
        candidates: list[tuple[str, ModelSpec, bool]] = [(model, spec, False)]
        if spec.fallback:
            candidates.append((spec.fallback, self._config.require_model(spec.fallback), True))

        last_exc: Exception | None = None
        any_allowed = False
        for name, cand_spec, is_fallback in candidates:
            breaker = self._breaker(name)
            if not breaker.allow():
                log.warning("gateway.circuit_open", model=name)
                continue
            any_allowed = True
            try:
                return await self._attempt(
                    name,
                    cand_spec,
                    breaker,
                    messages,
                    is_fallback,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                    timeout_s=timeout_s,
                )
            except NonRetryableError:
                raise
            except _Exhausted as exc:
                last_exc = exc.last
                continue
        if not any_allowed:
            raise CircuitOpenError(f"circuit open for '{model}' and fallback")
        raise AllAttemptsFailed(f"all attempts failed for '{model}'", last_error=last_exc)

    async def _attempt(
        self,
        model: str,
        spec: ModelSpec,
        breaker: CircuitBreaker,
        messages: list[ChatMessage],
        is_fallback: bool,
        *,
        max_tokens: int | None,
        temperature: float | None,
        response_format: dict[str, Any] | None,
        timeout_s: float | None,
    ) -> ProviderResult:
        gw = self._config.gateway
        base, key, deployment = self._resolve_endpoint(spec)
        payload_model = f"{spec.litellm_provider}/{deployment or model}"
        timeout = timeout_s if timeout_s is not None else gw.request_timeout_s
        payload = [m.model_dump() for m in messages]
        attempt = 0
        while True:
            attempt += 1
            started = self._clock()
            try:
                resp = await asyncio.wait_for(
                    self._completion(
                        model=payload_model,
                        messages=payload,
                        api_base=base,
                        api_key=key,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_format=response_format,
                    ),
                    timeout=timeout,
                )
            except Exception as exc:
                if classify_error(exc) == "fatal":
                    raise NonRetryableError(
                        str(exc), status_code=getattr(exc, "status_code", None)
                    ) from exc
                if attempt <= gw.max_retries:
                    delay = gw.backoff_base_s * (2 ** (attempt - 1))
                    await self._sleep(random.uniform(0, delay) if delay > 0 else 0.0)
                    log.info("gateway.retry", model=model, attempt=attempt)
                    continue
                breaker.record_failure()
                raise _Exhausted(exc) from exc
            latency_ms = (self._clock() - started) * 1000.0
            text, finish, usage = _extract(resp)
            breaker.record_success()
            cost = self._cost(model, usage)
            sink = _run_cost.get()
            if sink is not None:
                sink.append(cost)
            return ProviderResult(
                model=model,
                text=text,
                finish_reason=finish,
                usage=usage,
                cost_usd=cost,
                latency_ms=latency_ms,
                attempts=attempt,
                used_fallback=is_fallback,
            )

    async def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[str]:
        spec = self._config.require_model(model)
        base, key, deployment = self._resolve_endpoint(spec)
        payload_model = f"{spec.litellm_provider}/{deployment or model}"
        resp = await self._completion(
            model=payload_model,
            messages=[m.model_dump() for m in messages],
            api_base=base,
            api_key=key,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in resp:
            piece = _extract_delta(chunk)
            if piece:
                yield piece


def _extract(resp: Any) -> tuple[str, str | None, Usage]:
    choice = resp.choices[0]
    message = getattr(choice, "message", None)
    text = getattr(message, "content", None) or ""
    finish = getattr(choice, "finish_reason", None)
    raw_usage = getattr(resp, "usage", None)
    usage = Usage(
        prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(raw_usage, "total_tokens", 0) or 0),
    )
    return text, finish, usage


def _extract_delta(chunk: Any) -> str:
    try:
        delta = getattr(chunk.choices[0], "delta", None)
        return getattr(delta, "content", None) or ""
    except (AttributeError, IndexError):
        return ""


async def _litellm_acompletion(**kwargs: Any) -> Any:
    import litellm

    clean = {k: v for k, v in kwargs.items() if v is not None}
    return await litellm.acompletion(**clean)

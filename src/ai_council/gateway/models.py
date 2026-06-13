"""Typed inputs, outputs, and errors for the LLM gateway.

Nothing outside the gateway passes raw provider dicts around; everything crosses
this boundary as a pydantic model or a typed exception.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ProviderResult(BaseModel):
    """The single typed result of one gateway call (after retries/fallback)."""

    model: str
    text: str
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    attempts: int = 1
    used_fallback: bool = False
    from_cache: bool = False


class GatewayError(Exception):
    """Base class for all gateway failures."""


class NonRetryableError(GatewayError):
    """A 4xx-class error (bad request, auth, not found). Never retried."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TransientError(GatewayError):
    """A retryable error (429, 5xx, timeout, connection)."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CircuitOpenError(GatewayError):
    """The circuit breaker for the model (and any fallback) is open."""


class AllAttemptsFailed(GatewayError):
    """Primary and fallback both exhausted their retries."""

    def __init__(self, message: str, *, last_error: Exception | None = None) -> None:
        super().__init__(message)
        self.last_error = last_error

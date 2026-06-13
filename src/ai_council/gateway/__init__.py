"""LLM gateway: the single boundary between the app and any model provider."""

from ai_council.gateway.client import CircuitBreaker, LLMGateway, classify_error
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

__all__ = [
    "AllAttemptsFailed",
    "ChatMessage",
    "CircuitBreaker",
    "CircuitOpenError",
    "GatewayError",
    "LLMGateway",
    "NonRetryableError",
    "ProviderResult",
    "TransientError",
    "Usage",
    "classify_error",
]

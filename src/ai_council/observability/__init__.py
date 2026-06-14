"""Observability: per-request tracing and Prometheus metrics."""

from ai_council.observability.metrics import (
    record_error,
    record_llm_call,
    record_request,
    record_stage,
    render,
)
from ai_council.observability.tracing import (
    NoopTracer,
    RecordingTracer,
    Tracer,
    make_tracer,
    span,
    use_tracer,
)

__all__ = [
    "NoopTracer",
    "RecordingTracer",
    "Tracer",
    "make_tracer",
    "record_error",
    "record_llm_call",
    "record_request",
    "record_stage",
    "render",
    "span",
    "use_tracer",
]

"""Optional production tracer adapters.

Each lazily imports its SDK inside __init__, so this module imports fine without
the SDK installed; construction then fails and make_tracer falls back to the
no-op tracer until you install the dependency and enable it in config:

    uv add langfuse            # enables LangfuseTracer  (observability.langfuse: true)
    uv add opentelemetry-sdk   # enables OtelTracer      (observability.otel: true)

Excluded from type-checking because the SDKs are optional, not core deps.
"""

from __future__ import annotations

from typing import Any


class _LangfuseSpan:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def set(self, **attrs: Any) -> None:
        update = getattr(self._obj, "update", None)
        if update is not None:
            update(metadata=attrs)

    def end(self) -> None:
        end = getattr(self._obj, "end", None)
        if end is not None:
            end()


class LangfuseTracer:
    def __init__(self) -> None:
        from langfuse import Langfuse

        self._client = Langfuse()

    def start(self, name: str, parent: Any | None, **attrs: Any) -> Any:
        if parent is None or not hasattr(parent, "_obj"):
            return _LangfuseSpan(self._client.trace(name=name, metadata=attrs))
        return _LangfuseSpan(parent._obj.span(name=name, metadata=attrs))


class _OtelSpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set(self, **attrs: Any) -> None:
        for key, value in attrs.items():
            self._span.set_attribute(key, value)

    def end(self) -> None:
        self._span.end()


class OtelTracer:
    def __init__(self) -> None:
        from opentelemetry import trace

        self._trace = trace
        self._tracer = trace.get_tracer("ai_council")

    def start(self, name: str, parent: Any | None, **attrs: Any) -> Any:
        ctx = None
        if parent is not None and hasattr(parent, "_span"):
            ctx = self._trace.set_span_in_context(parent._span)
        span = self._tracer.start_span(name, context=ctx)
        for key, value in attrs.items():
            span.set_attribute(key, value)
        return _OtelSpan(span)

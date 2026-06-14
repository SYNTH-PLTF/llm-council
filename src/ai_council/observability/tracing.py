"""Tracing abstraction: one trace per request with nested spans.

Spans nest via context vars, so a span opened in a stage automatically becomes
the parent of the gateway spans created by that stage's concurrent LLM calls
(child asyncio tasks copy the context). The default tracer is a no-op;
``RecordingTracer`` is used by tests; ``LangfuseTracer`` / ``OtelTracer`` are
lazy, optional production adapters selected by ``make_tracer`` from config.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_council.telemetry.logging import get_logger

log = get_logger("observability.tracing")


class Span(Protocol):
    def set(self, **attrs: Any) -> None: ...
    def end(self) -> None: ...


class Tracer(Protocol):
    def start(self, name: str, parent: Any | None, **attrs: Any) -> Span: ...


class _NoopSpan:
    def set(self, **attrs: Any) -> None:
        return None

    def end(self) -> None:
        return None


class NoopTracer:
    def start(self, name: str, parent: Any | None, **attrs: Any) -> Span:
        return _NoopSpan()


@dataclass
class RecordingSpan:
    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[RecordingSpan] = field(default_factory=list)

    def set(self, **attrs: Any) -> None:
        self.attrs.update(attrs)

    def end(self) -> None:
        return None


class RecordingTracer:
    """In-memory tracer for tests: builds the span tree so it can be asserted."""

    def __init__(self) -> None:
        self.roots: list[RecordingSpan] = []

    def start(self, name: str, parent: Any | None, **attrs: Any) -> Span:
        node = RecordingSpan(name=name, attrs=dict(attrs))
        if isinstance(parent, RecordingSpan):
            parent.children.append(node)
        else:
            self.roots.append(node)
        return node

    def names(self) -> list[str]:
        out: list[str] = []

        def walk(node: RecordingSpan) -> None:
            out.append(node.name)
            for child in node.children:
                walk(child)

        for root in self.roots:
            walk(root)
        return out


_tracer: ContextVar[Tracer | None] = ContextVar("ai_council_tracer", default=None)
_stack: ContextVar[tuple[Any, ...]] = ContextVar("ai_council_span_stack", default=())


@contextmanager
def use_tracer(tracer: Tracer | None) -> Iterator[None]:
    token = _tracer.set(tracer)
    try:
        yield
    finally:
        _tracer.reset(token)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    tracer = _tracer.get()
    if tracer is None:
        yield _NoopSpan()
        return
    stack = _stack.get()
    parent = stack[-1] if stack else None
    handle = tracer.start(name, parent, **attrs)
    token = _stack.set((*stack, handle))
    started = time.monotonic()
    try:
        yield handle
    finally:
        handle.set(duration_ms=(time.monotonic() - started) * 1000.0)
        _stack.reset(token)
        handle.end()


def make_tracer(langfuse: bool = False, otel: bool = False) -> Tracer:
    if langfuse:
        adapter = _build("ai_council.observability.adapters", "LangfuseTracer")
        if adapter is not None:
            return adapter
    if otel:
        adapter = _build("ai_council.observability.adapters", "OtelTracer")
        if adapter is not None:
            return adapter
    return NoopTracer()


def _build(module: str, attr: str) -> Tracer | None:
    try:
        import importlib

        cls = getattr(importlib.import_module(module), attr)
        return cls()
    except Exception as exc:  # missing optional dep or misconfig -> degrade to noop
        log.warning("observability.tracer_unavailable", adapter=attr, error=str(exc))
        return None

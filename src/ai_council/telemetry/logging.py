"""Structured JSON logging with a per-request correlation id.

``configure_logging`` sets up structlog to emit one JSON object per log line.
``bind_correlation_id`` / ``clear_correlation_id`` attach the current request's
id via context vars, so every log line emitted while handling a request carries
it (``merge_contextvars`` pulls it into the event).
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    level_num = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)


def bind_correlation_id(correlation_id: str) -> None:
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)


def clear_correlation_id() -> None:
    structlog.contextvars.unbind_contextvars("correlation_id")

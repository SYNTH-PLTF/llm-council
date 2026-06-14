"""FastAPI application: lifespan, correlation-id middleware, health + chat.

The lifespan builds one shared Orchestrator (config + gateway) and stores it on
app.state so routes can reach it. Tests can set app.state.orchestrator directly.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_council import __version__
from ai_council.api.routes import router as api_router
from ai_council.cache.redis_cache import RedisCache, make_redis
from ai_council.council.orchestrator import Orchestrator
from ai_council.gateway.client import LLMGateway
from ai_council.observability.metrics import render
from ai_council.observability.tracing import make_tracer
from ai_council.settings import get_config, get_settings
from ai_council.telemetry.logging import (
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    get_logger,
)

log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    redis_url = os.environ.get("REDIS_URL")
    cache = RedisCache(make_redis(redis_url)) if redis_url else None
    gateway = LLMGateway(config, cache=cache)
    tracer = make_tracer(
        langfuse=config.observability.langfuse, otel=config.observability.otel
    )
    app.state.cache = cache
    app.state.orchestrator = Orchestrator(config, gateway, tracer=tracer)
    log.info("app.ready", proposers=len(config.council.proposers), cache=cache is not None)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(title="AI Council", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        request.state.correlation_id = cid
        bind_correlation_id(cid)
        log.info("request.start", method=request.method, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            clear_correlation_id()
        response.headers["X-Correlation-ID"] = cid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz() -> Response:
        try:
            cfg = get_config()
            proposers = len(cfg.council.proposers)
        except Exception as exc:
            log.error("readiness.failed", error=str(exc))
            return JSONResponse(
                {"status": "not_ready", "error": str(exc)}, status_code=503
            )
        return JSONResponse({"status": "ready", "proposers": proposers})

    @app.get("/metrics")
    async def metrics() -> Response:
        payload, content_type = render()
        return Response(content=payload, media_type=content_type)

    app.include_router(api_router)
    return app


app = create_app()

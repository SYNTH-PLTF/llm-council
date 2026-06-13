"""FastAPI application: lifespan, correlation-id middleware, health/readiness.

Business routes (/v1/chat, ...) are added in later phases. Phase 0 ships only
liveness (/healthz) and readiness (/readyz, which proves the config loads).
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from ai_council import __version__
from ai_council.settings import get_config, get_settings
from ai_council.telemetry.logging import (
    bind_correlation_id,
    clear_correlation_id,
    configure_logging,
    get_logger,
)

log = get_logger("api")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    app = FastAPI(title="AI Council", version=__version__)

    @app.middleware("http")
    async def _correlation_id(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        cid = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        bind_correlation_id(cid)
        log.info("request.start", method=request.method, path=request.url.path)
        try:
            response = await call_next(request)
        finally:
            clear_correlation_id()
        response.headers["X-Correlation-ID"] = cid
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

    return app


app = create_app()

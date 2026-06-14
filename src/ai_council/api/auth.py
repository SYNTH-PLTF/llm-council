"""API authentication: bearer token on non-health routes.

When AI_COUNCIL_API_AUTH_TOKEN is unset (local dev) auth is disabled. When set,
every protected route requires `Authorization: Bearer <token>`.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from ai_council.settings import get_settings


async def require_api_key(authorization: str | None = Header(default=None)) -> str:
    expected = get_settings().api_auth_token
    if not expected:
        return "anonymous"  # auth disabled in dev
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="invalid token")
    return token

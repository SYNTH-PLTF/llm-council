"""Phase 12 verification: API auth, security headers, no secret leakage."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ai_council.api.app import create_app
from ai_council.council.orchestrator import RunContext, RunResult
from ai_council.settings import get_settings


class _FakeOrchestrator:
    async def run(self, ctx: RunContext) -> RunResult:
        return RunResult(
            correlation_id=ctx.correlation_id, decision="single_model", final_answer="ok"
        )


@pytest.fixture
def _fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _client() -> TestClient:
    app = create_app()
    app.state.orchestrator = _FakeOrchestrator()
    app.state.cache = None
    return TestClient(app)


def test_unauthenticated_request_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _fresh_settings: None
) -> None:
    monkeypatch.setenv("AI_COUNCIL_API_AUTH_TOKEN", "secret-token")
    get_settings.cache_clear()
    resp = _client().post("/v1/chat", json={"query": "hello there friend"})
    assert resp.status_code == 401


def test_valid_token_is_authorized(
    monkeypatch: pytest.MonkeyPatch, _fresh_settings: None
) -> None:
    monkeypatch.setenv("AI_COUNCIL_API_AUTH_TOKEN", "secret-token")
    get_settings.cache_clear()
    resp = _client().post(
        "/v1/chat",
        json={"query": "hello there friend"},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 200


def test_invalid_token_is_not_echoed(
    monkeypatch: pytest.MonkeyPatch, _fresh_settings: None
) -> None:
    monkeypatch.setenv("AI_COUNCIL_API_AUTH_TOKEN", "secret-token")
    get_settings.cache_clear()
    resp = _client().post(
        "/v1/chat",
        json={"query": "hello there friend"},
        headers={"Authorization": "Bearer wrong-token-9999"},
    )
    assert resp.status_code == 401
    assert "wrong-token-9999" not in resp.text  # the bad token is not reflected back


def test_security_headers_present(_fresh_settings: None) -> None:
    # No auth token configured -> auth disabled, but headers still applied.
    resp = _client().get("/healthz")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"

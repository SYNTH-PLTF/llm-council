"""Phase 0 verification: the app boots and health/readiness work."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai_council.api.app import app


def test_healthz_ok() -> None:
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert resp.headers.get("X-Correlation-ID")


def test_readyz_ready() -> None:
    client = TestClient(app)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_config_loads_pools() -> None:
    from ai_council.settings import get_config

    cfg = get_config("config/default.yaml")
    assert len(cfg.council.proposers) == 3
    assert cfg.council.chairman.model in cfg.models
    for name in cfg.council.proposers:
        assert name in cfg.models

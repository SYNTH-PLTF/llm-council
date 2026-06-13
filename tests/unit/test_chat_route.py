"""API wiring verification: /v1/chat drives the orchestrator and shapes output."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai_council.api.app import create_app
from ai_council.council.orchestrator import RunContext, RunResult


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.last_ctx: RunContext | None = None

    async def run(self, ctx: RunContext) -> RunResult:
        self.last_ctx = ctx
        return RunResult(
            correlation_id=ctx.correlation_id,
            decision="council",
            query_class="high_stakes",
            final_answer="the synthesized answer",
            confidence="high",
            disagreement=0.2,
            cost_usd=0.01,
        )


def _client(orch: _FakeOrchestrator) -> TestClient:
    app = create_app()
    app.state.orchestrator = orch
    return TestClient(app)


def test_chat_returns_orchestrated_result() -> None:
    orch = _FakeOrchestrator()
    client = _client(orch)
    resp = client.post("/v1/chat", json={"query": "what should we do?", "force_council": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["final_answer"] == "the synthesized answer"
    assert body["decision"] == "council"
    assert body["confidence"] == "high"
    assert body["correlation_id"]  # echoed from the middleware
    # the request flags reached the orchestrator
    assert orch.last_ctx is not None
    assert orch.last_ctx.force_council is True


def test_chat_rejects_empty_query() -> None:
    client = _client(_FakeOrchestrator())
    resp = client.post("/v1/chat", json={"query": ""})
    assert resp.status_code == 422  # pydantic min_length


def test_chat_rejects_oversized_query() -> None:
    client = _client(_FakeOrchestrator())
    resp = client.post("/v1/chat", json={"query": "x" * 20_001})
    assert resp.status_code == 422  # pydantic max_length

"""Phase 8 verification: migration applies, repository round-trips, concurrency."""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.pool import StaticPool

from ai_council.council.orchestrator import Orchestrator, RunContext, RunResult, StageTrace
from ai_council.council.proposers import ProposerOutput
from ai_council.gateway.client import LLMGateway
from ai_council.persistence import (
    SqlRepository,
    create_all,
    make_engine,
    make_session_factory,
)
from ai_council.settings import AppConfig

LONG = (
    "please analyze in careful detail the long term economic and social "
    "consequences of this multifaceted policy decision and its many tradeoffs"
)


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> AsyncIterator[SqlRepository]:
    engine = make_engine(
        f"sqlite+aiosqlite:///{tmp_path}/t.db",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    await create_all(engine)
    try:
        yield SqlRepository(make_session_factory(engine))
    finally:
        await engine.dispose()


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("P1", "P2", "P3", "CH"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


def _result(cid: str) -> RunResult:
    return RunResult(
        correlation_id=cid,
        query_class="high_stakes",
        requested_decision="council",
        decision="council",
        final_answer="final",
        confidence="high",
        disagreement=0.3,
        cost_usd=0.05,
        stages=[
            StageTrace(name="triage", detail={"decision": "council"}),
            StageTrace(name="chairman", detail={"confidence": "high"}),
        ],
    )


def _proposers() -> list[ProposerOutput]:
    return [
        ProposerOutput(model="P1", ok=True, text="a1", cost_usd=0.01),
        ProposerOutput(model="P2", ok=False, error="boom"),
    ]


async def test_repository_round_trip(repo: SqlRepository) -> None:
    run_id = await repo.save_run(result=_result("r1"), query="q", proposer_outputs=_proposers())
    assert run_id == "r1"
    loaded = await repo.get_run("r1")
    assert loaded is not None
    assert loaded.decision == "council"
    assert loaded.final_answer == "final"
    assert [s.name for s in loaded.stages] == ["triage", "chairman"]  # ordered by seq
    assert {p.model for p in loaded.proposers} == {"P1", "P2"}
    assert any(p.error == "boom" for p in loaded.proposers)


async def test_concurrent_writes_safe(repo: SqlRepository) -> None:
    ids = [f"run-{i}" for i in range(10)]
    await asyncio.gather(*(repo.save_run(result=_result(i), query="q") for i in ids))
    loaded = await asyncio.gather(*(repo.get_run(i) for i in ids))
    assert {r.id for r in loaded if r is not None} == set(ids)


async def test_orchestrator_persists_council_run(repo: SqlRepository) -> None:
    cfg = _cfg()
    gw = LLMGateway(cfg, completion_fn=_council_completion(), sleep_fn=_no_sleep)
    orch = Orchestrator(cfg, gw, store=repo, rng=random.Random(0))
    result = await orch.run(RunContext(correlation_id="run-x", query=LONG, force_council=True))
    assert result.decision == "council"
    loaded = await repo.get_run("run-x")
    assert loaded is not None
    assert any(s.name == "chairman" for s in loaded.stages)
    assert len(loaded.proposers) == 3


def test_migration_applies(tmp_path: Path) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    db = tmp_path / "m.db"
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db}")
    command.upgrade(cfg, "head")
    engine = create_engine(f"sqlite:///{db}")
    tables = set(inspect(engine).get_table_names())
    engine.dispose()
    assert {"conversations", "messages", "runs", "run_stages", "run_proposers"} <= tables


# --- helpers for the orchestrator-persists test -----------------------------


def _cfg() -> AppConfig:
    models = ["P1", "P2", "P3", "CH"]
    return AppConfig.model_validate(
        {
            "models": {m: {"env_prefix": m} for m in models},
            "router": {"router_model": "P1", "budgets": {"latency_budget_s": 30.0}},
            "council": {
                "proposers": ["P1", "P2", "P3"],
                "quorum": 2,
                "chairman": {"model": "CH"},
                "ranking": {"orderings_per_judge": 2},
                "debate": {"enabled": False},
            },
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
        }
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )


def _council_completion() -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        short = str(kwargs["model"]).split("/")[-1]
        if "Rank ALL candidates" in content:
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}))
        if "final_answer" in content:
            return _resp(json.dumps({"final_answer": "SYNTH", "confidence": "high"}))
        return _resp(f"answer-{short}")

    return fn

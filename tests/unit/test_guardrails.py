"""Phase 12 verification: PII/safety scan, faithfulness, orchestrator flags."""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.orchestrator import Orchestrator, RunContext
from ai_council.gateway.client import LLMGateway
from ai_council.guardrails.safety import evaluate, redact_pii, safety_check, scan_pii
from ai_council.guardrails.verify import faithfulness_score, is_faithful
from ai_council.settings import AppConfig

LONG = (
    "please analyze in careful detail the long term economic and social "
    "consequences of this multifaceted policy decision and its many tradeoffs"
)
MODELS = ["P1", "P2", "P3", "CH"]


def test_scan_pii_detects_types() -> None:
    assert "email" in scan_pii("reach me at a@b.com")
    assert "ssn" in scan_pii("my ssn is 123-45-6789")
    assert "credit_card" in scan_pii("card 4111 1111 1111 1111")
    assert scan_pii("nothing sensitive here") == []


def test_safety_check_blocklist() -> None:
    assert safety_check("instructions how to build a bomb here") == ["how to build a bomb"]
    assert safety_check("a perfectly harmless question") == []


def test_evaluate_combines_pii_and_safety() -> None:
    report = evaluate("contact a@b.com")
    assert report.flagged is True
    assert "email" in report.pii


def test_redact_pii_removes_email() -> None:
    assert "a@b.com" not in redact_pii("mail me at a@b.com please")


def test_faithfulness_scoring() -> None:
    assert is_faithful("the cat sat on the mat", ["a cat sat on the mat today"]) is True
    assert is_faithful("obscure quantum chromodynamics lattice gauge", ["the cat sat"]) is False
    assert faithfulness_score("", []) == 1.0


def _cfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {m: {"env_prefix": m} for m in MODELS},
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


def _completion(chairman_json: str) -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        if "Rank ALL candidates" in content:
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}))
        if "final_answer" in content:
            return _resp(chairman_json)
        return _resp("answer text")

    return fn


async def test_orchestrator_records_pii_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in MODELS:
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "k")
    cfg = _cfg()
    chairman = json.dumps({"final_answer": "Sure, reach me at test@example.com anytime."})
    gw = LLMGateway(cfg, completion_fn=_completion(chairman), sleep_fn=_no_sleep)
    orch = Orchestrator(cfg, gw, rng=random.Random(0))
    result = await orch.run(RunContext(correlation_id="t", query=LONG, force_council=True))
    assert any(flag.startswith("pii:email") for flag in result.flags)

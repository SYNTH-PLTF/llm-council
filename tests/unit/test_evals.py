"""Phase 11 verification: council uplift, the decision rule, the CI gate."""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.orchestrator import Orchestrator
from ai_council.evals.harness import (
    EvalItem,
    assert_no_regression,
    decision_overrides,
    run_eval,
    shadow_single,
)
from ai_council.evals.run import load_dataset
from ai_council.gateway.client import LLMGateway
from ai_council.settings import AppConfig

MODELS = ["P1", "P2", "P3", "CH"]
DATASET = [
    EvalItem(
        query="should we restructure the org in detail please consider many angles carefully",
        query_class="high_stakes",
        reference="it depends on context",
    ),
    EvalItem(
        query="another consequential strategic question to weigh carefully across many factors",
        query_class="high_stakes",
        reference="it depends on context",
    ),
]


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


def _completion() -> Any:
    async def fn(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][-1]["content"])
        if "Rank ALL candidates" in content:
            letters = re.findall(r"(?m)^([A-Z]):", content)
            return _resp(json.dumps({"ranking": letters or ["A"]}))
        if "final_answer" in content:  # chairman = the council answer
            return _resp(json.dumps({"final_answer": "SYNTH-ANSWER", "confidence": "high"}))
        return _resp("weak-answer")  # proposer / single-model answer

    return fn


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in MODELS:
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "k")


def _orch() -> Orchestrator:
    cfg = _cfg()
    gw = LLMGateway(cfg, completion_fn=_completion(), sleep_fn=_no_sleep)
    return Orchestrator(cfg, gw, rng=random.Random(0))


def _synth_scorer(answer: str, _item: EvalItem) -> float:
    return 1.0 if "SYNTH" in answer else 0.2


async def test_run_eval_computes_positive_uplift() -> None:
    report = await run_eval(_orch(), DATASET, scorer=_synth_scorer)
    high = report.classes["high_stakes"]
    assert high.n == 2
    assert high.council_avg == pytest.approx(1.0)
    assert high.single_avg == pytest.approx(0.2)
    assert high.uplift == pytest.approx(0.8)


async def test_decision_rule_routes_flat_class_to_single() -> None:
    report = await run_eval(_orch(), DATASET, scorer=lambda _a, _i: 0.5)  # no uplift
    assert decision_overrides(report) == {"high_stakes": "single_model"}


async def test_ci_gate_passes_then_fails_on_regression() -> None:
    good = await run_eval(_orch(), DATASET, scorer=_synth_scorer)
    assert_no_regression(good, classes=["high_stakes"], min_uplift=0.5)  # no raise

    regressed = await run_eval(_orch(), DATASET, scorer=lambda a, _i: 0.2 if "SYNTH" in a else 1.0)
    with pytest.raises(AssertionError):
        assert_no_regression(regressed, classes=["high_stakes"], min_uplift=0.5)


async def test_shadow_single_uses_single_model() -> None:
    result = await shadow_single(_orch(), "a consequential query to weigh carefully here today")
    assert result.decision == "single_model"
    assert result.final_answer == "weak-answer"


def test_load_dataset_reads_json_files() -> None:
    items = load_dataset("evals/datasets")
    assert len(items) >= 4
    assert any(item.query_class == "verifiable_reasoning" for item in items)

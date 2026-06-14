"""Offline eval harness: prove the council beats single-best per query class.

Each labeled item is run twice through the orchestrator (force_council vs
force_single); both answers are scored and council uplift = council - single is
computed per class. The decision rule routes a class to single_model when its
uplift is not positive, and the CI gate fails when a high-value class regresses
below a threshold.

Scoring is deterministic by default (exact-match for verifiable answers, token
overlap otherwise) so the harness runs without an LLM judge; a real LLM-judge
scorer can be supplied for production eval runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from ai_council.council.orchestrator import Orchestrator, RunContext, RunResult
from ai_council.council.voting import extract_answer


class EvalItem(BaseModel):
    query: str
    query_class: str = "standard"
    reference: str | None = None


Scorer = Callable[[str, "EvalItem"], float]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def overlap_score(answer: str, item: EvalItem) -> float:
    if not item.reference:
        return 0.0
    answer_tokens, ref_tokens = _tokens(answer), _tokens(item.reference)
    if not ref_tokens:
        return 0.0
    return len(answer_tokens & ref_tokens) / len(ref_tokens)


def default_scorer(answer: str, item: EvalItem) -> float:
    if item.query_class == "verifiable_reasoning" and item.reference is not None:
        got = extract_answer(answer)
        want = extract_answer(item.reference)
        return 1.0 if got is not None and got == want else 0.0
    return overlap_score(answer, item)


@dataclass
class ClassReport:
    query_class: str
    n: int = 0
    council_total: float = 0.0
    single_total: float = 0.0
    council_cost: float = 0.0
    single_cost: float = 0.0

    @property
    def council_avg(self) -> float:
        return self.council_total / self.n if self.n else 0.0

    @property
    def single_avg(self) -> float:
        return self.single_total / self.n if self.n else 0.0

    @property
    def uplift(self) -> float:
        return self.council_avg - self.single_avg


@dataclass
class EvalReport:
    classes: dict[str, ClassReport] = field(default_factory=dict)

    def bucket(self, name: str) -> ClassReport:
        if name not in self.classes:
            self.classes[name] = ClassReport(query_class=name)
        return self.classes[name]

    def total_cost(self) -> float:
        return sum(c.council_cost + c.single_cost for c in self.classes.values())


async def run_eval(
    orchestrator: Orchestrator,
    dataset: list[EvalItem],
    *,
    scorer: Scorer = default_scorer,
) -> EvalReport:
    report = EvalReport()
    for item in dataset:
        council = await orchestrator.run(
            RunContext(correlation_id="eval-council", query=item.query, force_council=True)
        )
        single = await orchestrator.run(
            RunContext(correlation_id="eval-single", query=item.query, force_single=True)
        )
        bucket = report.bucket(item.query_class)
        bucket.n += 1
        bucket.council_total += scorer(council.final_answer, item)
        bucket.single_total += scorer(single.final_answer, item)
        bucket.council_cost += council.cost_usd
        bucket.single_cost += single.cost_usd
    return report


def decision_overrides(report: EvalReport) -> dict[str, str]:
    """Classes with no council uplift get routed to single_model (the Phase 11
    decision rule). Apply these to config.class_routing."""
    return {
        name: "single_model" for name, cls in report.classes.items() if cls.uplift <= 0.0
    }


def assert_no_regression(report: EvalReport, *, classes: list[str], min_uplift: float) -> None:
    """CI gate: raise if any listed high-value class regresses below min_uplift."""
    for name in classes:
        cls = report.classes.get(name)
        if cls is not None and cls.uplift < min_uplift:
            raise AssertionError(
                f"council uplift regression: {name} uplift={cls.uplift:.3f} < {min_uplift}"
            )


async def shadow_single(orchestrator: Orchestrator, query: str) -> RunResult:
    """Online shadow-mode hook: run single-best for a council query so a judge
    can compare later. Call on a sampled fraction; never block the user response."""
    return await orchestrator.run(
        RunContext(correlation_id="shadow", query=query, force_single=True)
    )

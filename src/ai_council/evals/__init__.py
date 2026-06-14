"""Offline eval harness: prove the council beats single-best per query class."""

from ai_council.evals.harness import (
    ClassReport,
    EvalItem,
    EvalReport,
    assert_no_regression,
    decision_overrides,
    default_scorer,
    overlap_score,
    run_eval,
    shadow_single,
)

__all__ = [
    "ClassReport",
    "EvalItem",
    "EvalReport",
    "assert_no_regression",
    "decision_overrides",
    "default_scorer",
    "overlap_score",
    "run_eval",
    "shadow_single",
]

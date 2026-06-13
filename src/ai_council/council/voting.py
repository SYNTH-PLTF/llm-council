"""Stage 2b (verifiable tasks): self-consistency majority vote.

For math/code/logic with a checkable answer, an LLM judge is unreliable at
deciding correctness, so we skip it entirely: extract each proposer's final
answer and take the majority vote, breaking ties by configured proposer
quality order. No gateway calls happen here; the decision is pure tallying.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

_PATTERNS = [
    re.compile(r"\\boxed\{([^}]*)\}"),
    re.compile(r"final answer\s*(?:is|:|=)?\s*(.+)", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|:|=)\s*(.+)", re.IGNORECASE),
]


class VoteResult(BaseModel):
    winner: str | None
    counts: dict[str, int]
    support: list[str]
    extracted: dict[str, str | None]
    tie_break_used: bool = False


def _normalize(value: str) -> str:
    return value.strip().rstrip(".").strip().lower()


def extract_answer(text: str) -> str | None:
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize(match.group(1))
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return _normalize(lines[-1]) if lines else None


def majority_vote(
    proposer_answers: list[tuple[str, str]], quality_order: list[str]
) -> VoteResult:
    extracted: dict[str, str | None] = {
        pid: extract_answer(text) for pid, text in proposer_answers
    }
    counts: dict[str, int] = {}
    supporters: dict[str, list[str]] = {}
    for pid, _text in proposer_answers:
        answer = extracted[pid]
        if answer is None:
            continue
        counts[answer] = counts.get(answer, 0) + 1
        supporters.setdefault(answer, []).append(pid)
    if not counts:
        return VoteResult(winner=None, counts={}, support=[], extracted=extracted)

    top = max(counts.values())
    tied = [a for a, c in counts.items() if c == top]
    tie_break = len(tied) > 1

    def best_rank(answer: str) -> int:
        ranks = [quality_order.index(p) for p in supporters[answer] if p in quality_order]
        return min(ranks) if ranks else len(quality_order)

    winner = tied[0] if not tie_break else min(tied, key=lambda a: (best_rank(a), a))
    return VoteResult(
        winner=winner,
        counts=counts,
        support=supporters[winner],
        extracted=extracted,
        tie_break_used=tie_break,
    )

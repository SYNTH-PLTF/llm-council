"""Lightweight faithfulness check: does the synthesis stay grounded in the
top-ranked proposer answers? A very low overlap suggests the Chairman
introduced unsupported claims. This flags, it does not rewrite."""

from __future__ import annotations

import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def faithfulness_score(answer: str, sources: list[str]) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 1.0
    source_tokens: set[str] = set()
    for source in sources:
        source_tokens |= _tokens(source)
    return len(answer_tokens & source_tokens) / len(answer_tokens)


def is_faithful(answer: str, sources: list[str], *, threshold: float = 0.3) -> bool:
    return faithfulness_score(answer, sources) >= threshold

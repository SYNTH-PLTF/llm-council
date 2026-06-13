"""Stage 2b (council path): one bounded debate round when judges disagree.

Runs only for the 'council' decision (verifiable tasks vote instead), only when
disagreement exceeds the configured threshold, and only up to max_debate_rounds
(default 1) with an early stop when answers stabilize. It can never loop
unbounded: the round count is a hard for-range cap.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_council.council.ranking import Candidate, RankingResult, label_candidates
from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("council.debate")

RerankFn = Callable[[list[Candidate]], Awaitable[RankingResult]]

_DEFAULT_DEBATE_PROMPT = (
    "Below are candidate answers to a question, anonymized as letters, plus the "
    "current best. Produce your single best REVISED answer to the question. Keep "
    "yours if it is already strongest; improve it if another is better. Do not "
    "mention other models.\n\n"
    'Question:\n"""{{query}}"""\n\n'
    "Candidate answers:\n{{labeled_candidates}}\n\n"
    "Current best: {{top_letter}}\n\nYour revised answer:"
)


@dataclass
class DebateResult:
    candidates: list[Candidate]
    ranking: RankingResult
    rounds: int
    converged: bool


def should_debate(config: AppConfig, decision: str, disagreement: float) -> bool:
    debate = config.council.debate
    return bool(debate.enabled and decision == "council" and disagreement > debate.threshold)


def _norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def converged(old: list[Candidate], new: list[Candidate]) -> bool:
    old_text = {c.id: _norm(c.text) for c in old}
    return all(_norm(c.text) == old_text.get(c.id) for c in new)


async def run_debate_round(
    gateway: LLMGateway,
    config: AppConfig,
    query: str,
    candidates: list[Candidate],
    ranking: RankingResult,
    rng: random.Random,
    *,
    prompt: str | None = None,
) -> list[Candidate]:
    template = prompt if prompt is not None else _DEFAULT_DEBATE_PROMPT
    labeled, mapping = label_candidates(candidates, rng)
    top_id = ranking.ordering[0] if ranking.ordering else ""
    top_letter = next((ltr for ltr, cid in mapping.items() if cid == top_id), "?")
    body = (
        template.replace("{{query}}", query)
        .replace("{{labeled_candidates}}", labeled)
        .replace("{{top_letter}}", top_letter)
    )
    tasks = [
        gateway.complete(
            c.id, [ChatMessage(role="user", content=body)], max_tokens=1024, temperature=0.5
        )
        for c in candidates
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    revised: list[Candidate] = []
    for cand, res in zip(candidates, results, strict=True):
        if isinstance(res, BaseException):
            log.warning("council.debate_failed", model=cand.id, error=str(res))
            revised.append(cand)
        else:
            revised.append(Candidate(id=cand.id, text=res.text))
    return revised


async def run_debate(
    gateway: LLMGateway,
    config: AppConfig,
    query: str,
    candidates: list[Candidate],
    ranking: RankingResult,
    *,
    rerank: RerankFn,
    rng: random.Random,
    prompt: str | None = None,
) -> DebateResult:
    current = candidates
    current_ranking = ranking
    max_rounds = max(1, config.council.debate.max_rounds)
    conv = False
    rounds = 0
    for _ in range(max_rounds):
        new = await run_debate_round(
            gateway, config, query, current, current_ranking, rng, prompt=prompt
        )
        rounds += 1
        conv = converged(current, new)
        current = new
        current_ranking = await rerank(current)
        if conv:
            break
    return DebateResult(
        candidates=current, ranking=current_ranking, rounds=rounds, converged=conv
    )

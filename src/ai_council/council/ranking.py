"""Stage 2: debiased peer ranking.

Position bias can swing an LLM judge's pairwise verdict by >10%, so ranking is
debiased two ways:

  * Anonymization - each judge sees candidates as "A/B/C" under a per-judge,
    per-ordering RANDOM letter mapping, so no model can recognize or favor its
    own answer and identity cannot leak across judges.
  * Order-swap averaging - every judge ranks the candidates >=2 times under
    different presentation orders; we average each candidate's rank position
    across those orderings before consensus.

Consensus uses Borda count across judges; disagreement is reported as a
normalized mean pairwise Kendall tau (0 = unanimous, 1 = fully reversed).

The math lives in pure module-level functions (unit-tested directly); ``Ranker``
only orchestrates the gateway calls around them.
"""

from __future__ import annotations

import asyncio
import json
import random
import string
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("council.ranking")

_LETTERS = string.ascii_uppercase


class Candidate(BaseModel):
    id: str
    text: str


class RankingResult(BaseModel):
    ordering: list[str]
    borda_scores: dict[str, float] = Field(default_factory=dict)
    mean_ranks: dict[str, float] = Field(default_factory=dict)
    disagreement: float = 0.0
    judge_count: int = 0

    def top_k(self, k: int) -> list[str]:
        return self.ordering[:k]


def label_candidates(
    candidates: list[Candidate], rng: random.Random
) -> tuple[str, dict[str, str]]:
    """Shuffle candidates and label them A/B/C...; return text + letter->id map."""
    order = list(candidates)
    rng.shuffle(order)
    letter_to_id: dict[str, str] = {}
    blocks: list[str] = []
    for i, cand in enumerate(order):
        letter = _LETTERS[i]
        letter_to_id[letter] = cand.id
        blocks.append(f"{letter}: {cand.text}")
    return "\n\n".join(blocks), letter_to_id


def letters_to_ids(
    letters: list[str], letter_to_id: dict[str, str], all_ids: list[str]
) -> list[str]:
    """Map a judge's letter ranking back to ids; drop invalid, append dropped."""
    seen: set[str] = set()
    ordered: list[str] = []
    for letter in letters:
        cid = letter_to_id.get(str(letter).strip().upper()[:1])
        if cid is not None and cid not in seen:
            ordered.append(cid)
            seen.add(cid)
    for cid in all_ids:
        if cid not in seen:
            ordered.append(cid)
            seen.add(cid)
    return ordered


def positions(ordering: list[str], all_ids: list[str]) -> dict[str, int]:
    """0-indexed rank position per id (0 = best)."""
    pos = {cid: idx for idx, cid in enumerate(ordering)}
    return {cid: pos.get(cid, len(all_ids) - 1) for cid in all_ids}


def average_positions(per_ordering: list[dict[str, int]]) -> dict[str, float]:
    """Order-swap averaging: mean rank position per id across orderings."""
    if not per_ordering:
        return {}
    ids = list(per_ordering[0].keys())
    return {cid: sum(o[cid] for o in per_ordering) / len(per_ordering) for cid in ids}


def borda_consensus(
    per_judge: list[dict[str, float]], all_ids: list[str]
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    """Aggregate per-judge mean ranks into a Borda ordering + scores + mean."""
    n = len(all_ids)
    borda: dict[str, float] = dict.fromkeys(all_ids, 0.0)
    mean: dict[str, float] = dict.fromkeys(all_ids, 0.0)
    for judge in per_judge:
        for cid in all_ids:
            rank = judge.get(cid, n - 1)
            borda[cid] += (n - 1) - rank
            mean[cid] += rank
    j = max(1, len(per_judge))
    mean = {cid: mean[cid] / j for cid in all_ids}
    ordering = sorted(all_ids, key=lambda cid: (-borda[cid], mean[cid], cid))
    return ordering, borda, mean


def kendall_tau(p1: dict[str, float], p2: dict[str, float], ids: list[str]) -> float:
    """Kendall tau between two rank dicts; ties are skipped. 1.0 when no info."""
    conc = disc = 0
    for i in range(len(ids)):
        for k in range(i + 1, len(ids)):
            a, b = ids[i], ids[k]
            s1 = p1[a] - p1[b]
            s2 = p2[a] - p2[b]
            if s1 == 0 or s2 == 0:
                continue
            if (s1 > 0) == (s2 > 0):
                conc += 1
            else:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom else 1.0


def disagreement(per_judge: list[dict[str, float]], ids: list[str]) -> float:
    """Normalized mean pairwise Kendall tau: 0 = unanimous, 1 = fully reversed."""
    if len(per_judge) < 2 or len(ids) < 2:
        return 0.0
    taus: list[float] = []
    for i in range(len(per_judge)):
        for k in range(i + 1, len(per_judge)):
            taus.append(kendall_tau(per_judge[i], per_judge[k], ids))
    mean_tau = sum(taus) / len(taus)
    return max(0.0, min(1.0, (1.0 - mean_tau) / 2.0))


def load_ranker_prompt(config_dir: str = "config/prompts") -> str:
    return Path(config_dir, "ranker.md").read_text(encoding="utf-8")


class Ranker:
    def __init__(
        self,
        config: AppConfig,
        gateway: LLMGateway,
        *,
        prompt: str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._prompt = prompt if prompt is not None else load_ranker_prompt()
        self._rng = rng if rng is not None else random.Random()

    def _judges(self) -> list[str]:
        council = self._config.council
        if council.judges == "separate_pool" and council.judge_pool:
            return list(council.judge_pool)
        return list(council.proposers)

    def _trivial(self, ids: list[str]) -> RankingResult:
        return RankingResult(
            ordering=ids,
            borda_scores=dict.fromkeys(ids, 0.0),
            mean_ranks=dict.fromkeys(ids, 0.0),
            disagreement=0.0,
            judge_count=0,
        )

    async def rank(self, query: str, candidates: list[Candidate]) -> RankingResult:
        ids = [c.id for c in candidates]
        if len(candidates) <= 1:
            return self._trivial(ids)

        judges = self._judges()
        orderings_n = max(2, self._config.council.ranking.orderings_per_judge)
        tasks = []
        meta: list[tuple[str, dict[str, str]]] = []
        for judge in judges:
            for _ in range(orderings_n):
                labeled, mapping = label_candidates(candidates, self._rng)
                tasks.append(self._ask_judge(judge, query, labeled))
                meta.append((judge, mapping))
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        per_judge_orderings: dict[str, list[dict[str, int]]] = defaultdict(list)
        for (judge, mapping), res in zip(meta, raw, strict=True):
            if isinstance(res, BaseException):
                log.warning("council.judge_failed", judge=judge, error=str(res))
                continue
            ordered_ids = letters_to_ids(res, mapping, ids)
            per_judge_orderings[judge].append(positions(ordered_ids, ids))

        per_judge_mean = [
            average_positions(o) for o in per_judge_orderings.values() if o
        ]
        if not per_judge_mean:
            log.warning("council.ranking_no_signal")
            return self._trivial(ids)

        ordering, borda, mean = borda_consensus(per_judge_mean, ids)
        return RankingResult(
            ordering=ordering,
            borda_scores=borda,
            mean_ranks=mean,
            disagreement=disagreement(per_judge_mean, ids),
            judge_count=len(per_judge_mean),
        )

    async def _ask_judge(self, judge: str, query: str, labeled: str) -> list[str]:
        prompt = self._prompt.replace("{{labeled_candidates}}", labeled).replace(
            "{{query}}", query
        )
        result = await self._gateway.complete(
            judge,
            [ChatMessage(role="user", content=prompt)],
            max_tokens=300,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(_first_json_object(result.text))
        ranking = data.get("ranking")
        if not isinstance(ranking, list):
            raise ValueError("ranking is not a list")
        return [str(x) for x in ranking]


def _first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no json object in judge output")
    return text[start : end + 1]

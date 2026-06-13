"""The council orchestrator: an explicit state machine over the stages.

    triage -> (single_model | council)
    council -> proposers -> rank -> (debate | vote)? -> chairman -> result

A single immutable ``RunContext`` (correlation id, query, conversation history,
budgets) is threaded through every stage so history is never lost. The whole
run is bounded by a wall-clock budget: if it is exceeded mid-run we return the
best result captured so far with ``timeout_partial=True`` rather than hanging.
Total cost across all stages is captured via the gateway's ``cost_capture``.
Persistence (Phase 8) and guardrails (Phase 12) attach as optional hooks.
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai_council.council.chairman import Chairman, ChairmanVerdict
from ai_council.council.debate import run_debate, should_debate
from ai_council.council.proposers import ProposerOutput, run_stage1
from ai_council.council.ranking import Candidate, Ranker, RankingResult
from ai_council.council.voting import VoteResult, majority_vote
from ai_council.gateway.client import LLMGateway, cost_capture
from ai_council.gateway.models import ChatMessage, GatewayError
from ai_council.router.triage import LLMRouter, Router, RouteRequest, TriageResult
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("council.orchestrator")

Guardrail = Callable[[str], "str | Awaitable[str]"]

_TOP_K = 3
_SINGLE_MAX_TOKENS = 1024


@dataclass(frozen=True)
class RunContext:
    correlation_id: str
    query: str
    history: tuple[ChatMessage, ...] = ()
    request_budget_usd: float | None = None
    user_daily_remaining_usd: float | None = None
    force_council: bool = False
    force_single: bool = False


class StageTrace(BaseModel):
    name: str
    detail: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    correlation_id: str = ""
    query_class: str = "standard"
    requested_decision: str = "single_model"
    decision: str = "single_model"
    final_answer: str = ""
    confidence: str = "medium"
    dissent_notes: str = ""
    contributing_sources: list[str] = Field(default_factory=list)
    disagreement: float = 0.0
    degraded: bool = False
    timeout_partial: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    proposer_models: list[str] = Field(default_factory=list)
    stages: list[StageTrace] = Field(default_factory=list)


@dataclass
class _Ledger:
    stages: list[StageTrace] = field(default_factory=list)

    def add(self, name: str, detail: dict[str, Any]) -> None:
        self.stages.append(StageTrace(name=name, detail=detail))


@dataclass
class _Progress:
    query_class: str = "standard"
    requested_decision: str = "single_model"
    decision: str = "single_model"
    candidates: list[Candidate] = field(default_factory=list)
    ranking: RankingResult | None = None
    verdict: ChairmanVerdict | None = None
    proposer_models: list[str] = field(default_factory=list)
    disagreement: float = 0.0
    degraded: bool = False


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        gateway: LLMGateway,
        *,
        router: Router | None = None,
        ranker: Ranker | None = None,
        chairman: Chairman | None = None,
        guardrail: Guardrail | None = None,
        proposer_prompt: str | None = None,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._rng = rng if rng is not None else random.Random()
        self._router = router if router is not None else LLMRouter(config, gateway)
        self._ranker = ranker if ranker is not None else Ranker(config, gateway, rng=self._rng)
        self._chairman = chairman if chairman is not None else Chairman(config, gateway)
        self._guardrail = guardrail
        self._proposer_prompt = proposer_prompt
        self._clock = clock

    async def run(self, ctx: RunContext) -> RunResult:
        ledger = _Ledger()
        progress = _Progress()
        started = self._clock()
        budget_s = self._config.router.budgets.latency_budget_s
        with cost_capture() as costs:
            try:
                if budget_s and budget_s > 0:
                    result = await asyncio.wait_for(
                        self._pipeline(ctx, ledger, progress), timeout=budget_s
                    )
                else:
                    result = await self._pipeline(ctx, ledger, progress)
            except TimeoutError:
                log.warning("orchestrator.timeout_partial", correlation_id=ctx.correlation_id)
                result = self._partial(progress)
                ledger.add("timeout", {"partial": True})
            if self._guardrail is not None:
                result.final_answer = await _apply_guardrail(self._guardrail, result.final_answer)
                ledger.add("guardrail", {"applied": True})
        result.correlation_id = ctx.correlation_id
        result.cost_usd = costs.total
        result.latency_ms = (self._clock() - started) * 1000.0
        result.stages = ledger.stages
        return result

    async def _pipeline(self, ctx: RunContext, ledger: _Ledger, progress: _Progress) -> RunResult:
        triage = await self._router.route(
            RouteRequest(
                query=ctx.query,
                force_council=ctx.force_council,
                force_single=ctx.force_single,
                budget_remaining_usd=ctx.request_budget_usd,
                user_daily_remaining_usd=ctx.user_daily_remaining_usd,
            )
        )
        ledger.add(
            "triage",
            {
                "query_class": triage.query_class,
                "decision": triage.decision,
                "source": triage.source,
                "reason": triage.reason,
            },
        )
        progress.query_class = triage.query_class
        progress.requested_decision = triage.decision
        progress.decision = triage.decision
        if triage.decision == "single_model":
            return await self._run_single(ctx, triage, ledger, progress)
        return await self._run_council(ctx, triage, ledger, progress)

    async def _run_single(
        self, ctx: RunContext, triage: TriageResult, ledger: _Ledger, progress: _Progress
    ) -> RunResult:
        model = self._best_model()
        progress.proposer_models = [model]
        final = ""
        try:
            res = await self._gateway.complete(
                model, self._messages(ctx), max_tokens=_SINGLE_MAX_TOKENS, temperature=0.2
            )
            final = res.text
        except GatewayError as exc:
            ledger.add("single_error", {"error": str(exc)})
        ledger.add("single_model", {"model": model})
        progress.verdict = ChairmanVerdict(final_answer=final)
        return RunResult(
            query_class=triage.query_class,
            requested_decision=triage.decision,
            decision="single_model",
            final_answer=final,
            proposer_models=[model],
        )

    async def _run_council(
        self, ctx: RunContext, triage: TriageResult, ledger: _Ledger, progress: _Progress
    ) -> RunResult:
        stage1 = await run_stage1(self._gateway, self._config, self._messages(ctx))
        progress.degraded = stage1.degraded
        progress.proposer_models = [o.model for o in stage1.round.outputs]
        ledger.add(
            "proposers",
            {
                "successes": len(stage1.round.successes),
                "quorum": stage1.round.quorum,
                "degraded": stage1.degraded,
            },
        )
        if stage1.degraded:
            single = stage1.single
            final = single.text if single is not None and single.ok else ""
            progress.decision = "single_model"
            progress.verdict = ChairmanVerdict(final_answer=final)
            return RunResult(
                query_class=triage.query_class,
                requested_decision=triage.decision,
                decision="single_model",
                final_answer=final,
                degraded=True,
                proposer_models=progress.proposer_models,
            )

        successes = stage1.round.successes
        candidates = [Candidate(id=o.model, text=o.text) for o in successes]
        progress.candidates = candidates

        if triage.decision == "council_with_voting":
            vote = majority_vote(
                [(o.model, o.text) for o in successes], list(self._config.council.proposers)
            )
            ledger.add(
                "voting",
                {"winner": vote.winner, "counts": vote.counts, "tie_break": vote.tie_break_used},
            )
            return RunResult(
                query_class=triage.query_class,
                requested_decision=triage.decision,
                decision="council_with_voting",
                final_answer=self._vote_answer(vote, successes),
                contributing_sources=vote.support,
                proposer_models=progress.proposer_models,
            )

        ranking = await self._ranker.rank(ctx.query, candidates)
        progress.ranking = ranking
        progress.disagreement = ranking.disagreement
        ledger.add("ranking", {"ordering": ranking.ordering, "disagreement": ranking.disagreement})

        if should_debate(self._config, triage.decision, ranking.disagreement):

            async def _rerank(cands: list[Candidate]) -> RankingResult:
                return await self._ranker.rank(ctx.query, cands)

            debate = await run_debate(
                self._gateway,
                self._config,
                ctx.query,
                candidates,
                ranking,
                rerank=_rerank,
                rng=self._rng,
            )
            candidates = debate.candidates
            ranking = debate.ranking
            progress.candidates = candidates
            progress.ranking = ranking
            progress.disagreement = ranking.disagreement
            ledger.add("debate", {"rounds": debate.rounds, "converged": debate.converged})

        top = self._top_candidates(candidates, ranking)
        verdict = await self._chairman.synthesize(ctx.query, top)
        progress.verdict = verdict
        ledger.add("chairman", {"confidence": verdict.confidence})
        return RunResult(
            query_class=triage.query_class,
            requested_decision=triage.decision,
            decision="council",
            final_answer=verdict.final_answer,
            confidence=verdict.confidence,
            dissent_notes=verdict.dissent_notes,
            contributing_sources=verdict.contributing_sources,
            disagreement=ranking.disagreement,
            proposer_models=progress.proposer_models,
        )

    def _messages(self, ctx: RunContext) -> list[ChatMessage]:
        if self._proposer_prompt is not None:
            body = self._proposer_prompt.replace("{{conversation_context}}", "").replace(
                "{{query}}", ctx.query
            )
            return [*ctx.history, ChatMessage(role="user", content=body)]
        return [*ctx.history, ChatMessage(role="user", content=ctx.query)]

    def _best_model(self) -> str:
        proposers = self._config.council.proposers
        return proposers[0] if proposers else self._config.router.router_model

    def _top_candidates(
        self, candidates: list[Candidate], ranking: RankingResult
    ) -> list[Candidate]:
        by_id = {c.id: c for c in candidates}
        k = min(_TOP_K, len(candidates))
        top = [by_id[cid] for cid in ranking.top_k(k) if cid in by_id]
        return top or candidates[:k]

    def _vote_answer(self, vote: VoteResult, successes: list[ProposerOutput]) -> str:
        if vote.winner is None:
            return successes[0].text if successes else ""
        support = set(vote.support)
        for name in self._config.council.proposers:
            if name in support:
                return next((o.text for o in successes if o.model == name), "")
        return successes[0].text if successes else ""

    def _partial(self, progress: _Progress) -> RunResult:
        if progress.verdict is not None:
            final = progress.verdict.final_answer
        elif progress.ranking is not None and progress.candidates:
            top_id = progress.ranking.ordering[0] if progress.ranking.ordering else ""
            final = next(
                (c.text for c in progress.candidates if c.id == top_id),
                progress.candidates[0].text,
            )
        elif progress.candidates:
            final = progress.candidates[0].text
        else:
            final = ""
        return RunResult(
            query_class=progress.query_class,
            requested_decision=progress.requested_decision,
            decision=progress.decision,
            final_answer=final,
            disagreement=progress.disagreement,
            degraded=progress.degraded,
            timeout_partial=True,
            proposer_models=progress.proposer_models,
        )


async def _apply_guardrail(guardrail: Guardrail, text: str) -> str:
    out = guardrail(text)
    if inspect.isawaitable(out):
        return await out
    return out

"""Stage 1: run the proposer pool in parallel, tolerate partial failure.

All proposers run concurrently through the gateway (so each inherits timeout,
retry, fallback, and the circuit breaker). A failure of one proposer never
fails the request: as long as a quorum of proposers succeed, the council
proceeds with the survivors. Below quorum, we degrade to a single-model answer
(reusing a survivor if there is one, else one best-model call).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage, GatewayError, ProviderResult, Usage
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("council.proposers")

_PROPOSER_MAX_TOKENS = 1024
_PROPOSER_TEMPERATURE = 0.7


class ProposerOutput(BaseModel):
    model: str
    ok: bool
    text: str = ""
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    used_fallback: bool = False
    error: str | None = None


@dataclass
class ProposalRound:
    outputs: list[ProposerOutput]
    quorum: int

    @property
    def successes(self) -> list[ProposerOutput]:
        return [o for o in self.outputs if o.ok]

    @property
    def failures(self) -> list[ProposerOutput]:
        return [o for o in self.outputs if not o.ok]

    @property
    def met(self) -> bool:
        return len(self.successes) >= self.quorum


@dataclass
class Stage1Result:
    round: ProposalRound
    degraded: bool
    single: ProposerOutput | None


def build_proposer_messages(
    prompt_template: str, query: str, context: str = ""
) -> list[ChatMessage]:
    body = prompt_template.replace("{{conversation_context}}", context).replace(
        "{{query}}", query
    )
    return [ChatMessage(role="user", content=body)]


def _effective_quorum(config: AppConfig, n: int) -> int:
    return max(1, min(config.council.quorum, n))


def _success_output(model: str, result: ProviderResult) -> ProposerOutput:
    return ProposerOutput(
        model=model,
        ok=True,
        text=result.text,
        usage=result.usage,
        cost_usd=result.cost_usd,
        latency_ms=result.latency_ms,
        used_fallback=result.used_fallback,
    )


async def gather_proposals(
    gateway: LLMGateway,
    config: AppConfig,
    messages: list[ChatMessage],
    *,
    proposers: list[str] | None = None,
) -> ProposalRound:
    names = list(proposers) if proposers is not None else list(config.council.proposers)
    quorum = _effective_quorum(config, len(names))
    tasks = [
        gateway.complete(
            name, messages, max_tokens=_PROPOSER_MAX_TOKENS, temperature=_PROPOSER_TEMPERATURE
        )
        for name in names
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: list[ProposerOutput] = []
    for name, res in zip(names, results, strict=True):
        if isinstance(res, BaseException):
            outputs.append(
                ProposerOutput(model=name, ok=False, error=str(res) or type(res).__name__)
            )
            log.warning("council.proposer_failed", model=name, error=str(res))
        elif res.text and res.text.strip():
            outputs.append(_success_output(name, res))
        else:
            # A successful HTTP call that returns empty/whitespace content is NOT
            # a usable proposal. Count it as a failure so quorum and graceful
            # degradation work, instead of letting a blank draft pass as success.
            outputs.append(ProposerOutput(model=name, ok=False, error="empty response"))
            log.warning("council.proposer_empty", model=name)
    rnd = ProposalRound(outputs=outputs, quorum=quorum)
    if not rnd.met:
        log.warning("council.below_quorum", successes=len(rnd.successes), quorum=quorum)
    return rnd


async def run_stage1(
    gateway: LLMGateway,
    config: AppConfig,
    messages: list[ChatMessage],
    *,
    proposers: list[str] | None = None,
) -> Stage1Result:
    rnd = await gather_proposals(gateway, config, messages, proposers=proposers)
    if rnd.met:
        return Stage1Result(round=rnd, degraded=False, single=None)
    single = await _degrade_to_single(gateway, config, messages, rnd)
    return Stage1Result(round=rnd, degraded=True, single=single)


async def _degrade_to_single(
    gateway: LLMGateway,
    config: AppConfig,
    messages: list[ChatMessage],
    rnd: ProposalRound,
) -> ProposerOutput:
    # Prefer reusing a survivor (best by configured proposer order).
    for name in config.council.proposers:
        for out in rnd.successes:
            if out.model == name:
                log.info("council.degrade_reuse", model=name)
                return out
    # Nobody survived: one more attempt with the best available model.
    model = config.council.proposers[0] if config.council.proposers else config.router.router_model
    log.warning("council.degrade_single_call", model=model)
    try:
        res = await gateway.complete(
            model, messages, max_tokens=_PROPOSER_MAX_TOKENS, temperature=0.0
        )
        return _success_output(model, res)
    except GatewayError as exc:
        return ProposerOutput(model=model, ok=False, error=str(exc))

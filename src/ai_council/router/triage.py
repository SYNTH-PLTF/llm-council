"""Triage: classify a query and pick a routing decision.

Order of decision (cheapest first):
  1. Deterministic rule overrides (explicit flags, very short queries) - no LLM.
  2. A cheap-model classifier prompted for strict JSON. Invalid or unparseable
     output falls back to the safe default (standard -> single_model).
  3. The budget/latency policy gate, which can downgrade council -> single_model.

The classifier is hidden behind the ``Router`` protocol so a learned router can
replace ``LLMRouter`` without touching callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel

from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage, GatewayError
from ai_council.router.policy import CouncilPolicy
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("router")

QueryClass = Literal["trivial", "standard", "high_stakes", "verifiable_reasoning"]
RoutingDecision = Literal["single_model", "council", "council_with_voting"]
TriageSource = Literal["rules", "classifier", "fallback", "policy"]

_VALID_CLASSES: set[str] = {"trivial", "standard", "high_stakes", "verifiable_reasoning"}
_VALID_DECISIONS: set[str] = {"single_model", "council", "council_with_voting"}


class RouteRequest(BaseModel):
    query: str
    force_council: bool = False
    force_single: bool = False
    budget_remaining_usd: float | None = None
    user_daily_remaining_usd: float | None = None


class TriageResult(BaseModel):
    query_class: QueryClass
    decision: RoutingDecision
    reason: str = ""
    source: TriageSource = "classifier"


class Router(Protocol):
    async def route(self, request: RouteRequest) -> TriageResult: ...


def load_triage_prompt(config_dir: str = "config/prompts") -> str:
    return Path(config_dir, "triage.md").read_text(encoding="utf-8")


class LLMRouter:
    """The default rules + cheap-classifier + budget-gate router."""

    def __init__(
        self,
        config: AppConfig,
        gateway: LLMGateway,
        *,
        prompt: str | None = None,
        policy: CouncilPolicy | None = None,
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._prompt = prompt if prompt is not None else load_triage_prompt()
        self._policy = policy or CouncilPolicy(config)

    async def route(self, request: RouteRequest) -> TriageResult:
        rules = self._config.router.rules
        if request.force_single:
            return self._gate(request, "standard", "single_model", "rules", "forced single")
        if request.force_council and rules.force_council_on_user_flag:
            return self._gate(request, "high_stakes", "council", "rules", "forced council")
        if len(request.query.split()) < rules.force_single_if_tokens_lt:
            return self._gate(request, "trivial", "single_model", "rules", "short query")
        qclass, reason, source = await self._classify(request.query)
        return self._gate(request, qclass, self._decision_for(qclass), source, reason)

    def _decision_for(self, qclass: QueryClass) -> RoutingDecision:
        cr = self._config.class_routing
        mapping: dict[QueryClass, str] = {
            "trivial": cr.trivial,
            "standard": cr.standard,
            "high_stakes": cr.high_stakes,
            "verifiable_reasoning": cr.verifiable_reasoning,
        }
        mapped = mapping[qclass]
        if mapped in _VALID_DECISIONS:
            return cast(RoutingDecision, mapped)
        return "single_model"

    async def _classify(self, query: str) -> tuple[QueryClass, str, TriageSource]:
        prompt = self._prompt.replace("{{query}}", query)
        try:
            result = await self._gateway.complete(
                self._config.router.router_model,
                [ChatMessage(role="user", content=prompt)],
                max_tokens=200,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(_first_json_object(result.text))
            qclass = data.get("query_class")
            if qclass in _VALID_CLASSES:
                return cast(QueryClass, qclass), str(data.get("reason", ""))[:200], "classifier"
            log.warning("router.invalid_class", got=str(qclass))
        except (GatewayError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("router.classify_failed", error=str(exc))
        return "standard", "classifier fallback (safe default)", "fallback"

    def _gate(
        self,
        request: RouteRequest,
        qclass: QueryClass,
        decision: RoutingDecision,
        source: TriageSource,
        reason: str,
    ) -> TriageResult:
        final, gate_reason = self._policy.gate(
            decision,
            budget_remaining_usd=request.budget_remaining_usd,
            user_daily_remaining_usd=request.user_daily_remaining_usd,
            query=request.query,
        )
        if final != decision:
            return TriageResult(
                query_class=qclass, decision=final, reason=gate_reason or reason, source="policy"
            )
        return TriageResult(query_class=qclass, decision=decision, reason=reason, source=source)


def _first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no json object in classifier output")
    return text[start : end + 1]

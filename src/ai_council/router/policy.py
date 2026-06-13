"""Budget / latency policy: the router's cost guardian.

A council of n proposers costs roughly n generation calls + n ranking passes +
1 chairman synthesis (the chairman sees all proposer outputs, so its input
grows with n). If the estimated council cost exceeds the remaining per-request
or per-user budget, the policy downgrades the decision to single_model.
"""

from __future__ import annotations

from typing import Literal

from ai_council.settings import AppConfig

RoutingDecision = Literal["single_model", "council", "council_with_voting"]

_DEFAULT_IN_TOKENS = 400
_DEFAULT_OUT_TOKENS = 600
_RANK_OUT_TOKENS = 120


class CouncilPolicy:
    def __init__(self, config: AppConfig) -> None:
        self._c = config

    def _call_cost(self, model: str, in_tokens: int, out_tokens: int) -> float:
        price = self._c.pricing_usd_per_1k_tokens.get(model)
        if price is None:
            return 0.0
        return in_tokens / 1000.0 * price.input + out_tokens / 1000.0 * price.output

    def estimate_cost_usd(self, query: str) -> float:
        """Rough upper-ish estimate of one full council run, in USD."""
        council = self._c.council
        n = len(council.proposers)
        in_toks = max(_DEFAULT_IN_TOKENS, len(query.split()) * 2)
        total = 0.0
        for name in council.proposers:
            total += self._call_cost(name, in_toks, _DEFAULT_OUT_TOKENS)
        for name in council.proposers:
            total += (
                self._call_cost(name, in_toks * n, _RANK_OUT_TOKENS)
                * council.ranking.orderings_per_judge
            )
        total += self._call_cost(
            council.chairman.model, in_toks * n, council.chairman.max_output_tokens
        )
        return total

    def gate(
        self,
        decision: RoutingDecision,
        *,
        budget_remaining_usd: float | None = None,
        user_daily_remaining_usd: float | None = None,
        query: str = "",
    ) -> tuple[RoutingDecision, str | None]:
        """Return the (possibly downgraded) decision and a reason if changed."""
        if decision == "single_model":
            return decision, None
        est = self.estimate_cost_usd(query)
        if budget_remaining_usd is not None and est > budget_remaining_usd:
            return "single_model", (
                f"council est ${est:.4f} > per-request budget ${budget_remaining_usd:.4f}"
            )
        if user_daily_remaining_usd is not None and est > user_daily_remaining_usd:
            return "single_model", (
                f"council est ${est:.4f} > daily remaining ${user_daily_remaining_usd:.4f}"
            )
        return decision, None

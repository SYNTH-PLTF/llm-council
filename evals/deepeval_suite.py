"""Optional DeepEval suite: programmatic answer-quality metrics for the council.

Not part of the core pytest gate (it needs a real LLM judge). Enable with:

    uv add deepeval
    uv run python evals/deepeval_suite.py        # uses AI_COUNCIL_CONFIG_PATH

It scores each council answer and prints the metric, complementing the
deterministic uplift in ai_council.evals.harness.
"""

from __future__ import annotations

import asyncio


async def _run() -> None:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    from ai_council.council.orchestrator import Orchestrator, RunContext
    from ai_council.evals.run import load_dataset
    from ai_council.gateway.client import LLMGateway
    from ai_council.settings import get_config

    config = get_config()
    orchestrator = Orchestrator(config, LLMGateway(config))
    metric = AnswerRelevancyMetric()
    for item in load_dataset():
        result = await orchestrator.run(
            RunContext(correlation_id="deepeval", query=item.query, force_council=True)
        )
        case = LLMTestCase(input=item.query, actual_output=result.final_answer)
        metric.measure(case)
        print(f"{item.query_class:24} relevancy={metric.score}")


if __name__ == "__main__":
    asyncio.run(_run())

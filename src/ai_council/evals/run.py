"""`make eval` entrypoint: run the offline eval and print per-class uplift + cost.

Uses the active config (set AI_COUNCIL_CONFIG_PATH=config/test.yaml to run the
cheap pool). Datasets are the JSON files under evals/datasets/.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ai_council.council.orchestrator import Orchestrator
from ai_council.evals.harness import EvalItem, decision_overrides, run_eval
from ai_council.gateway.client import LLMGateway
from ai_council.settings import get_config


def load_dataset(directory: str = "evals/datasets") -> list[EvalItem]:
    items: list[EvalItem] = []
    for path in sorted(Path(directory).glob("*.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            items.append(EvalItem.model_validate(raw))
    return items


async def _run() -> None:
    config = get_config()
    orchestrator = Orchestrator(config, LLMGateway(config))
    report = await run_eval(orchestrator, load_dataset())

    header = f"{'class':<22}{'n':>4}{'council':>9}{'single':>9}{'uplift':>9}{'cost$':>9}"
    print(header)
    print("-" * len(header))
    for name, cls in sorted(report.classes.items()):
        cost = cls.council_cost + cls.single_cost
        print(
            f"{name:<22}{cls.n:>4}{cls.council_avg:>9.3f}{cls.single_avg:>9.3f}"
            f"{cls.uplift:>+9.3f}{cost:>9.4f}"
        )
    print(f"\ntotal cost: ${report.total_cost():.4f}")

    overrides = decision_overrides(report)
    if overrides:
        print(f"decision-rule overrides (route to single_model): {overrides}")
    else:
        print("decision rule: council shows uplift on all classes")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

"""Phase 4 verification: anonymization, order-swap averaging, Borda, disagreement."""

from __future__ import annotations

import json
import random
import re
from types import SimpleNamespace
from typing import Any

import pytest

from ai_council.council.ranking import (
    Candidate,
    Ranker,
    average_positions,
    borda_consensus,
    disagreement,
    kendall_tau,
    label_candidates,
    letters_to_ids,
    positions,
)
from ai_council.gateway.client import LLMGateway
from ai_council.settings import AppConfig

# --- pure functions ---------------------------------------------------------


def test_label_is_randomized_and_reversible() -> None:
    cands = [
        Candidate(id="x", text="tx"),
        Candidate(id="y", text="ty"),
        Candidate(id="z", text="tz"),
    ]
    seen_orderings: set[tuple[str, ...]] = set()
    for seed in range(12):
        _labeled, mapping = label_candidates(cands, random.Random(seed))
        assert set(mapping.keys()) == {"A", "B", "C"}
        assert set(mapping.values()) == {"x", "y", "z"}  # bijective
        recovered = letters_to_ids(["A", "B", "C"], mapping, ["x", "y", "z"])
        assert recovered == [mapping["A"], mapping["B"], mapping["C"]]  # reversible
        seen_orderings.add(tuple(mapping[letter] for letter in ("A", "B", "C")))
    assert len(seen_orderings) > 1  # the mapping actually varies (randomized)


def test_letters_to_ids_handles_invalid_and_missing() -> None:
    mapping = {"A": "x", "B": "y", "C": "z"}
    # 'Q' is invalid; 'C' (z) omitted by the judge -> appended deterministically.
    assert letters_to_ids(["B", "Q", "A"], mapping, ["x", "y", "z"]) == ["y", "x", "z"]


def test_positions_and_order_swap_average() -> None:
    assert positions(["x", "y", "z"], ["x", "y", "z"]) == {"x": 0, "y": 1, "z": 2}
    averaged = average_positions([{"x": 0, "y": 1, "z": 2}, {"x": 2, "y": 1, "z": 0}])
    assert averaged == {"x": 1.0, "y": 1.0, "z": 1.0}


def test_borda_unanimous_fixture() -> None:
    judge = {"x": 0.0, "y": 1.0, "z": 2.0}
    ordering, borda, _mean = borda_consensus([dict(judge), dict(judge)], ["x", "y", "z"])
    assert ordering == ["x", "y", "z"]
    assert borda == {"x": 4.0, "y": 2.0, "z": 0.0}


def test_borda_split_is_a_tie() -> None:
    j1 = {"x": 0.0, "y": 1.0, "z": 2.0}
    j2 = {"x": 2.0, "y": 1.0, "z": 0.0}
    ordering, borda, _mean = borda_consensus([j1, j2], ["x", "y", "z"])
    assert borda == {"x": 2.0, "y": 2.0, "z": 2.0}
    assert ordering == ["x", "y", "z"]  # tie broken by mean then id


def test_disagreement_zero_when_unanimous() -> None:
    judge = {"x": 0.0, "y": 1.0, "z": 2.0}
    assert disagreement([dict(judge), dict(judge)], ["x", "y", "z"]) == 0.0


def test_disagreement_one_when_reversed() -> None:
    j1 = {"x": 0.0, "y": 1.0, "z": 2.0}
    j2 = {"x": 2.0, "y": 1.0, "z": 0.0}
    assert disagreement([j1, j2], ["x", "y", "z"]) == pytest.approx(1.0)


def test_kendall_self_is_one() -> None:
    judge = {"x": 0.0, "y": 1.0, "z": 2.0}
    assert kendall_tau(judge, judge, ["x", "y", "z"]) == 1.0


# --- end-to-end with a deterministic "smart" judge --------------------------


def _icfg() -> AppConfig:
    return AppConfig.model_validate(
        {
            "models": {"J1": {"env_prefix": "J1"}, "J2": {"env_prefix": "J2"}},
            "router": {"router_model": "J1"},
            "council": {
                "proposers": ["J1", "J2"],
                "chairman": {"model": "J1"},
                "ranking": {"orderings_per_judge": 2},
            },
            "gateway": {"max_retries": 0, "backoff_base_s": 0.0},
        }
    )


async def _no_sleep(_delay: float) -> None:
    return None


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )


@pytest.fixture(autouse=True)
def _endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    for prefix in ("J1", "J2"):
        monkeypatch.setenv(f"{prefix}_BASE_URL", "https://example.test/v1")
        monkeypatch.setenv(f"{prefix}_API_KEY", "test-key")


async def test_ranker_orders_by_quality_regardless_of_anonymization() -> None:
    # A judge that ranks purely by the content quality, blind to identity/order.
    async def smart_judge(**kwargs: Any) -> Any:
        content = str(kwargs["messages"][0]["content"])
        pairs = re.findall(r"([A-Z]): quality=(\d+)", content)
        ranked = [letter for letter, _ in sorted(pairs, key=lambda p: int(p[1]), reverse=True)]
        return _resp(json.dumps({"ranking": ranked}))

    cands = [
        Candidate(id="x", text="quality=1"),
        Candidate(id="y", text="quality=3"),
        Candidate(id="z", text="quality=2"),
    ]
    gw = LLMGateway(_icfg(), completion_fn=smart_judge, sleep_fn=_no_sleep)
    ranker = Ranker(
        _icfg(), gw, prompt="Q: {{query}}\n{{labeled_candidates}}", rng=random.Random(0)
    )

    result = await ranker.rank("question", cands)

    assert result.ordering == ["y", "z", "x"]  # by quality desc
    assert result.top_k(2) == ["y", "z"]
    assert result.disagreement == 0.0
    assert result.judge_count == 2

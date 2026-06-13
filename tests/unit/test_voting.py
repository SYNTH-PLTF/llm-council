"""Phase 5 verification: answer extraction and majority voting (no LLM judge)."""

from __future__ import annotations

from ai_council.council.voting import extract_answer, majority_vote


def test_extract_various_formats() -> None:
    assert extract_answer("The final answer: 42") == "42"
    assert extract_answer("answer = 7") == "7"
    assert extract_answer("blah\nThe answer is Paris.") == "paris"
    assert extract_answer(r"so \boxed{x+1} done") == "x+1"
    assert extract_answer("reasoning here\n42") == "42"  # fallback = last line
    assert extract_answer("") is None


def test_majority_clear_winner() -> None:
    res = majority_vote(
        [("A", "answer: 42"), ("B", "answer: 42"), ("C", "answer: 7")], ["A", "B", "C"]
    )
    assert res.winner == "42"
    assert res.counts == {"42": 2, "7": 1}
    assert set(res.support) == {"A", "B"}
    assert res.tie_break_used is False


def test_majority_tie_breaks_by_quality_order() -> None:
    res = majority_vote([("A", "answer: 42"), ("B", "answer: 7")], ["A", "B"])
    assert res.winner == "42"  # A outranks B in quality order
    assert res.tie_break_used is True


def test_majority_no_extraction_returns_none() -> None:
    res = majority_vote([("A", ""), ("B", "")], ["A", "B"])
    assert res.winner is None

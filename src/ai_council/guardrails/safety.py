"""Output safety and PII scanning for the final answer.

These flag (and optionally redact) rather than silently drop: the orchestrator
records the flags on the run so problems are visible, and policy decides whether
to withhold. Rules are deliberately simple and dependency-free; swap in a model
or a dedicated library (Presidio, etc.) for stronger coverage.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
}

_DEFAULT_BLOCKLIST = (
    "how to build a bomb",
    "how to make a weapon",
    "synthesize a nerve agent",
)


class SafetyReport(BaseModel):
    flagged: bool = False
    pii: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


def scan_pii(text: str) -> list[str]:
    return sorted({name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)})


def safety_check(text: str, *, blocklist: tuple[str, ...] = _DEFAULT_BLOCKLIST) -> list[str]:
    low = text.lower()
    return [term for term in blocklist if term in low]


def redact_pii(text: str) -> str:
    redacted = text
    for name, pattern in _PII_PATTERNS.items():
        redacted = pattern.sub(f"[redacted-{name}]", redacted)
    return redacted


def evaluate(text: str, *, pii_scan: bool = True, safety: bool = True) -> SafetyReport:
    pii = scan_pii(text) if pii_scan else []
    reasons = safety_check(text) if safety else []
    return SafetyReport(flagged=bool(pii or reasons), pii=pii, reasons=reasons)

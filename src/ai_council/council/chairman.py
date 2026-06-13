"""Stage 3: the Chairman synthesizes one final answer with structured output.

The Chairman (a large-context model) reads the top-ranked candidate answers and
returns a structured verdict: the final answer plus its confidence, explicit
dissent notes (so disagreement is surfaced, not hidden), and which candidates
informed it. Malformed output is repaired once, then fails with a typed error.
The API streams ``final_answer`` to the client and attaches the rest as metadata.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ai_council.council.ranking import Candidate
from ai_council.gateway.client import LLMGateway
from ai_council.gateway.models import ChatMessage, GatewayError
from ai_council.settings import AppConfig
from ai_council.telemetry.logging import get_logger

log = get_logger("council.chairman")


class ChairmanError(Exception):
    """The Chairman could not produce schema-valid output, even after repair."""


class ChairmanVerdict(BaseModel):
    final_answer: str
    confidence: Literal["low", "medium", "high"] = "medium"
    dissent_notes: str = ""
    contributing_sources: list[str] = Field(default_factory=list)


class Chairman:
    def __init__(
        self, config: AppConfig, gateway: LLMGateway, *, prompt: str | None = None
    ) -> None:
        self._config = config
        self._gateway = gateway
        self._prompt = prompt if prompt is not None else load_chairman_prompt()

    def _format_candidates(self, top: list[Candidate]) -> str:
        return "\n\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(top))

    async def synthesize(self, query: str, top: list[Candidate]) -> ChairmanVerdict:
        prompt = self._prompt.replace("{{query}}", query).replace(
            "{{top_k_candidates}}", self._format_candidates(top)
        )
        verdict = await self._call_and_parse(prompt)
        if verdict is not None:
            return verdict
        repair = (
            prompt
            + "\n\nYour previous reply was not valid JSON for the required schema. "
            "Reply with ONLY the JSON object, nothing else."
        )
        verdict = await self._call_and_parse(repair)
        if verdict is not None:
            return verdict
        raise ChairmanError("chairman returned no schema-valid output after one repair")

    async def _call_and_parse(self, prompt: str) -> ChairmanVerdict | None:
        try:
            result = await self._gateway.complete(
                self._config.council.chairman.model,
                [ChatMessage(role="user", content=prompt)],
                max_tokens=self._config.council.chairman.max_output_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(_first_json_object(result.text))
            return ChairmanVerdict.model_validate(data)
        except (GatewayError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            log.warning("council.chairman_parse_failed", error=str(exc))
            return None


async def stream_text(text: str, chunk_size: int = 64) -> AsyncIterator[str]:
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def load_chairman_prompt(config_dir: str = "config/prompts") -> str:
    return Path(config_dir, "chairman.md").read_text(encoding="utf-8")


def _first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no json object in chairman output")
    return text[start : end + 1]

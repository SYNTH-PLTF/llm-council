"""Async repository: persist a full run (with stages + proposer outputs) and
reload it for audit/replay. Structurally satisfies the orchestrator's RunStore.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from ai_council.council.orchestrator import RunResult
from ai_council.council.proposers import ProposerOutput
from ai_council.persistence.models import Run, RunProposer, RunStage


class SqlRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def save_run(
        self,
        *,
        result: RunResult,
        query: str,
        proposer_outputs: list[ProposerOutput] | None = None,
        conversation_id: str | None = None,
    ) -> str:
        run_id = result.correlation_id or uuid.uuid4().hex
        async with self._sf() as session, session.begin():
            run = Run(
                id=run_id,
                conversation_id=conversation_id,
                query=query,
                query_class=result.query_class,
                requested_decision=result.requested_decision,
                decision=result.decision,
                final_answer=result.final_answer,
                confidence=result.confidence,
                dissent_notes=result.dissent_notes,
                disagreement=result.disagreement,
                degraded=result.degraded,
                timeout_partial=result.timeout_partial,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
            for i, stage in enumerate(result.stages):
                run.stages.append(RunStage(seq=i, name=stage.name, detail=stage.detail))
            for out in proposer_outputs or []:
                run.proposers.append(
                    RunProposer(
                        model=out.model,
                        ok=out.ok,
                        text=out.text,
                        cost_usd=out.cost_usd,
                        latency_ms=out.latency_ms,
                        error=out.error,
                    )
                )
            session.add(run)
        return run_id

    async def get_run(self, run_id: str) -> Run | None:
        async with self._sf() as session:
            stmt = (
                select(Run)
                .where(Run.id == run_id)
                .options(selectinload(Run.stages), selectinload(Run.proposers))
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

# AI Council - Project Conventions

## What this is
A selective, production-grade multi-LLM council. A router decides single-model
vs council. Council = parallel proposers -> debiased peer ranking ->
(debate | vote) -> Chairman synthesis, with full observability, evals, cost
control, and graceful degradation. The 3-stage idea comes from
karpathy/llm-council; everything around it is rebuilt for production.

## Golden rules
- Route first; never run the council on every query.
- Proposers must be comparable-quality and genuinely diverse. No weak models
  in the pool.
- Always anonymize and order-swap-average when ranking (position bias).
- Bound debate rounds; enforce termination and a wall-clock budget.
- Verifiable tasks (math/code) -> majority vote, not an LLM judge.
- Tolerate partial failures via quorum, else degrade to single-model.
- Config-driven: no hardcoded models, prices, or thresholds in business logic.
- Every LLM call goes through gateway/ and is traced to Langfuse.
- Prove uplift with evals; route a class to single-model if the council does
  not beat it.

## Stack
Python 3.11+ (venv pinned to 3.12) / uv / FastAPI / Pydantic v2 / LiteLLM
gateway / Postgres (async + Alembic) / Redis / Langfuse + OTel / promptfoo +
DeepEval / React + Vite / Docker compose / GitHub Actions.

## Workflow
Phase by phase (see BUILD_SPEC.md). Run Verify and commit each phase. Mock the
gateway in all non-load tests. Conventional commits (feat:, fix:, test:,
chore:). When unsure of a model string, price, or API, check docs at build time
and record the resolved value in config with a comment.

## Output style
ASCII only. No em dashes (use commas, periods, or parentheses). No emojis.

## Layout
See the repo tree in BUILD_SPEC.md section 3.

## Decisions log (documented per build-spec section 8)
- Python: venv pinned to 3.12 (.python-version) for broad wheel coverage;
  requires-python >=3.11. The local interpreter is 3.14 but we do not build
  against it, to avoid missing compiled wheels.
- Provider shape: the council models are reached as OpenAI-compatible endpoints
  (per-model Target URI + key), resolved at run time from env vars named by
  each model's env_prefix (see config). Keys live only in a gitignored .env,
  never in config or git.
- Two config profiles: config/default.yaml (frontier production pool:
  claude-opus-4-8, gpt-5.4, grok-4.3; chairman claude-opus-4-8; router
  claude-sonnet-4-6) and config/test.yaml (cheap pool: claude-sonnet-4-6,
  DeepSeek-V4-Pro, Kimi-K2.6; router gpt-chat-latest). Local runs and the
  end-to-end smoke test use the cheap profile.
- Docker is not installed in the dev environment. All phases are built so the
  test suite runs without Docker (fakeredis + aiosqlite in later phases); the
  real Postgres/Redis/Langfuse stack ships via docker compose for when Docker
  is available. `docker compose up` (Definition of Done item 1) needs Docker
  Desktop.
- Pyright runs in "standard" mode initially; can tighten to "strict" later.
- The model endpoints in the local key folder are for running and testing this
  service only.

## Verify commands
- make check     (ruff + pyright + pytest)
- make test-cov
- make eval      (Phase 11+)

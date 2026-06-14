# AI Council

A selective, production-grade multi-LLM council. A router decides whether a
query needs one strong model or a full council. The council runs parallel
proposers, debiased peer ranking, optional bounded debate or majority voting,
and a Chairman synthesis, with graceful degradation, cost control,
observability, and evals that prove the council earns its cost.

The 3-stage idea (answer, rank, synthesize) comes from `karpathy/llm-council`;
everything around it here is rebuilt for real deployment.

## Core design principle

Do not convene the council on every query. Mixing models often lowers quality
versus a single strong model, and naive multi-agent systems fail often. The
production win comes from a router that decides when a council is worth it, a
debiased judge (anonymized + order-swap averaged), graceful degradation, and
evals that prove uplift. This is a selective council, not an always-on one.

## Architecture

```
client
  -> FastAPI ingress (auth, rate-limit, idempotency, security headers)
  -> Triage/Router (classify -> decision; budget guardian)
       -> single_model: one strong model
       -> council:
            Stage 1 Proposers (parallel, quorum, degrade-to-single)
            Stage 2 Rank (anonymize + order-swap-average -> Borda + disagreement)
                    or Vote (verifiable tasks: majority vote, no LLM judge)
            (optional) one bounded debate round when judges disagree
            Stage 3 Chairman synthesis (structured: answer, confidence, dissent)
  -> Guardrails (PII/safety/faithfulness flags)
  -> Persistence (Postgres) + Cache (Redis) + Observability (traces + Prometheus)
```

Every LLM call goes through one gateway (timeout, transient-only retries,
per-model fallback, circuit breaker, cost capture) and is traced.

## Quick start (local dev)

```bash
uv sync                 # install (Python pinned to 3.12 via .python-version)
make check              # ruff + pyright + pytest
make dev                # API at http://127.0.0.1:8000
curl localhost:8000/healthz
curl localhost:8000/metrics
```

Frontend:

```bash
cd frontend && npm install && npm run dev   # http://localhost:5173 (proxies /v1 to :8000)
npm run test                                # Vitest
```

Full stack (requires Docker):

```bash
cp .env.example .env     # fill per-model endpoints (see Configuration)
make up                  # docker compose: api, frontend, postgres, redis, langfuse
```

## Configuration

All policy lives in versioned config (no hardcoded models, prices, or
thresholds in business logic):

- `config/default.yaml` - production pool (frontier models).
- `config/test.yaml` - cheap pool for local runs and the smoke test. Select it
  with `AI_COUNCIL_CONFIG_PATH=config/test.yaml`.

Each model is reached as an OpenAI-compatible endpoint resolved at run time from
environment variables named by the model's `env_prefix` (see `.env.example`):
`<PREFIX>_BASE_URL`, `<PREFIX>_API_KEY`, and optional `<PREFIX>_MODEL`. Keys live
only in a gitignored `.env`, never in config or git.

Key env vars: `DATABASE_URL`, `REDIS_URL`, `AI_COUNCIL_API_AUTH_TOKEN` (enables
bearer auth when set), `AI_COUNCIL_CONFIG_PATH`, `LANGFUSE_*`.

## Routing decision table

| Query class | Default decision | Why |
|---|---|---|
| trivial | single_model | one model suffices |
| standard | single_model | one strong model suffices |
| high_stakes | council | multiple perspectives add value |
| verifiable_reasoning | council_with_voting | majority vote on extracted answers |

These defaults live in `config.class_routing` and are overridden by the eval
decision rule (a class with no measured uplift is routed to single_model).

## Cost model

A council of `n` proposers costs roughly `n` generation calls + `n` ranking
passes + 1 Chairman synthesis (the Chairman sees all proposer outputs, so its
input grows with `n`). The router estimates this and downgrades to single-model
when it exceeds the remaining per-request or per-user-daily budget; the
orchestrator re-checks actual cost before the Chairman call. Default `n = 3`,
hard cap `n = 5`. Fill `pricing_usd_per_1k_tokens` in config for accurate cost.

## How to add a model

1. Add an entry under `models:` in the config with a unique `env_prefix`, the
   `litellm_provider`, context window, and tier.
2. Add its `<PREFIX>_BASE_URL` / `<PREFIX>_API_KEY` to `.env`.
3. Reference it in `council.proposers`, `council.chairman.model`, or
   `router.router_model`. Add a `pricing_usd_per_1k_tokens` row.
4. Run `make eval` to confirm the new pool still shows council uplift.

## Evals

```bash
make eval        # per-class council-vs-single uplift + cost (uses the active config)
```

`make eval` runs every labeled item through the orchestrator twice (council vs
single), scores both, and prints per-class uplift. If a class shows no uplift it
is reported as a decision-rule override (route it to single_model). The uplift
logic and a regression gate are also exercised by `pytest` (so CI fails on a
seeded regression). A `promptfoo` config (`evals/promptfooconfig.yaml`) and an
optional DeepEval suite are included for judge-based comparison against a
running API.

## Observability and security

- One trace per request with nested spans (triage, each proposer, ranking,
  chairman). Optional Langfuse/OTel adapters; `/metrics` serves Prometheus.
- Bearer-token auth on `/v1/chat` (set `AI_COUNCIL_API_AUTH_TOKEN`), security
  headers, locked-down CORS, input size limits, PII/safety/faithfulness flags,
  and a non-blocking `pip-audit` in CI.

## Testing

```bash
make check                       # ruff + pyright + pytest (backend)
cd frontend && npm run test      # Vitest (frontend)
make load                        # locust load test (needs the stack up)
```

## Project layout

See `BUILD_SPEC.md` for the full phased plan and `CLAUDE.md` for conventions.
Backend in `src/ai_council/` (gateway, router, council, guardrails, persistence,
cache, observability, evals, api); frontend in `frontend/`; deployment via
`Dockerfile`, `frontend/Dockerfile`, and `docker-compose.yml`; CI in
`.github/workflows/ci.yml`.

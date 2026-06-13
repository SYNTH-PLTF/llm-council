# Build Spec: Production-Grade AI Council

Audience: Claude Code (Opus 4.8), executing autonomously in an agentic coding
session.
Goal: Build a production-grade "AI Council" service - multiple LLMs collaborate
to answer hard questions - using `karpathy/llm-council` as conceptual backbone,
but hardened for real deployment.
How to use this file: Save at repo root. Work through the phases in order.
After each phase, run its verification commands and commit. Do not skip the
eval and observability phases - they are what make this "production grade"
rather than a demo.

---

## 0. North Star and Core Design Principle

The backbone (`karpathy/llm-council`) is a 3-stage pipeline: (1) all models
answer in parallel, (2) each model ranks the others' anonymized answers, (3) a
"Chairman" model synthesizes a final answer. The backbone is a disposable
weekend hack with JSON-file storage, no error handling, no auth, no evals, no
cost control, and no observability.

Keep the 3-stage idea. Replace everything around it.

The single most important design decision, grounded in the research (Self-MoA,
Princeton ICLR 2025; MAST, Berkeley NeurIPS 2025):

> Do not convene the full council on every query. Mixing models often lowers
> quality versus a single strong model, and naive multi-agent systems fail
> 40-85% of the time. The production win comes from a router that decides when a
> council is worth it, a debiased judge, graceful degradation, and evals that
> prove the council beats a single model on the target workload.

Build a selective council, not an always-on council.

### Hard rules (non-negotiable; treat as acceptance constraints)
1. Router first. Cheap/clear queries go to one strong model. Only escalate to a
   council for high-stakes or genuinely diverse-task queries.
2. Debias the judge. Anonymize candidate identities AND evaluate every
   comparison in both orders, then average. Position bias alone can swing
   pairwise judgments by >10%.
3. Comparable-quality proposers. The proposer pool must be models of similar
   quality with genuine diversity. Never put a much weaker model in the pool.
4. Partial-failure tolerance. The council must produce a result if a quorum of
   proposers succeed. If below quorum, degrade to single-model. Never hang or
   500 because one provider timed out.
5. Bounded everything. Max debate rounds, max tokens, per-call timeouts,
   per-request budget. Enforce explicit termination conditions.
6. Verify the output. For verifiable tasks (math/code) use majority vote on
   extracted answers, not an LLM judge. For all tasks, run a final guardrail
   pass.
7. Prove value. Ship an eval harness. If the council does not beat single-best
   on a query class, route that class to single-best.
8. Config-driven. Every model, threshold, and policy lives in versioned config.
   No hardcoded model names or magic numbers in business logic.

---

## 1. Target Architecture

```
client
  -> FastAPI ingress (auth, rate-limit, idempotency)
  -> Triage/Router (classify query -> decision)
       -> single_model: one strong model (stream back)
       -> council:
            Stage 1 Proposers (parallel): N models, async, timeouts, retries,
              fallbacks, quorum
            Stage 2 Review and Rank: anonymize + order-swap-avg
              -> consensus rank + disagreement (verifiable -> majority vote)
            (optional) bounded debate round
            Stage 3 Chairman synthesis: structured output + dissent
       -> Guardrail / verification pass
  -> persistence (Postgres): full trace
  -> cache (Redis): proposer outputs
  -> observability (Langfuse/OTel): cost and latency metrics, scores
```

Cross-cutting: every LLM call goes through a gateway abstraction (LiteLLM) so
providers, fallbacks, caching, and budgets are centralized. Every call emits a
span to Langfuse.

Cost geometry warning: a council of n proposers with peer-ranking is n
generation calls + up to n^2 ranking comparisons + 1 synthesis. Context for the
Chairman grows with all proposer outputs. Budget and quorum logic must account
for this. Default n = 3. Do not exceed n = 5 without explicit config override.

---

## 2. Tech Stack (decisions + rationale)

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Async, ecosystem, parity with backbone |
| Pkg mgmt | uv | Parity, fast, lockfile |
| API | FastAPI + uvicorn | Async, SSE streaming, pydantic-native |
| Models/validation | Pydantic v2 | Strict typed I/O everywhere |
| LLM gateway | LiteLLM | One OpenAI-compatible interface; retries, fallbacks, budgets, caching |
| Provider access | OpenAI-compatible endpoints (Target URI + key) | The 8 local models |
| Router | rules + small-model classifier (pluggable) | Cheap, controllable |
| Persistence | PostgreSQL (asyncpg + SQLAlchemy 2.0 async) + Alembic | JSON files do not survive prod |
| Cache / rate-limit / locks | Redis | Exact + semantic cache, token-bucket, idempotency |
| Observability | Langfuse + OpenTelemetry | Per-call traces, prompt versioning, scores, cost |
| Orchestration | Plain async + small state machine (optionally LangGraph) | Council is a deterministic DAG |
| Offline evals | promptfoo (CI gate) + DeepEval | Prove council > single-best; regression gate |
| Frontend | React + Vite | Stage tabs, cost/latency badges, route + disagreement, streaming |
| Containerization | Docker + docker-compose | api, frontend, postgres, redis, langfuse |
| CI/CD | GitHub Actions | ruff + pyright + pytest + eval gate + build |

Action: verify the exact current model identifier strings against the chosen
provider's docs at build time. Put them in config only.

---

## 3. Repository Structure

```
ai-council/
  CLAUDE.md
  README.md
  pyproject.toml
  uv.lock
  .env.example
  docker-compose.yml
  Dockerfile
  Makefile
  config/
    default.yaml
    prompts/ (triage.md, proposer.md, ranker.md, chairman.md)
  src/ai_council/
    settings.py
    api/ (app.py, routes.py, auth.py, schemas.py)
    gateway/ (client.py, models.py)
    router/ (triage.py, policy.py)
    council/ (orchestrator.py, proposers.py, ranking.py, debate.py, voting.py, chairman.py)
    guardrails/ (safety.py, verify.py)
    persistence/ (db.py, models.py, repository.py)
    cache/ (redis_cache.py)
    observability/ (tracing.py, metrics.py)
    telemetry/ (logging.py)
  frontend/
  evals/ (datasets/, promptfooconfig.yaml, deepeval_suite.py)
  tests/ (unit/, integration/, load/)
  .github/workflows/ci.yml
```

---

## 4. Implementation Phases

Each phase ends with Verify (commands that must pass) and Commit. Use
conventional commits.

- Phase 0 - Scaffolding and conventions. uv init, deps, CLAUDE.md, .env.example,
  Makefile, pyproject with ruff + pyright + pytest. settings.py
  (pydantic-settings) loads config/default.yaml + env, fail fast on missing
  secrets. Structured JSON logging with per-request correlation_id middleware.
  Verify: ruff, pyright, clean import, make dev boots /healthz. Commit.
- Phase 1 - Gateway abstraction. complete()/stream(); per-call timeout; bounded
  jittered retries on transient (429/5xx/timeout) only, never 4xx; per-model
  fallback; per-provider circuit breaker; capture usage+cost+latency+
  finish_reason into typed ProviderResult. Verify: mocked unit tests for every
  path; cost correct. Commit.
- Phase 2 - Router/triage. Classify QueryClass {trivial, standard, high_stakes,
  verifiable_reasoning} -> RoutingDecision {single_model, council,
  council_with_voting} via cheap router_model (strict JSON) + rule overrides;
  Router interface. policy.py budget/latency gate. Verify. Commit.
- Phase 3 - Stage 1 proposers. asyncio.gather(return_exceptions=True) through
  the gateway; quorum; degrade to single-model. Verify. Commit.
- Phase 4 - Stage 2 ranking + debias. anonymize per-judge random map; order-swap
  averaging (>=2 orderings/judge); Borda consensus; disagreement metric. Verify.
  Commit.
- Phase 5 - bounded debate + voting. one bounded debate round with convergence;
  majority vote for verifiable tasks (skip LLM judge). Verify. Commit.
- Phase 6 - Stage 3 chairman synthesis (structured: final_answer, confidence,
  dissent_notes, contributing_sources); stream final_answer. Verify. Commit.
- Phase 7 - Orchestrator state machine; immutable RunContext carries
  correlation_id, budget, full history; wall-clock cap -> partial. Verify e2e
  per path. Commit.
- Phase 8 - Persistence: conversations, messages, runs, run_stages; async repo;
  replace JSON. Verify. Commit.
- Phase 9 - Observability and cost: nested Langfuse spans + OTel; custom scores;
  Prometheus /metrics. Verify. Commit.
- Phase 10 - Caching, rate limiting, budgets, idempotency (Redis). Verify.
  Commit.
- Phase 11 - Eval harness: promptfoo CI gate + DeepEval + council-uplift +
  shadow mode; decision rule (uplift<=0 -> single_model). Verify. Commit.
- Phase 12 - Guardrails and security: safety/PII + faithfulness; auth, input
  limits, no secret logging, CORS, dep audit. Verify. Commit.
- Phase 13 - Frontend: stage tabs, route badge, cost/latency badges,
  disagreement, dissent panel, streaming, honest degraded states. Verify.
  Commit.
- Phase 14 - Containerization, CI/CD, docs, load test. compose up healthy; make
  test/eval/load green; cold-clone. Tag v1.0.0.

---

## 5. Non-Functional Requirements (acceptance constraints)

- Reliability: no single provider failure can fail a request that has a quorum
  or a viable single-model fallback. All external calls bounded by timeout +
  retries + circuit breaker.
- Latency budget: council p95 <= configurable target (default 30s); single-model
  p95 <= 8s. Enforce wall-clock cap; return partial-with-flag rather than hang.
- Cost: every run records cost; per-user daily cap enforced; council only runs
  when budget allows.
- Idempotency: safe to retry any request with an idempotency key.
- Observability: 100% of LLM calls traced with cost/latency; one trace per
  request.
- Determinism in tests: the gateway is mockable; integration tests never hit
  real providers. Provide record/replay fixtures.
- Config over code: changing the model pool, thresholds, or routing defaults
  requires only a config edit + restart, no code change.

---

## 6. Anti-Patterns to Avoid (explicit "do NOT")

- Do NOT run the council on every query. Route first.
- Do NOT mix wildly different-quality models in the proposer pool.
- Do NOT rank candidates without anonymization and order-swap averaging.
- Do NOT let debate loop without a hard round cap and convergence check.
- Do NOT use an LLM judge to decide correctness on math/code; use majority vote.
- Do NOT store conversations in flat JSON files.
- Do NOT lose conversation history between stages.
- Do NOT swallow provider errors silently or block forever on a slow model.
- Do NOT hardcode model identifiers, prices, or thresholds in business logic.
- Do NOT ship without the eval harness proving council uplift.

---

## 7. Definition of Done

1. docker compose up yields a healthy stack (api, frontend, postgres, redis,
   langfuse).
2. A query flows end-to-end through router -> (single | council) -> guardrail ->
   response, with the final answer streamed and full trace persisted + visible
   in Langfuse.
3. Killing one proposer mid-flight still returns a correct answer (quorum);
   killing the pool degrades to single-model.
4. make test (unit + integration) green; coverage on gateway, router,
   council/* >= 80%.
5. make eval reports per-class council uplift and cost; CI eval gate active.
6. Auth, rate limiting, budgets, idempotency, and caching all demonstrably
   working.
7. README lets a new engineer clone and run the stack and the evals without
   help.

---

## 8. Build Discipline for Claude Code

- Work phase by phase; run the Verify step and commit before moving on. Do not
  write the whole app in one pass.
- After each phase, briefly summarize what changed and what the next phase
  needs.
- Prefer many small, focused diffs over large rewrites.
- When you hit an unknown (exact model string, current pricing, a library's
  current API), check the provider/library docs at build time rather than
  guessing; record the resolved value in config with a comment.
- Write tests alongside code, not after. Mock the gateway in all non-load tests.
- If a phase reveals the spec is ambiguous, make a reasonable, documented
  decision in CLAUDE.md and proceed; flag it in your summary.

---

## 9. Reference Config

See config/default.yaml (production pool) and config/test.yaml (cheap pool).
The whole file is the contract; business logic reads from there. Keys: gateway
(timeout, retries, circuit_breaker), router (router_model, budgets, rules),
council (proposers, quorum, judges, ranking, debate, voting, chairman),
class_routing, cache, guardrails, observability, pricing_usd_per_1k_tokens.

---

## 10. Prompt Templates

See config/prompts/ (triage.md, proposer.md, ranker.md, chairman.md). They
encode the non-obvious production decisions: strict JSON from triage,
anonymized + order-swap-averaged ranking, and structured Chairman synthesis
(final_answer, confidence, dissent_notes, contributing_sources).

---

## 11. CLAUDE.md

Created at repo root. Captures golden rules, stack, workflow, layout, and the
decisions log.

---

## 12. Reference Material

- Backbone: github.com/karpathy/llm-council - the 3-stage pattern to upgrade.
- Mixture-of-Agents: github.com/togethercomputer/MoA; arXiv:2406.04692.
- Self-MoA (why mixing can hurt): github.com/wenzhe-li/Self-MoA;
  arXiv:2502.00674.
- LLM-Blender (PairRanker + GenFuser): github.com/yuchenlin/LLM-Blender;
  arXiv:2306.02561.
- MAST (why multi-agent systems fail):
  github.com/multi-agent-systems-failure-taxonomy/MAST; arXiv:2503.13657.
- LLM Ensemble survey: arXiv:2502.18036;
  github.com/junchenzhi/Awesome-LLM-Ensemble.
- Routing: github.com/Not-Diamond/awesome-ai-model-routing; RouteLLM.
- Gateways: LiteLLM, OpenRouter, Portkey docs.
- Observability/evals: Langfuse, promptfoo, DeepEval docs.

Build it. Verify each phase. Make the evals prove the council is worth it.

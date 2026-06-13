# AI Council

A selective, production-grade multi-LLM council. A router decides whether a
query needs one strong model or a full council; the council runs parallel
proposers, debiased peer ranking, optional bounded debate or majority voting,
and a Chairman synthesis, with graceful degradation, cost control,
observability, and evals that prove the council earns its cost.

Status: under active construction (see `BUILD_SPEC.md` for the phased plan and
`CLAUDE.md` for conventions). This README is expanded in Phase 14.

## Quick start (dev)

```bash
make setup     # uv sync
make dev       # boot the API on http://127.0.0.1:8000
curl localhost:8000/healthz
```

## Configuration

All policy (models, thresholds, routing) lives in `config/`:

- `config/default.yaml` - production pool (frontier models).
- `config/test.yaml` - cheap pool for local runs and tests.

Per-model endpoint secrets are read from environment variables at run time
(see `.env.example`); they are never hardcoded or committed.

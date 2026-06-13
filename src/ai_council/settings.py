"""Application settings and typed configuration loading.

Two layers:

* ``AppSettings`` - process/runtime settings read from the environment (.env):
  which config file to load, log level, the API auth token.
* ``AppConfig`` - the versioned council policy loaded from a YAML file
  (``config/default.yaml`` or an overlay such as ``config/test.yaml``). This is
  the single contract the business logic reads from; no model names, prices, or
  thresholds are hardcoded anywhere else.

Per-model endpoint secrets (Target URI + key) are deliberately NOT loaded here.
The gateway resolves them lazily from environment variables named by each
model's ``env_prefix`` so raw keys never live in this module or in the config.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class CircuitBreakerConfig(BaseModel):
    fail_threshold: int = 5
    reset_after_s: float = 30.0


class GatewayConfig(BaseModel):
    provider: str = "openai_compatible"
    request_timeout_s: float = 45.0
    max_retries: int = 2
    backoff_base_s: float = 0.5
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)


class ModelSpec(BaseModel):
    """How to reach one model.

    ``env_prefix`` names the env vars that hold the endpoint:
    ``<PREFIX>_BASE_URL``, ``<PREFIX>_API_KEY``, and optional ``<PREFIX>_MODEL``
    (the deployment id sent to the endpoint).
    """

    env_prefix: str
    litellm_provider: str = "openai"
    context_window: int = 200_000
    tier: Literal["frontier", "mid", "cheap"] = "mid"
    fallback: str | None = None


class RouterRules(BaseModel):
    force_single_if_tokens_lt: int = 12
    force_council_on_user_flag: bool = True


class RouterBudgets(BaseModel):
    per_request_usd: float = 0.50
    per_user_daily_usd: float = 20.0
    latency_budget_s: float = 30.0


class RouterConfig(BaseModel):
    router_model: str
    default_decision: str = "single_model"
    rules: RouterRules = Field(default_factory=RouterRules)
    budgets: RouterBudgets = Field(default_factory=RouterBudgets)


class RankingConfig(BaseModel):
    method: Literal["borda", "mean_rank"] = "borda"
    orderings_per_judge: int = 2
    anonymize: bool = True


class DebateConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.4
    max_rounds: int = 1


class VotingConfig(BaseModel):
    enabled: bool = True
    tie_break_by_quality_order: bool = True


class ChairmanConfig(BaseModel):
    model: str
    structured_output: bool = True
    max_output_tokens: int = 2048


class CouncilConfig(BaseModel):
    proposers: list[str]
    chairman: ChairmanConfig
    quorum: int = 2
    judges: Literal["proposers", "separate_pool"] = "proposers"
    judge_pool: list[str] = Field(default_factory=list)
    ranking: RankingConfig = Field(default_factory=RankingConfig)
    debate: DebateConfig = Field(default_factory=DebateConfig)
    voting: VotingConfig = Field(default_factory=VotingConfig)


class ClassRoutingConfig(BaseModel):
    trivial: str = "single_model"
    standard: str = "single_model"
    high_stakes: str = "council"
    verifiable_reasoning: str = "council_with_voting"


class CacheConfig(BaseModel):
    exact_match: bool = True
    semantic: bool = False
    ttl_s: int = 86_400


class GuardrailsConfig(BaseModel):
    safety_check: bool = True
    pii_scan: bool = True
    verify_faithfulness: bool = True


class ObservabilityConfig(BaseModel):
    langfuse: bool = False
    otel: bool = False
    prometheus_metrics: bool = True


class ModelPricing(BaseModel):
    input: float = 0.0
    output: float = 0.0


class AppConfig(BaseModel):
    """The full versioned council policy. Read-only at runtime."""

    models: dict[str, ModelSpec]
    router: RouterConfig
    council: CouncilConfig
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    class_routing: ClassRoutingConfig = Field(default_factory=ClassRoutingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    pricing_usd_per_1k_tokens: dict[str, ModelPricing] = Field(default_factory=dict)

    def require_model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError as exc:
            raise KeyError(f"model '{name}' is not defined in config.models") from exc


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AI_COUNCIL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["dev", "test", "prod"] = "dev"
    config_path: Path = Path("config/default.yaml")
    log_level: str = "INFO"
    log_json: bool = True
    api_auth_token: str | None = None


@functools.lru_cache
def get_settings() -> AppSettings:
    return AppSettings()


@functools.lru_cache
def get_config(path: str | None = None) -> AppConfig:
    """Load and validate the YAML council policy. Fails fast and loudly."""
    cfg_path = Path(path) if path is not None else get_settings().config_path
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"config file not found: {cfg_path}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"config file is not a mapping: {cfg_path}")
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise RuntimeError(f"invalid config {cfg_path}:\n{exc}") from exc

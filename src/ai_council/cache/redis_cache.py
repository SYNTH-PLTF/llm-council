"""Redis-backed exact cache, idempotency, rate limiting, and spend ledger.

Uses redis.asyncio in production and fakeredis in tests (no Docker required).

  * Exact cache  - identical proposer prompts skip the provider entirely.
  * Idempotency  - a repeated Idempotency-Key returns the stored response.
  * Token bucket - per-API-key rate limiting (refill at rate, burst to capacity).
  * Spend ledger - per-user daily cost for budget enforcement.

Semantic cache is intentionally NOT enabled by default: embedding-similarity
hits can return subtly stale or wrong answers, so it is left as a documented,
opt-in extension rather than shipped on.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import Any

RedisClient = Any  # redis.asyncio.Redis or fakeredis FakeAsyncRedis (duck-typed)


def _normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _params_hash(params: dict[str, Any]) -> str:
    raw = repr(sorted((k, str(v)) for k, v in params.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def exact_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha256(_normalize(prompt).encode()).hexdigest()[:24]
    return f"cache:exact:{model}:{digest}:{_params_hash(params)}"


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _field(data: dict[Any, Any], name: str, default: float) -> float:
    for raw_key, raw_val in data.items():
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        if key == name:
            return float(raw_val.decode() if isinstance(raw_val, bytes) else raw_val)
    return default


class RedisCache:
    def __init__(
        self,
        redis: RedisClient,
        *,
        ttl_s: int = 86_400,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._r = redis
        self._ttl = ttl_s
        self._now = now

    # --- exact-match cache ---------------------------------------------------

    async def get_exact(self, model: str, prompt: str, params: dict[str, Any]) -> str | None:
        return _as_str(await self._r.get(exact_key(model, prompt, params)))

    async def set_exact(
        self, model: str, prompt: str, params: dict[str, Any], value: str
    ) -> None:
        await self._r.set(exact_key(model, prompt, params), value, ex=self._ttl)

    # --- idempotency ---------------------------------------------------------

    async def get_idempotent(self, key: str) -> str | None:
        return _as_str(await self._r.get(f"idem:{key}"))

    async def set_idempotent(self, key: str, value: str, *, ttl_s: int = 86_400) -> None:
        await self._r.set(f"idem:{key}", value, ex=ttl_s)

    # --- token-bucket rate limiting -----------------------------------------

    async def allow(self, key: str, *, rate_per_s: float, capacity: float) -> tuple[bool, float]:
        """Consume one token; return (allowed, retry_after_seconds)."""
        bucket = f"rl:{key}"
        now = self._now()
        data = await self._r.hgetall(bucket)
        elapsed = max(0.0, now - _field(data, "ts", now))
        tokens = min(capacity, _field(data, "tokens", capacity) + elapsed * rate_per_s)
        if tokens >= 1.0:
            tokens -= 1.0
            allowed, retry = True, 0.0
        else:
            allowed = False
            retry = (1.0 - tokens) / rate_per_s if rate_per_s > 0 else 60.0
        await self._r.hset(bucket, mapping={"tokens": tokens, "ts": now})
        ttl = math.ceil(capacity / rate_per_s) + 1 if rate_per_s > 0 else 60
        await self._r.expire(bucket, ttl)
        return allowed, retry

    # --- per-user daily spend ledger ----------------------------------------

    async def add_spend(self, user_id: str, cost_usd: float, *, day: str) -> float:
        key = f"spend:{user_id}:{day}"
        total = await self._r.incrbyfloat(key, cost_usd)
        await self._r.expire(key, 172_800)
        return float(total)

    async def get_spend(self, user_id: str, *, day: str) -> float:
        value = _as_str(await self._r.get(f"spend:{user_id}:{day}"))
        return float(value) if value is not None else 0.0


def make_redis(url: str) -> RedisClient:
    import redis.asyncio as aioredis

    return aioredis.from_url(url)

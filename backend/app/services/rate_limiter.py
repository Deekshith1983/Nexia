"""Reusable Token Bucket Rate Limiting Engine supporting In-Memory and Redis backends."""

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Dict, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("nexia.rate_limiter")


@dataclass
class RateLimitResult:
    """Outcome of a rate limit check."""

    allowed: bool
    remaining_tokens: float
    retry_after_seconds: float = 0.0
    capacity: int = 1
    refill_rate: float = 1.0


class RateLimitPolicy:
    """Defines bucket capacity and refill rate."""

    def __init__(self, capacity: int, refill_interval_seconds: float, cost: int = 1):
        self.capacity = max(1, capacity)
        self.refill_interval_seconds = max(0.001, refill_interval_seconds)
        # tokens added per second
        self.refill_rate = 1.0 / self.refill_interval_seconds
        self.cost = cost


class InMemoryTokenBucket:
    """In-memory Token Bucket with continuous monotonic refill and thread/task safety."""

    def __init__(self):
        # key -> (tokens: float, last_updated: float)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def consume(self, key: str, policy: RateLimitPolicy, cost: Optional[int] = None) -> RateLimitResult:
        consume_cost = cost if cost is not None else policy.cost
        now = time.monotonic()

        async with self._lock:
            if key in self._buckets:
                tokens, last_updated = self._buckets[key]
                elapsed = max(0.0, now - last_updated)
                tokens = min(float(policy.capacity), tokens + (elapsed * policy.refill_rate))
            else:
                tokens = float(policy.capacity)

            if tokens >= consume_cost:
                tokens -= consume_cost
                self._buckets[key] = (tokens, now)
                return RateLimitResult(
                    allowed=True,
                    remaining_tokens=tokens,
                    retry_after_seconds=0.0,
                    capacity=policy.capacity,
                    refill_rate=policy.refill_rate,
                )
            else:
                needed = consume_cost - tokens
                retry_after = round(needed / policy.refill_rate, 2)
                # Store updated tokens with current timestamp without deducting
                self._buckets[key] = (tokens, now)
                return RateLimitResult(
                    allowed=False,
                    remaining_tokens=tokens,
                    retry_after_seconds=max(0.1, retry_after),
                    capacity=policy.capacity,
                    refill_rate=policy.refill_rate,
                )

    async def clear_expired(self, max_idle_seconds: float = 3600.0) -> None:
        """Evict stale keys to prevent unbounded memory growth."""
        now = time.monotonic()
        async with self._lock:
            stale_keys = [
                k for k, (_, last_up) in self._buckets.items()
                if (now - last_up) > max_idle_seconds
            ]
            for k in stale_keys:
                self._buckets.pop(k, None)


# Redis Lua Script for Atomic Token Bucket consumption
REDIS_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if not tokens then
    tokens = capacity
    last_updated = now
else
    local elapsed = math.max(0, now - last_updated)
    tokens = math.min(capacity, tokens + (elapsed * refill_rate))
end

if tokens >= cost then
    tokens = tokens - cost
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', now)
    redis.call('EXPIRE', key, ttl)
    return {1, string.format('%.2f', tokens), 0}
else
    local needed = cost - tokens
    local retry_after = needed / refill_rate
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', now)
    redis.call('EXPIRE', key, ttl)
    return {0, string.format('%.2f', tokens), string.format('%.2f', retry_after)}
end
"""


class RedisTokenBucket:
    """Redis-backed Token Bucket with atomic Lua evaluation."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None
        self._lua_script = None

    async def _get_client(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
                self._lua_script = self._redis.register_script(REDIS_TOKEN_BUCKET_LUA)
            except Exception as e:
                logger.warning("Redis initialization failed: %s", e)
                raise
        return self._redis

    async def consume(self, key: str, policy: RateLimitPolicy, cost: Optional[int] = None) -> RateLimitResult:
        consume_cost = cost if cost is not None else policy.cost
        now = time.time()
        ttl = max(60, int(policy.capacity * policy.refill_interval_seconds * 2))

        try:
            client = await self._get_client()
            res = await self._lua_script(
                keys=[f"rl:{key}"],
                args=[policy.capacity, policy.refill_rate, consume_cost, now, ttl],
            )
            allowed = bool(res[0] == 1)
            remaining = float(res[1])
            retry_after = float(res[2])
            return RateLimitResult(
                allowed=allowed,
                remaining_tokens=remaining,
                retry_after_seconds=retry_after,
                capacity=policy.capacity,
                refill_rate=policy.refill_rate,
            )
        except Exception as e:
            logger.warning("Redis rate limiter error: %s", e)
            if settings.RATE_LIMIT_FAIL_OPEN:
                return RateLimitResult(allowed=True, remaining_tokens=1.0, retry_after_seconds=0.0)
            return RateLimitResult(allowed=False, remaining_tokens=0.0, retry_after_seconds=5.0)


class RateLimiterService:
    """Centralized rate limiting service providing endpoint policy resolution."""

    _instance: Optional["RateLimiterService"] = None

    def __init__(self):
        self.enabled = settings.RATE_LIMIT_ENABLED
        self.backend_type = settings.RATE_LIMIT_BACKEND
        self.in_memory = InMemoryTokenBucket()
        self.redis_bucket = RedisTokenBucket(settings.RATE_LIMIT_REDIS_URL) if self.backend_type == "redis" else None

        # Pre-configured policies
        self.policies = {
            "answer": RateLimitPolicy(
                capacity=settings.RATE_LIMIT_ANSWER_CAPACITY,
                refill_interval_seconds=settings.RATE_LIMIT_ANSWER_REFILL_SECONDS,
                cost=1,
            ),
            "conversation_create": RateLimitPolicy(
                capacity=settings.RATE_LIMIT_CONV_CREATE_CAPACITY,
                refill_interval_seconds=settings.RATE_LIMIT_CONV_CREATE_REFILL_SECONDS,
                cost=1,
            ),
            "conversation_read": RateLimitPolicy(
                capacity=settings.RATE_LIMIT_CONV_READ_CAPACITY,
                refill_interval_seconds=settings.RATE_LIMIT_CONV_READ_REFILL_SECONDS,
                cost=1,
            ),
        }

    @classmethod
    def get_instance(cls) -> "RateLimiterService":
        if cls._instance is None:
            cls._instance = RateLimiterService()
        return cls._instance

    async def check(self, policy_name: str, client_ip: str, cost: Optional[int] = None) -> RateLimitResult:
        if not self.enabled:
            return RateLimitResult(allowed=True, remaining_tokens=100.0, retry_after_seconds=0.0)

        policy = self.policies.get(policy_name)
        if not policy:
            return RateLimitResult(allowed=True, remaining_tokens=100.0, retry_after_seconds=0.0)

        key = f"{policy_name}:{client_ip}"

        if self.backend_type == "redis" and self.redis_bucket:
            try:
                return await self.redis_bucket.consume(key, policy, cost=cost)
            except Exception:
                if not settings.RATE_LIMIT_FAIL_OPEN:
                    return RateLimitResult(allowed=False, remaining_tokens=0.0, retry_after_seconds=5.0)
                # Fall back to in-memory on redis failure
                return await self.in_memory.consume(key, policy, cost=cost)
        else:
            return await self.in_memory.consume(key, policy, cost=cost)


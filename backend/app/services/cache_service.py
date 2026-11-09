"""Tiered Caching Service supporting In-Memory and Redis backends with granular telemetry."""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("nexia.cache_service")


class InMemoryCache:
    """In-memory cache with monotonic TTL expiration and LRU-style eviction."""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        # key -> (value_json: str, expire_at: float, access_time: float)
        self._store: Dict[str, Tuple[str, float, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        async with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            val_json, expire_at, _ = entry
            if now > expire_at:
                # Expired
                self._store.pop(key, None)
                return None
            # Update access time for LRU tracking
            self._store[key] = (val_json, expire_at, now)
            try:
                return json.loads(val_json)
            except Exception:
                return val_json

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        now = time.monotonic()
        expire_at = now + max(0.1, ttl_seconds)
        try:
            val_json = json.dumps(value, default=str)
        except Exception:
            val_json = str(value)

        async with self._lock:
            # Capacity bounding: evict oldest accessed entry if full
            if len(self._store) >= self.max_entries and key not in self._store:
                oldest_key = min(self._store.keys(), key=lambda k: self._store[k][2])
                self._store.pop(oldest_key, None)

            self._store[key] = (val_json, expire_at, now)

    async def delete(self, key: str) -> bool:
        async with self._lock:
            return self._store.pop(key, None) is not None

    async def delete_prefix(self, prefix: str) -> int:
        async with self._lock:
            keys_to_del = [k for k in self._store.keys() if k.startswith(prefix)]
            for k in keys_to_del:
                self._store.pop(k, None)
            return len(keys_to_del)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


class RedisCache:
    """Redis-backed distributed cache with TTL support."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None

    async def _get_client(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        client = await self._get_client()
        val = await client.get(f"cache:{key}")
        if val is None:
            return None
        try:
            return json.loads(val)
        except Exception:
            return val

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        client = await self._get_client()
        try:
            val_json = json.dumps(value, default=str)
        except Exception:
            val_json = str(value)
        await client.set(f"cache:{key}", val_json, ex=max(1, int(ttl_seconds)))

    async def delete(self, key: str) -> bool:
        client = await self._get_client()
        res = await client.delete(f"cache:{key}")
        return res > 0

    async def delete_prefix(self, prefix: str) -> int:
        client = await self._get_client()
        keys = await client.keys(f"cache:{prefix}*")
        if keys:
            return await client.delete(*keys)
        return 0

    async def clear(self) -> None:
        client = await self._get_client()
        await client.flushdb()


class CacheService:
    """Centralized Cache Service with metrics recording and structured keys."""

    _instance: Optional["CacheService"] = None

    def __init__(self):
        self.enabled = settings.CACHE_ENABLED
        self.backend_type = settings.CACHE_BACKEND
        self.in_memory = InMemoryCache(max_entries=2000)
        self.redis_cache = RedisCache(settings.REDIS_URL) if self.backend_type == "redis" else None

        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }

    @classmethod
    def get_instance(cls) -> "CacheService":
        if cls._instance is None:
            cls._instance = CacheService()
        return cls._instance

    @property
    def hit_ratio(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return round(self.stats["hits"] / total, 3) if total > 0 else 0.0

    @staticmethod
    def build_key(namespace: str, *args: Any) -> str:
        """Create a deterministic hash-based cache key from normalized components."""
        serialized = ":".join(str(a).strip().lower() for a in args)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        return f"{namespace}:{digest}"

    async def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None

        try:
            if self.backend_type == "redis" and self.redis_cache:
                val = await self.redis_cache.get(key)
            else:
                val = await self.in_memory.get(key)

            if val is not None:
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return val
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache get error for key '%s': %s", key, e)
            return None

    async def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if not self.enabled:
            return

        ttl = ttl_seconds if ttl_seconds is not None else float(settings.CACHE_DEFAULT_TTL_SECONDS)
        try:
            if self.backend_type == "redis" and self.redis_cache:
                await self.redis_cache.set(key, value, ttl)
            else:
                await self.in_memory.set(key, value, ttl)
            self.stats["sets"] += 1
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache set error for key '%s': %s", key, e)

    async def delete(self, key: str) -> bool:
        if not self.enabled:
            return False

        try:
            self.stats["deletes"] += 1
            if self.backend_type == "redis" and self.redis_cache:
                return await self.redis_cache.delete(key)
            return await self.in_memory.delete(key)
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache delete error for key '%s': %s", key, e)
            return False

    async def invalidate_prefix(self, prefix: str) -> int:
        if not self.enabled:
            return 0
        try:
            if self.backend_type == "redis" and self.redis_cache:
                return await self.redis_cache.delete_prefix(prefix)
            return await self.in_memory.delete_prefix(prefix)
        except Exception as e:
            logger.warning("Cache invalidate_prefix error for '%s': %s", prefix, e)
            return 0


"""
Controlled Local Cache Architecture for Stage 13.
"""
from collections import OrderedDict
import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.optimization.models import CacheEntry

logger = logging.getLogger("nexia.optimization.cache")

_TIME_SENSITIVE_KEYWORDS = {
    "current", "currently", "now", "present", "today", "recent", "recently",
    "leader", "president", "ceo", "prime minister", "status", "latest",
    "holding office", "incumbent",
}


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class LocalCacheManager:
    """
    In-memory controlled cache for Wikipedia evidence, validation fingerprints,
    and service readiness. Enforces bounded capacity and freshness rules.
    """

    def __init__(self, max_entries: int = 1000, default_ttl_seconds: float = 3600.0):
        self.max_entries = max_entries
        self.default_ttl = default_ttl_seconds
        self._wiki_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._validation_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._readiness_cache: Optional[CacheEntry] = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    # ── Wikipedia Caching ────────────────────────────────────────────────────

    def get_wikipedia(
        self,
        query_or_title: str,
        entity: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[Any]:
        """Retrieve cached Wikipedia evidence if fresh and permitted."""
        if force_refresh:
            self._misses += 1
            return None

        key = _norm_key(f"{query_or_title}::{entity or ''}")

        # Check time sensitivity
        words = set(re.findall(r"\b\w+\b", key))
        is_time_sensitive = bool(words & _TIME_SENSITIVE_KEYWORDS)

        entry = self._wiki_cache.get(key)
        if not entry:
            self._misses += 1
            return None

        # Expire time-sensitive data much faster (e.g. 5 minutes or explicit entry ttl) vs static history (1 hour)
        effective_ttl = min(300.0, entry.ttl_seconds) if (is_time_sensitive or entry.is_time_sensitive) else entry.ttl_seconds
        if entry.is_expired(effective_ttl):
            self._wiki_cache.pop(key, None)
            self._misses += 1
            return None

        # Move to end (LRU)
        self._wiki_cache.move_to_end(key)
        self._hits += 1
        logger.debug("[CACHE] Wikipedia hit for key=%r", key)
        return entry.value

    def clear_all(self) -> None:
        """Alias for clear()."""
        self.clear()

    @property
    def cache_hits(self) -> int:
        return self._hits

    @property
    def cache_misses(self) -> int:
        return self._misses

    @property
    def max_size(self) -> int:
        return self.max_entries

    @property
    def _wikipedia_cache(self) -> OrderedDict:
        return self._wiki_cache

    def set_wikipedia(
        self,
        query_or_title: str,
        value: Any = None,
        entity: Optional[str] = None,
        ttl_seconds: Optional[float] = None,
        is_time_sensitive: bool = False,
        is_stable: bool = True,
        **kwargs: Any,
    ) -> None:
        """Store Wikipedia evidence in cache with bounded eviction."""
        # Support keyword arguments 'result' or positional swaps
        if value is None and "result" in kwargs:
            value = kwargs["result"]
        if entity is None and "entity" in kwargs:
            entity = kwargs["entity"]

        # If value is a string (and not a result object) while entity has retrieval attributes, swap them
        if isinstance(value, str) and not hasattr(value, "status") and entity is not None and (hasattr(entity, "status") or hasattr(entity, "page_title")):
            value, entity = entity, value

        key = _norm_key(f"{query_or_title}::{entity or ''}")
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # Evict oldest if capacity exceeded
        if len(self._wiki_cache) >= self.max_entries and key not in self._wiki_cache:
            self._wiki_cache.popitem(last=False)
            self._evictions += 1

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl,
            is_time_sensitive=is_time_sensitive or (not is_stable),
        )
        self._wiki_cache[key] = entry

    # ── Validation Caching ───────────────────────────────────────────────────

    def get_validation(self, answer: str, evidence_fingerprint: str) -> Optional[Any]:
        """Retrieve cached ValidationResult if answer and supporting evidence are unchanged."""
        ans_hash = _hash_content(answer.strip())
        key = f"{ans_hash}::{evidence_fingerprint}"

        entry = self._validation_cache.get(key)
        if not entry or entry.is_expired(self.default_ttl):
            return None

        self._validation_cache.move_to_end(key)
        self._hits += 1
        logger.debug("[CACHE] Validation result reused for key=%r", key)
        return entry.value

    def set_validation(self, answer: str, evidence_fingerprint: str, validation_result: Any) -> None:
        """Cache successful validation result against answer + evidence fingerprint."""
        ans_hash = _hash_content(answer.strip())
        key = f"{ans_hash}::{evidence_fingerprint}"

        if len(self._validation_cache) >= self.max_entries and key not in self._validation_cache:
            self._validation_cache.popitem(last=False)
            self._evictions += 1

        self._validation_cache[key] = CacheEntry(
            key=key,
            value=validation_result,
            created_at=time.time(),
            ttl_seconds=self.default_ttl,
        )

    # ── Model Readiness Caching ──────────────────────────────────────────────

    def get_model_readiness(self) -> Optional[bool]:
        """Retrieve cached model health status (valid for 60 seconds)."""
        if self._readiness_cache and not self._readiness_cache.is_expired(60.0):
            return bool(self._readiness_cache.value)
        return None

    def set_model_readiness(self, is_ready: bool, ttl_seconds: float = 60.0) -> None:
        """Cache model readiness state."""
        self._readiness_cache = CacheEntry(
            key="model_readiness",
            value=is_ready,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
        )

    def clear(self) -> None:
        """Clear all in-memory caches."""
        self._wiki_cache.clear()
        self._validation_cache.clear()
        self._readiness_cache = None
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def stats(self) -> Dict[str, Any]:
        """Observability metrics for local cache."""
        total_lookups = self._hits + self._misses
        hit_rate = (self._hits / total_lookups) if total_lookups > 0 else 0.0
        return {
            "wikipedia_cache_size": len(self._wiki_cache),
            "validation_cache_size": len(self._validation_cache),
            "total_hits": self._hits,
            "total_misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 3),
        }


local_cache_manager = LocalCacheManager()

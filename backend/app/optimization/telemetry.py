"""
Telemetry and Performance Observability for Stage 13.
"""
import logging
import time
from typing import Any, Dict, Optional

from app.optimization.models import ComplexityLevel, ComputationalProfile, PerformanceMetrics

logger = logging.getLogger("nexia.optimization.telemetry")


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce *value* to float safely; return *default* for None / non-numeric values."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class TelemetryTracker:
    """
    In-memory performance telemetry recorder.
    Tracks granular per-request metrics across LLM, Wikipedia, generation,
    validation, memory, and caches.
    """

    def __init__(self) -> None:
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        self._completed_metrics: Dict[str, PerformanceMetrics] = {}

    def start_request(
        self,
        request_id: str,
        complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
    ) -> None:
        """Initialize telemetry recording for a new request."""
        self._active_requests[request_id] = {
            "request_id": request_id,
            "complexity": complexity,
            "start_time": time.monotonic(),
            "llm_call_count": 0,
            "llm_duration_ms": 0.0,
            "wikipedia_request_count": 0,
            "wikipedia_duration_ms": 0.0,
            "navigation_action_count": 0,
            "generation_duration_ms": 0.0,
            "validation_duration_ms": 0.0,
            "memory_duration_ms": 0.0,
            "knowledge_duration_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "concurrency_stats": {},
        }

    def _resolve_req_id(self, req_id: Optional[str]) -> Optional[str]:
        """Return a tracked request ID, falling back to the most-recently started request."""
        if isinstance(req_id, str) and req_id in self._active_requests:
            return req_id
        if self._active_requests:
            return next(reversed(self._active_requests))
        return None

    def _accumulate(
        self,
        req_id: Optional[str],
        field: str,
        duration_ms: Any,
    ) -> None:
        """
        Add *duration_ms* to *field* for the resolved request.

        Callers may pass the value in seconds (< 50) or milliseconds (>= 50).
        Values that are None or non-numeric are silently coerced to 0.0.
        """
        resolved = self._resolve_req_id(req_id)
        if not resolved or resolved not in self._active_requests:
            return
        dur = _safe_float(duration_ms)
        self._active_requests[resolved][field] += dur * 1000.0 if dur < 50 else dur

    # ── Public record_* API ────────────────────────────────────────────────────

    def record_llm_call(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Record one LLM call. *duration_ms* may be seconds (<50) or milliseconds."""
        resolved = self._resolve_req_id(request_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["llm_call_count"] += 1
        self._accumulate(resolved, "llm_duration_ms", duration_ms)

    def record_wikipedia_call(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        cached: bool = False,
        **_kwargs: Any,
    ) -> None:
        """Record a Wikipedia retrieval operation."""
        resolved = self._resolve_req_id(request_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["wikipedia_request_count"] += 1
        self._accumulate(resolved, "wikipedia_duration_ms", duration_ms)
        if cached:
            self._active_requests[resolved]["cache_hits"] += 1
        else:
            self._active_requests[resolved]["cache_misses"] += 1

    def record_navigation(
        self,
        request_id: Optional[str] = None,
        count: int = 1,
        **_kwargs: Any,
    ) -> None:
        """Record navigation actions."""
        resolved = self._resolve_req_id(request_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["navigation_action_count"] += count

    def record_generation(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Record final response generation duration."""
        self._accumulate(self._resolve_req_id(request_id), "generation_duration_ms", duration_ms)

    def record_validation(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Record validation duration."""
        self._accumulate(self._resolve_req_id(request_id), "validation_duration_ms", duration_ms)

    def record_memory(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Record memory operation duration."""
        self._accumulate(self._resolve_req_id(request_id), "memory_duration_ms", duration_ms)

    def record_knowledge(
        self,
        request_id: Optional[str] = None,
        duration_ms: Any = 0.0,
        **_kwargs: Any,
    ) -> None:
        """Record structured knowledge operation duration."""
        self._accumulate(self._resolve_req_id(request_id), "knowledge_duration_ms", duration_ms)

    def record_cache_hit(self, req_id: Optional[str] = None) -> None:
        """Record a cache hit."""
        resolved = self._resolve_req_id(req_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["cache_hits"] += 1

    def record_cache_miss(self, req_id: Optional[str] = None) -> None:
        """Record a cache miss."""
        resolved = self._resolve_req_id(req_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["cache_misses"] += 1

    def record_concurrency(
        self,
        parallel_count: int,
        max_concurrency: int,
        req_id: Optional[str] = None,
    ) -> None:
        """Record parallel action execution statistics."""
        resolved = self._resolve_req_id(req_id)
        if not resolved or resolved not in self._active_requests:
            return
        self._active_requests[resolved]["concurrency_stats"] = {
            "parallel_actions_executed": parallel_count,
            "max_concurrency_limit": max_concurrency,
        }

    def end_request(
        self,
        request_id: str,
        profile: Optional[ComputationalProfile] = None,
    ) -> PerformanceMetrics:
        """Alias for finalize."""
        return self.finalize(request_id, profile)

    def finalize(
        self,
        request_id: str,
        profile: Optional[ComputationalProfile] = None,
    ) -> PerformanceMetrics:
        """Finalize request telemetry and compute total elapsed duration."""
        data = self._active_requests.pop(request_id, None)
        if not data:
            return PerformanceMetrics(request_id=request_id)

        total_ms = (time.monotonic() - data["start_time"]) * 1000.0
        complexity = (
            profile.complexity if profile else data.get("complexity", ComplexityLevel.MEDIUM)
        )

        metrics = PerformanceMetrics(
            request_id=request_id,
            complexity=complexity,
            total_duration_ms=round(total_ms, 2),
            llm_call_count=data["llm_call_count"],
            llm_duration_ms=round(data["llm_duration_ms"], 2),
            wikipedia_request_count=data["wikipedia_request_count"],
            wikipedia_duration_ms=round(data["wikipedia_duration_ms"], 2),
            navigation_action_count=data["navigation_action_count"],
            generation_duration_ms=round(data["generation_duration_ms"], 2),
            validation_duration_ms=round(data["validation_duration_ms"], 2),
            memory_duration_ms=round(data["memory_duration_ms"], 2),
            knowledge_duration_ms=round(data["knowledge_duration_ms"], 2),
            cache_hits=data["cache_hits"],
            cache_misses=data["cache_misses"],
            concurrency_stats=data["concurrency_stats"],
        )

        self._completed_metrics[request_id] = metrics
        # Bound completed history size to avoid unbounded growth.
        if len(self._completed_metrics) > 500:
            oldest_key = next(iter(self._completed_metrics))
            self._completed_metrics.pop(oldest_key, None)

        logger.debug(
            "[TELEMETRY] request=%s complexity=%s total=%.2fms "
            "llm_calls=%d wiki_calls=%d cache_hits=%d",
            request_id[:8],
            metrics.complexity.value,
            metrics.total_duration_ms,
            metrics.llm_call_count,
            metrics.wikipedia_request_count,
            metrics.cache_hits,
        )
        return metrics

    def get_metrics(self, request_id: str) -> Optional[PerformanceMetrics]:
        """Retrieve completed metrics for a request."""
        return self._completed_metrics.get(request_id)


telemetry_tracker = TelemetryTracker()

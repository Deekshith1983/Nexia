"""
Models for Stage 13 Local Intelligence & Performance Optimization.
"""
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class ComplexityLevel(str, Enum):
    """Complexity classification for incoming user requests."""
    LOW = "LOW"        # Greetings, conversational affirmations, smalltalk
    MEDIUM = "MEDIUM"  # Single factual query, definition, targeted lookup
    HIGH = "HIGH"      # Comparisons, multi-entity relationships, multi-hop reasoning


class ComputationalProfile(BaseModel):
    """
    Transient computational budget profile per request.
    Adapts computation to request complexity without calling LLM for classification.
    """
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    llm_budget: int = 2
    research_budget: int = 1
    context_budget_chars: int = 12000
    max_concurrency: int = 2
    enable_reflection: bool = True
    enable_decomposition: bool = False
    enable_navigation: bool = False
    latency_metadata: Dict[str, Any] = Field(default_factory=dict)


class CacheEntry(BaseModel):
    """Structured cache record with TTL and freshness awareness."""
    key: str
    value: Any
    created_at: float = Field(default_factory=time.time)
    ttl_seconds: float = 3600.0  # 1 hour default
    is_time_sensitive: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_expired(self, max_age: Optional[float] = None) -> bool:
        ttl = max_age if max_age is not None else self.ttl_seconds
        return (time.time() - self.created_at) > ttl


class PerformanceMetrics(BaseModel):
    """
    Comprehensive request-level performance telemetry.
    Observes duration, call counts, cache performance, and concurrency without leaking sensitive data.
    """
    request_id: str
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    total_duration_ms: float = 0.0
    llm_call_count: int = 0
    llm_duration_ms: float = 0.0
    wikipedia_request_count: int = 0
    wikipedia_duration_ms: float = 0.0
    navigation_action_count: int = 0
    generation_duration_ms: float = 0.0
    validation_duration_ms: float = 0.0
    memory_duration_ms: float = 0.0
    knowledge_duration_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    concurrency_stats: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

"""
Data models for Stage 14 — Evaluation, Benchmarking & Robustness.
"""
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class EvaluationCategory(str, Enum):
    CORRECTNESS = "CORRECTNESS"
    GROUNDING = "GROUNDING"
    AGENT_BEHAVIOR = "AGENT_BEHAVIOR"
    PERFORMANCE = "PERFORMANCE"
    ROBUSTNESS = "ROBUSTNESS"
    ISOLATION = "ISOLATION"
    SECURITY = "SECURITY"
    REGRESSION = "REGRESSION"


class ScenarioStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class EvaluationScenario(BaseModel):
    """
    Structured definition of an evaluation scenario.
    Describes the input, context, expected system behavior, state, and constraints.
    """
    scenario_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    category: EvaluationCategory
    description: str = ""
    input_query: str
    conversation_id: Optional[str] = None
    user_id: str = "default_eval_user"
    expected_intent: Optional[str] = None
    expected_capability: Optional[str] = None
    expected_status: Optional[str] = "SUCCESS"
    min_sources: int = 0
    required_terms: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    expected_grounding_status: Optional[str] = None
    max_llm_calls: Optional[int] = None
    max_wikipedia_calls: Optional[int] = None
    max_duration_seconds: Optional[float] = None
    adversarial_tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Result of running a single EvaluationScenario."""
    scenario_id: str
    name: str
    category: EvaluationCategory
    status: ScenarioStatus
    duration_ms: float = 0.0
    passed_assertions: List[str] = Field(default_factory=list)
    failed_assertions: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    provenance_trace: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = Field(default_factory=time.time)


class BaselineMetrics(BaseModel):
    """Performance baseline record for regression comparisons."""
    baseline_id: str
    stage: str
    created_at: float = Field(default_factory=time.time)
    average_duration_ms: float = 0.0
    average_llm_calls: float = 0.0
    average_wikipedia_calls: float = 0.0
    cache_hit_rate: float = 0.0
    scenarios_evaluated: int = 0
    metrics_by_category: Dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Comprehensive evaluation report across all evaluation categories."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = Field(default_factory=time.time)
    total_scenarios: int = 0
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    pass_rate: float = 0.0
    category_summary: Dict[str, Dict[str, int]] = Field(default_factory=dict)
    scenario_results: List[ScenarioResult] = Field(default_factory=list)
    baseline_comparison: Optional[Dict[str, Any]] = None
    system_health: Dict[str, Any] = Field(default_factory=dict)

"""Observability Manager: Centralized latency tracking, token usage metrics, and structured diagnostics."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.schemas.answer import RetrievalDiagnostics, TimingBreakdown

logger = logging.getLogger("nexia.observability")


@dataclass
class RequestTrace:
    """Telemetry context for a single answer generation request."""

    request_id: str
    conversation_id: Optional[UUID] = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    timings: TimingBreakdown = field(default_factory=TimingBreakdown)
    diagnostics: RetrievalDiagnostics = field(
        default_factory=lambda: RetrievalDiagnostics(embedding_model="nomic-embed-text", llm_model="gemma3:4b")
    )
    status: str = "IN_PROGRESS"
    events_emitted: int = 0
    first_token_latency_ms: Optional[float] = None

    def mark_first_token(self) -> None:
        if self.first_token_latency_ms is None:
            self.first_token_latency_ms = round((time.perf_counter() - self.start_time) * 1000, 2)

    def finalize(self, status: str = "COMPLETED") -> float:
        self.end_time = time.perf_counter()
        self.status = status
        total_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.timings.total_request_ms = total_ms
        return total_ms


class ObservabilityManager:
    """Centralizes request telemetry, diagnostics aggregation, and performance logging."""

    _instance: Optional["ObservabilityManager"] = None

    def __init__(self):
        self.active_traces: Dict[str, RequestTrace] = {}
        self.completed_traces: List[Dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> "ObservabilityManager":
        if cls._instance is None:
            cls._instance = ObservabilityManager()
        return cls._instance

    def start_trace(self, request_id: str, conversation_id: Optional[UUID] = None) -> RequestTrace:
        trace = RequestTrace(request_id=request_id, conversation_id=conversation_id)
        self.active_traces[request_id] = trace
        return trace

    def get_trace(self, request_id: str) -> Optional[RequestTrace]:
        return self.active_traces.get(request_id)

    def finish_trace(self, request_id: str, status: str = "COMPLETED") -> Optional[RequestTrace]:
        trace = self.active_traces.pop(request_id, None)
        if not trace:
            return None
        total_ms = trace.finalize(status)
        log_payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "conversation_id": str(trace.conversation_id) if trace.conversation_id else None,
            "status": trace.status,
            "total_request_ms": total_ms,
            "llm_request_ms": trace.timings.llm_request_ms,
            "first_token_ms": trace.first_token_latency_ms,
            "prompt_tokens": trace.timings.llm_prompt_tokens,
            "response_tokens": trace.timings.llm_response_tokens,
            "tokens_per_second": trace.timings.llm_tokens_per_second,
            "claims_verified": trace.diagnostics.claims_verified,
            "claims_supported": trace.diagnostics.claims_supported,
            "unsupported_claims_removed": trace.diagnostics.unsupported_claims_removed,
        }
        logger.info("[Observability] Request %s completed in %.2fms | TPS: %s | Claims: %d/%d",
                    request_id, total_ms, trace.timings.llm_tokens_per_second,
                    trace.diagnostics.claims_supported, trace.diagnostics.claims_verified)
        self.completed_traces.append(log_payload)
        if len(self.completed_traces) > 200:
            self.completed_traces.pop(0)
        return trace


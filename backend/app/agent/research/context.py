"""
ResearchContext — the temporary per-request research workspace.

The ResearchContext is the single source of truth for one agent run.
It accumulates evidence, tracks action history, records findings,
and tracks the completion state of the research loop.

ResearchContext is NOT persistent memory. It is created fresh per
request and discarded after the response is delivered. Conversation
state (ConversationManager) continues to track turn history separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.agent.research.action import ResearchAction
from app.agent.research.evidence import CompletionStatus, EvidenceItem
from app.agent.research.finding import Finding
from app.agent.research.plan import ResearchPlan


@dataclass
class ResearchContext:
    """
    Temporary research workspace for a single agent run (one user request).

    Fields:
        request_id           — unique ID for this research run (= agent run_id)
        conversation_id      — identifies the owning conversation session
        original_request     — raw user message as received
        resolved_request     — user message after context/coreference resolution
        objectives           — list of information goals for this run
        plan                 — the structured research plan
        completed_actions    — ordered history of executed actions
        pending_actions      — queue of planned but not yet executed actions
        evidence             — accumulated EvidenceItems from successful retrievals
        findings             — derived findings from accumulated evidence
        sources              — SourceMetadata list for final response attribution
        unresolved_questions — topics that could not be answered from Wikipedia
        completion_status    — current sufficiency assessment
        metadata             — observability: timing, action counts, strategy
    """
    request_id: str
    conversation_id: str
    original_request: str
    resolved_request: str
    objectives: List[str]
    plan: Optional[ResearchPlan]
    completed_actions: List[ResearchAction]
    pending_actions: List[ResearchAction]
    evidence: List[EvidenceItem]
    findings: List[Finding]
    sources: List[Any]              # List[SourceMetadata] — not typed to avoid circular
    unresolved_questions: List[str]
    completion_status: CompletionStatus
    information_needs: List[Any] = field(default_factory=list)
    navigation_history: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Lightweight summary for state metadata storage."""
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id[:8],
            "original_request": self.original_request[:80],
            "resolved_request": self.resolved_request[:80],
            "actions_completed": len(self.completed_actions),
            "actions_pending": len(self.pending_actions),
            "evidence_count": len(self.evidence),
            "findings_count": len(self.findings),
            "information_needs_count": len(self.information_needs),
            "navigation_steps": len(self.navigation_history.steps) if self.navigation_history else 0,
            "completion_status": self.completion_status.value,
            "evidence_pages": [e.page_title for e in self.evidence],
            "plan": self.plan.to_dict() if self.plan else None,
        }


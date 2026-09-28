"""
ResearchPlan — structured plan for a Wikipedia research run.

A plan is created at the start of each request and holds the ordered
set of actions the agent intends to execute. Plans are inspectable,
structured, and can be updated during execution (pending actions list).

Plans are temporary — they exist only for the current agent run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from app.agent.research.action import ResearchAction


class PlanStatus(str, Enum):
    """Lifecycle status of the overall research plan."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass
class ResearchPlan:
    """
    Structured research plan for a single agent run.

    Fields:
        objective            — the user's information need (resolved query)
        actions              — complete ordered list of planned actions
        completion_criteria  — human-readable description of what "done" means
        status               — PENDING → IN_PROGRESS → COMPLETE / FAILED
        metadata             — intent, entity count, multi-entity flag, etc.
    """
    objective: str
    actions: List[ResearchAction]
    completion_criteria: str
    status: PlanStatus = PlanStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "status": self.status.value,
            "completion_criteria": self.completion_criteria,
            "action_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata,
        }

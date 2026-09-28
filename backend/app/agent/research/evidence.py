"""
EvidenceItem and CompletionStatus for the research workspace.

EvidenceItem holds a single retrieved Wikipedia article's content
alongside the action that retrieved it. Evidence is accumulated
across multiple research actions and used for synthesis.

CompletionStatus represents the agent's internal assessment of
whether it has enough information to answer the user's question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class CompletionStatus(str, Enum):
    """
    The agent's current assessment of evidence completeness.

    States:
        SUFFICIENT            — evidence is adequate to generate a grounded answer
        MORE_INFORMATION_NEEDED — more Wikipedia research is required (and budget allows)
        INSUFFICIENT_KNOWLEDGE — Wikipedia does not contain the needed information
        FAILED                — retrieval or processing failure
        CANCELLED             — the request was cancelled by the user
    """
    SUFFICIENT = "SUFFICIENT"
    MORE_INFORMATION_NEEDED = "MORE_INFORMATION_NEEDED"
    INSUFFICIENT_KNOWLEDGE = "INSUFFICIENT_KNOWLEDGE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class EvidenceItem:
    """
    A single retrieved Wikipedia article stored in the research workspace.

    Each EvidenceItem corresponds to one completed WIKIPEDIA_RETRIEVE action.
    Multiple EvidenceItems are accumulated when the agent performs multi-step
    research (e.g., comparison questions requiring two distinct articles).

    Fields:
        action_id     — links to the ResearchAction that produced this item
        page_title    — Wikipedia article title (actual page title, not the query)
        page_url      — canonical Wikipedia desktop URL
        content       — cleaned article extract (markup artifacts already removed)
        source        — always "Wikipedia" for this implementation
        metadata      — retrieval strategy, snippet, etc.
    """
    action_id: str
    page_title: str
    page_url: str
    content: str
    source: str = "Wikipedia"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "page_title": self.page_title,
            "page_url": self.page_url,
            "content_len": len(self.content),
            "source": self.source,
        }

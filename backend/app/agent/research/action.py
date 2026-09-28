"""
ResearchAction — structured representation of a single Wikipedia research action.

An action is the atomic unit of work in the research loop:
    - WIKIPEDIA_SEARCH: full-text Wikipedia search when the exact article is unknown
    - WIKIPEDIA_RETRIEVE: direct retrieval of a specific Wikipedia article by entity/topic

The Agent Runtime plans, executes, and observes actions.
It does NOT expose actions to the user directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ResearchActionType(str, Enum):
    """Type of Wikipedia operation the action performs."""
    WIKIPEDIA_SEARCH = "WIKIPEDIA_SEARCH"
    WIKIPEDIA_RETRIEVE = "WIKIPEDIA_RETRIEVE"
    WIKIPEDIA_RETRIEVE_PAGE = "WIKIPEDIA_RETRIEVE_PAGE"
    WIKIPEDIA_RETRIEVE_SECTION = "WIKIPEDIA_RETRIEVE_SECTION"
    WIKIPEDIA_FOLLOW_LINK = "WIKIPEDIA_FOLLOW_LINK"
    WIKIPEDIA_RESOLVE_REDIRECT = "WIKIPEDIA_RESOLVE_REDIRECT"
    WIKIPEDIA_RESOLVE_DISAMBIGUATION = "WIKIPEDIA_RESOLVE_DISAMBIGUATION"


class ResearchActionStatus(str, Enum):
    """Lifecycle status of a single research action."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"   # duplicate prevention — action was intentionally skipped


@dataclass
class ResearchAction:
    """
    A single planned Wikipedia research or navigation action.

    Fields:
        action_id        — short unique identifier (for observability)
        action_type      — search, retrieve, section, follow link, disambiguation, or redirect
        objective        — human-readable description of what this action tries to find
        query            — the actual search/retrieval query string
        entity           — optional canonical entity name for direct lookup
        section_title    — optional target section title within article
        source_page      — originating article title for link following
        target_link      — target Wikipedia article link to follow
        navigation_depth — current navigation step depth (bounded by max_depth)
        status           — lifecycle state
        result           — raw RetrievalResult once completed (None if pending/failed)
        error            — error message if the action failed
        duration_ms      — wall-clock time for the action (set after execution)
        metadata         — lightweight observability payload
    """
    action_id: str
    action_type: ResearchActionType
    objective: str
    query: str
    entity: Optional[str] = None
    section_title: Optional[str] = None
    source_page: Optional[str] = None
    target_link: Optional[str] = None
    navigation_depth: int = 0
    status: ResearchActionStatus = ResearchActionStatus.PENDING
    result: Optional[Any] = None     # RetrievalResult — not typed to avoid circular import
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Lightweight serialization for state metadata storage."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "objective": self.objective,
            "query": self.query,
            "entity": self.entity,
            "section_title": self.section_title,
            "source_page": self.source_page,
            "target_link": self.target_link,
            "navigation_depth": self.navigation_depth,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


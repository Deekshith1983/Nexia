"""
ExecutionContext and AgentPlan for the Nexia Agent Runtime.

ExecutionContext carries all state needed for a single agent run.
AgentPlan records which capability was selected and why.

Neither is persistent — both are scoped to a single request lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.conversation_manager import ConversationState


@dataclass
class ExecutionContext:
    """
    Execution context for a single agent run (one user request).

    Carries the resolved query, conversation state reference, and
    execution metadata. This is NOT a persistent memory object —
    it is created fresh per request and discarded after the response
    is delivered.

    Fields:
        request_id          — unique ID for this agent run (for observability)
        conversation_id     — persistent conversation session ID
        original_query      — raw user message as received
        resolved_query      — query after coreference resolution
        resolved_entity     — primary named entity (if identified), else None
        conversation_state  — live ConversationState object (mutable reference)
        metadata            — typed sub-payloads: qu, resolution, plan, agent_plan
    """
    request_id: str
    conversation_id: str
    original_query: str
    resolved_query: str
    resolved_entity: Optional[str]
    conversation_state: ConversationState
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPlan:
    """
    Minimal structured plan produced by the Agent Runtime.

    Records which capability was selected for this request and
    the response mode that drove the selection.

    For Stage 3.1, one capability per request is the limit.
    The architecture is designed to allow multi-step plans later
    without changing this data structure.

    Fields:
        response_mode           — from ResponseMode (e.g. "grounded_knowledge")
        selected_capability_id  — registry key of the selected capability
        metadata                — selection rationale (intent, deterministic flag, etc.)
    """
    response_mode: str
    selected_capability_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

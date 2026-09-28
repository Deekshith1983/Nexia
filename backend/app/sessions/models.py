"""
Models for Stage 10 Long-Horizon Task Execution & Research Session Continuity.
"""
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Lifecycle status of a persistent ResearchSession."""
    DRAFT = "DRAFT"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ARCHIVED = "ARCHIVED"


class ResearchSession(BaseModel):
    """
    Persistent state representation of a long-horizon Wikipedia research task.
    Enables pause, save, restore, resume, and completion across requests and application restarts.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = "default_user"
    conversation_id: Optional[str] = None
    title: str
    objective: str
    original_request: str
    resolved_request: str
    information_needs: List[Dict[str, Any]] = Field(default_factory=list)
    research_plan: Optional[Dict[str, Any]] = None
    completed_actions: List[Dict[str, Any]] = Field(default_factory=list)
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list)
    navigation_history: Optional[Dict[str, Any]] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_workspace_state: Optional[Dict[str, Any]] = None
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_items: List[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.IN_PROGRESS
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    version: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

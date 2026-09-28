"""
Memory models for Nexia's Stage 4 persistent long-term memory system.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Supported structured memory categories."""
    PREFERENCE = "preference"
    PROFILE_FACT = "profile_fact"
    PROJECT_CONTEXT = "project_context"
    INTERACTION_PREFERENCE = "interaction_preference"
    LEARNED_CONTEXT = "learned_context"


class MemoryStatus(str, Enum):
    """Lifecycle states for memory records."""
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DELETED = "DELETED"


class Memory(BaseModel):
    """
    Structured long-term memory record.
    
    Persistent memories are intentional, selective items retained across sessions
    and separate from short-term ConversationState and per-task ResearchContext.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="default")
    type: MemoryType = Field(default=MemoryType.LEARNED_CONTEXT)
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE)
    source: str = Field(default="user_explicit")
    metadata: Dict[str, Any] = Field(default_factory=dict)

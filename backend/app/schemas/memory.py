"""Pydantic schemas for Nexia conversational and semantic memory management."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class MemoryType(str, Enum):
    SHORT_TERM = "SHORT_TERM"   # Active conversational subject, recent focus
    EPISODIC = "EPISODIC"       # Key milestone, user decision, historical interaction
    SEMANTIC = "SEMANTIC"       # Reusable distilled fact, user preference


class MemoryCreate(BaseModel):
    """Schema to store a new conversational memory."""

    conversation_id: Optional[UUID] = Field(default=None, description="Source conversation ID")
    memory_type: MemoryType = Field(default=MemoryType.SHORT_TERM)
    subject: str = Field(..., max_length=256, description="Entity or topic name")
    content: str = Field(..., description="Fact, decision, or state representation")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRead(BaseModel):
    """Schema returned when reading memories."""

    id: UUID
    conversation_id: Optional[UUID] = None
    memory_type: MemoryType
    subject: str
    content: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class MemoryUpdate(BaseModel):
    """Schema to update an existing memory."""

    content: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

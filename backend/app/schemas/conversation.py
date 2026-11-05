"""Pydantic schemas for Nexia multi-turn conversation management."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, ConfigDict, Field


class ConversationStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    COMPLETED = "COMPLETED"
    IN_PROGRESS = "IN_PROGRESS"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ChatMessage(BaseModel):
    """Individual message within a conversation."""

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: MessageRole
    content: str
    token_count: Optional[int] = None
    message_status: MessageStatus = MessageStatus.COMPLETED
    source_references: List[Dict[str, Any]] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


class ConversationCreate(BaseModel):
    """Schema to initiate a new conversation session."""

    title: Optional[str] = Field(default=None, max_length=256, description="Optional conversation title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary conversation metadata")


class ConversationUpdate(BaseModel):
    """Schema to update conversation metadata or title."""

    title: Optional[str] = Field(default=None, max_length=256)
    status: Optional[ConversationStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationRead(BaseModel):
    """Summary schema for a conversation."""

    id: UUID
    title: str
    status: ConversationStatus
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationWithHistory(ConversationRead):
    """Full conversation with message history."""

    messages: List[ChatMessage] = Field(default_factory=list)

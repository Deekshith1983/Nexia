"""API routes for managing conversations and message history."""

import logging
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.conversation import (
    ChatMessage,
    ConversationCreate,
    ConversationRead,
    ConversationStatus,
    ConversationUpdate,
    ConversationWithHistory,
)
from app.services.conversation_manager import ConversationManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["Conversations"])
conv_manager = ConversationManager()


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(req: ConversationCreate) -> ConversationRead:
    """Create a new conversational session."""
    return await conv_manager.create_conversation(title=req.title, metadata=req.metadata)


@router.get("", response_model=List[ConversationRead])
async def list_conversations(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[ConversationStatus] = Query(default=None),
) -> List[ConversationRead]:
    """List conversation sessions ordered by most recently updated."""
    return await conv_manager.list_conversations(limit=limit, offset=offset, status=status)


@router.get("/{conversation_id}", response_model=ConversationWithHistory)
async def get_conversation(
    conversation_id: UUID,
    message_limit: int = Query(default=50, ge=1, le=200),
) -> ConversationWithHistory:
    """Get conversation details with message history."""
    conv = await conv_manager.get_conversation_with_history(conversation_id, limit=message_limit)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return conv


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: UUID,
    req: ConversationUpdate,
) -> ConversationRead:
    """Update conversation title, status, or metadata."""
    updated = await conv_manager.update_conversation(
        conversation_id=conversation_id,
        title=req.title,
        status=req.status,
        metadata=req.metadata,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return updated


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID):
    """Delete a conversation and all its messages."""
    deleted = await conv_manager.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return None


@router.get("/{conversation_id}/messages", response_model=List[ChatMessage])
async def get_conversation_messages(
    conversation_id: UUID,
    limit: int = Query(default=50, ge=1, le=200),
) -> List[ChatMessage]:
    """Get chronological messages for a conversation."""
    return await conv_manager.get_messages(conversation_id=conversation_id, limit=limit)


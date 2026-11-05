"""Direct unit tests for Conversation route handlers and schemas."""

import pytest
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from app.api.routes import conversations
from app.schemas.conversation import (
    ChatMessage,
    ConversationCreate,
    ConversationRead,
    ConversationStatus,
    ConversationWithHistory,
    MessageRole,
)


@pytest.mark.asyncio
async def test_create_conversation_handler():
    """Verify create_conversation handler returns correctly formatted ConversationRead."""
    conv_id = uuid4()
    mock_conv = ConversationRead(
        id=conv_id,
        title="Mauryan Empire Inquiry",
        status=ConversationStatus.ACTIVE,
        metadata={"tags": ["history"]},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        message_count=0,
    )

    with patch.object(conversations.conv_manager, "create_conversation", AsyncMock(return_value=mock_conv)):
        req = ConversationCreate(title="Mauryan Empire Inquiry", metadata={"tags": ["history"]})
        res = await conversations.create_conversation(req)
        assert res.id == conv_id
        assert res.title == "Mauryan Empire Inquiry"
        assert res.status == ConversationStatus.ACTIVE


@pytest.mark.asyncio
async def test_get_conversation_handler():
    """Verify get_conversation handler returns conversation and message history."""
    conv_id = uuid4()
    mock_history = ConversationWithHistory(
        id=conv_id,
        title="Gupta Empire Discussion",
        status=ConversationStatus.ACTIVE,
        metadata={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        message_count=2,
        messages=[
            ChatMessage(
                id=uuid4(),
                conversation_id=conv_id,
                role=MessageRole.USER,
                content="Who succeeded Samudragupta?",
                created_at=datetime.now(timezone.utc),
            ),
            ChatMessage(
                id=uuid4(),
                conversation_id=conv_id,
                role=MessageRole.ASSISTANT,
                content="Chandragupta II succeeded Samudragupta in 375 CE.",
                created_at=datetime.now(timezone.utc),
            ),
        ],
    )

    with patch.object(conversations.conv_manager, "get_conversation_with_history", AsyncMock(return_value=mock_history)):
        res = await conversations.get_conversation(conv_id)
        assert res.id == conv_id
        assert len(res.messages) == 2
        assert res.messages[0].role == MessageRole.USER
        assert res.messages[1].role == MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_delete_conversation_handler():
    """Verify delete_conversation handler successfully calls deletion."""
    conv_id = uuid4()
    with patch.object(conversations.conv_manager, "delete_conversation", AsyncMock(return_value=True)):
        res = await conversations.delete_conversation(conv_id)
        assert res is None

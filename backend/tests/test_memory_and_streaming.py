"""Unit tests for MemoryManager, StreamingEngine, and ResponseController."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.schemas.memory import MemoryType
from app.services.memory_manager import MemoryManager
from app.services.response_controller import RequestCancelledError, ResponseController
from app.services.streaming_engine import SSEEvent, StreamingEngine


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.delete = AsyncMock()
    return session


@pytest.fixture
def mock_session_factory(mock_session):
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = mock_session
    factory.return_value.__aexit__.return_value = None
    return factory


@pytest.mark.asyncio
async def test_store_memory(mock_session, mock_session_factory):
    mgr = MemoryManager(session_factory=mock_session_factory)
    conv_id = uuid4()

    mem = await mgr.store_memory(
        subject="Samudragupta",
        content="User asked about Allahabad pillar inscription",
        memory_type=MemoryType.SHORT_TERM,
        conversation_id=conv_id,
        generate_embedding=False,
        session=mock_session,
    )

    assert mem.subject == "Samudragupta"
    assert mem.memory_type == MemoryType.SHORT_TERM
    assert mock_session.add.called
    assert mock_session.commit.called


def test_streaming_engine_sse_format():
    evt = StreamingEngine.emit_start(
        request_id="req-123",
        conversation_id=uuid4(),
        model="gemma3:4b",
    )
    assert evt.startswith("event: start\ndata: {")
    assert '"request_id": "req-123"' in evt
    assert '"model": "gemma3:4b"' in evt

    tok_evt = StreamingEngine.emit_token(request_id="req-123", token="India")
    assert "event: token" in tok_evt
    assert '"token": "India"' in tok_evt

    complete_evt = StreamingEngine.emit_complete(
        request_id="req-123",
        conversation_id=None,
        message_id=None,
        answer_text="India is a country [1].",
        answer_status="ANSWERED",
        source_references=[],
        diagnostics={},
    )
    assert "event: complete" in complete_evt
    assert '"answer_status": "ANSWERED"' in complete_evt


def test_response_controller_concurrency_and_cancellation():
    async def _test():
        ctrl = ResponseController(max_concurrent=1)
        req_id = "test-req-1"

        async def sample_inference():
            return "result"

        res = await ctrl.execute_bounded(req_id, sample_inference)
        assert res == "result"

        # Test cancellation flag
        ctrl.cancel_request("non-existent")
        assert not ctrl.is_cancelled("test-req-2")

    asyncio.run(_test())

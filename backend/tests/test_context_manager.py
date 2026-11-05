"""Unit tests for ContextManager and Dynamic Token Budgeting."""

import pytest
from uuid import uuid4

from app.schemas.conversation import ChatMessage, MessageRole
from app.schemas.memory import MemoryRead, MemoryType
from app.services.context_manager import ContextManager


def test_token_estimation():
    mgr = ContextManager()
    short_text = "Who was Chandragupta Maurya?"
    est = mgr.estimate_tokens(short_text)
    assert 5 <= est <= 15

    empty_est = mgr.estimate_tokens("")
    assert empty_est == 0


def test_assemble_context_priority_allocation():
    mgr = ContextManager(num_ctx=4096, reserved_generation=512)

    passages = [
        {
            "document_title": "Chandragupta Maurya",
            "section_title": "Early life",
            "passage_text": "Chandragupta Maurya was the founder of the Maurya Empire in ancient India.",
        },
        {
            "document_title": "Chandragupta Maurya",
            "section_title": "Conquest",
            "passage_text": "He defeated Seleucus I Nicator and established a vast empire across the subcontinent.",
        },
    ]

    history = [
        ChatMessage(
            conversation_id=uuid4(),
            role=MessageRole.USER,
            content="Who founded the Maurya Empire?",
        ),
        ChatMessage(
            conversation_id=uuid4(),
            role=MessageRole.ASSISTANT,
            content="Chandragupta Maurya founded the Maurya Empire in 322 BCE.",
        ),
    ]

    assembled = mgr.assemble_context(
        question="What was his relationship with Seleucus?",
        system_prompt="You are Nexia.",
        passages=passages,
        conversation_history=history,
    )

    assert assembled.question == "What was his relationship with Seleucus?"
    assert len(assembled.passages) == 2
    assert len(assembled.recent_messages) == 2
    assert assembled.budget_report.total_context_capacity == 4096
    assert assembled.budget_report.reserved_generation_tokens == 512
    assert assembled.budget_report.total_input_tokens > 0
    assert not assembled.budget_report.history_compressed


def test_assemble_context_with_compression_on_overflow():
    mgr = ContextManager(num_ctx=500, reserved_generation=150)

    # 10 history messages
    history = [
        ChatMessage(
            conversation_id=uuid4(),
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Turn message content number {i} discussing ancient Indian dynasties and imperial conquests.",
        )
        for i in range(10)
    ]

    passages = [
        {
            "document_title": "Gupta Empire",
            "section_title": "History",
            "passage_text": "The Gupta Empire was an ancient Indian empire existing from the mid-to-late 3rd century CE.",
        }
    ]

    assembled = mgr.assemble_context(
        question="Tell me about Samudragupta",
        system_prompt="You are Nexia.",
        passages=passages,
        conversation_history=history,
        max_history_turns=4,
    )

    assert assembled.budget_report.history_compressed is True
    assert assembled.conversation_summary is not None
    assert "Earlier topics discussed:" in assembled.conversation_summary


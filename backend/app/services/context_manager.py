"""Context Manager with Dynamic Token Budgeting and Rolling Multi-Turn Context."""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.conversation import ChatMessage, MessageRole
from app.schemas.memory import MemoryRead

logger = logging.getLogger("nexia.context_manager")


class TokenBudgetReport(BaseModel):
    """Detailed breakdown of token capacity allocation."""

    total_context_capacity: int
    reserved_generation_tokens: int
    system_prompt_tokens: int
    safety_margin_tokens: int
    available_input_tokens: int
    question_tokens: int
    evidence_tokens: int
    history_tokens: int
    memory_tokens: int
    total_input_tokens: int
    history_turns_included: int
    history_turns_omitted: int
    history_compressed: bool = False


class AssembledContext(BaseModel):
    """Structured context ready for dynamic prompt assembly."""

    question: str
    passages: List[Dict[str, Any]] = Field(default_factory=list)
    recent_messages: List[ChatMessage] = Field(default_factory=list)
    relevant_memories: List[MemoryRead] = Field(default_factory=list)
    conversation_summary: Optional[str] = None
    budget_report: TokenBudgetReport
    entities: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    missing_aspects: List[str] = Field(default_factory=list)


class ContextManager:
    """Intelligently manages context window budgeting, rolling dialogue windows, and compression."""

    def __init__(
        self,
        num_ctx: Optional[int] = None,
        reserved_generation: Optional[int] = None,
        safety_margin: int = 100,
    ):
        self.num_ctx = num_ctx or settings.LLM_NUM_CTX
        self.reserved_generation = reserved_generation or settings.LLM_NUM_PREDICT
        self.safety_margin = safety_margin

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Conservative token estimation for Gemma 3 (~3.6 chars per token on average)."""
        if not text:
            return 0
        char_tokens = math.ceil(len(text) / 3.6)
        word_tokens = math.ceil(len(text.split()) * 1.35)
        return max(char_tokens, word_tokens)

    def assemble_context(
        self,
        question: str,
        system_prompt: str,
        passages: List[Dict[str, Any]],
        conversation_history: Optional[List[ChatMessage]] = None,
        memories: Optional[List[MemoryRead]] = None,
        entities: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        missing_aspects: Optional[List[str]] = None,
        max_history_turns: int = 6,
    ) -> AssembledContext:
        """Select and budget context elements within the strict model context ceiling."""
        sys_tokens = self.estimate_tokens(system_prompt)
        q_tokens = self.estimate_tokens(question)

        # Usable input budget
        available_budget = (
            self.num_ctx
            - self.reserved_generation
            - sys_tokens
            - self.safety_margin
        )
        if available_budget < 256:
            available_budget = 256

        # 1. Budget Question (Highest Priority)
        current_budget = available_budget - q_tokens

        # 2. Budget Evidence Passages (Second Priority)
        selected_passages: List[Dict[str, Any]] = []
        evidence_tokens = 0
        for p in passages:
            p_text = f"{p.get('document_title', '')} {p.get('section_title', '')} {p.get('passage_text', '')}"
            p_toks = self.estimate_tokens(p_text)
            if evidence_tokens + p_toks <= int(current_budget * 0.70):  # Reserve at least 30% for history & memory
                selected_passages.append(p)
                evidence_tokens += p_toks
            elif not selected_passages:
                # Always retain at least the single best evidence passage even if slightly tight
                selected_passages.append(p)
                evidence_tokens += p_toks
                break

        current_budget -= evidence_tokens

        # 3. Budget Relevant Memories (Fourth Priority)
        selected_memories: List[MemoryRead] = []
        memory_tokens = 0
        if memories:
            for mem in memories:
                m_toks = self.estimate_tokens(f"{mem.subject}: {mem.content}")
                if memory_tokens + m_toks <= min(300, int(current_budget * 0.25)):
                    selected_memories.append(mem)
                    memory_tokens += m_toks

        current_budget -= memory_tokens

        # 4. Budget Conversation History (Third Priority, Rolling Window)
        selected_messages: List[ChatMessage] = []
        history_tokens = 0
        turns_included = 0
        turns_omitted = 0
        compressed_summary = None

        if conversation_history:
            # Sort chronologically, take the most recent messages up to max_history_turns
            recent_candidates = conversation_history[-max_history_turns:]
            older_turns = conversation_history[:-max_history_turns] if len(conversation_history) > max_history_turns else []

            # Work backwards from the most recent turn
            reversed_candidates = list(reversed(recent_candidates))
            included_reversed: List[ChatMessage] = []

            for msg in reversed_candidates:
                m_toks = self.estimate_tokens(msg.content)
                if history_tokens + m_toks <= current_budget:
                    included_reversed.append(msg)
                    history_tokens += m_toks
                    turns_included += 1
                else:
                    turns_omitted += 1

            selected_messages = list(reversed(included_reversed))

            # If older turns exist or turns were omitted, generate a compact summary of previous topics
            if older_turns or turns_omitted > 0:
                omitted_msgs = older_turns + [m for m in recent_candidates if m not in selected_messages]
                key_topics = []
                for om in omitted_msgs:
                    if om.role == MessageRole.USER:
                        # Extract first 60 chars of query as topic hint
                        clean_topic = re.sub(r"[^\w\s]", "", om.content).strip()
                        if clean_topic:
                            key_topics.append(" ".join(clean_topic.split()[:8]))
                if key_topics:
                    compressed_summary = "Earlier topics discussed: " + "; ".join(key_topics[-4:])

        total_input_tokens = sys_tokens + q_tokens + evidence_tokens + history_tokens + memory_tokens

        budget_report = TokenBudgetReport(
            total_context_capacity=self.num_ctx,
            reserved_generation_tokens=self.reserved_generation,
            system_prompt_tokens=sys_tokens,
            safety_margin_tokens=self.safety_margin,
            available_input_tokens=available_budget,
            question_tokens=q_tokens,
            evidence_tokens=evidence_tokens,
            history_tokens=history_tokens,
            memory_tokens=memory_tokens,
            total_input_tokens=total_input_tokens,
            history_turns_included=turns_included,
            history_turns_omitted=turns_omitted + (len(conversation_history or []) - len(selected_messages) - turns_omitted),
            history_compressed=compressed_summary is not None,
        )

        return AssembledContext(
            question=question,
            passages=selected_passages,
            recent_messages=selected_messages,
            relevant_memories=selected_memories,
            conversation_summary=compressed_summary,
            budget_report=budget_report,
            entities=entities or [],
            conflicts=conflicts or [],
            missing_aspects=missing_aspects or [],
        )


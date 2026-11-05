"""Modular LLM Orchestrator: Central coordinator for multi-turn sessions, context, inference, grounding, and streaming."""

import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from app.core.config import settings
from app.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
    RetrievalDiagnostics,
    SourceReference,
    TimingBreakdown,
)
from app.schemas.conversation import ChatMessage, MessageRole, MessageStatus
from app.services.context_manager import ContextManager
from app.services.conversation_manager import ConversationManager
from app.services.llm_adapter import OllamaLLMAdapter
from app.services.memory_manager import MemoryManager
from app.services.observability import ObservabilityManager
from app.services.prompts import PromptContext, build_answer_prompt
from app.services.response_controller import RequestCancelledError, ResponseController
from app.services.streaming_engine import StreamingEngine
from app.services.citation_manager import (
    PrimaryArticleSelector,
    format_answer_with_primary_reference,
)

logger = logging.getLogger("nexia.llm_orchestrator")


class LLMOrchestrator:
    """Coordinates the complete LLM lifecycle across multi-turn history, memory, inference, grounding, and streaming."""

    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        context_manager: Optional[ContextManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        llm_adapter: Optional[OllamaLLMAdapter] = None,
        response_controller: Optional[ResponseController] = None,
        observability_manager: Optional[ObservabilityManager] = None,
    ):
        self.conv_mgr = conversation_manager or ConversationManager()
        self.ctx_mgr = context_manager or ContextManager()
        self.mem_mgr = memory_manager or MemoryManager()
        self.llm_adapter = llm_adapter or OllamaLLMAdapter()
        self.resp_ctrl = response_controller or ResponseController.get_instance()
        self.obs_mgr = observability_manager or ObservabilityManager.get_instance()

    async def execute_turn(
        self,
        question: str,
        evidence_passages: List[Dict[str, Any]],
        conversation_id: Optional[UUID] = None,
        entities: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        missing_aspects: Optional[List[str]] = None,
        sufficiency_status: str = "SUFFICIENT",
        is_list_query: bool = False,
        request_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Optional[UUID]]:
        """Execute a non-streaming grounded answer turn."""
        req_id = request_id or str(uuid4())
        trace = self.obs_mgr.start_trace(req_id, conversation_id=conversation_id)

        # 1. Resolve or establish conversation
        history: List[ChatMessage] = []
        if conversation_id:
            try:
                # Store user message
                await self.conv_mgr.add_message(
                    conversation_id=conversation_id,
                    role=MessageRole.USER,
                    content=question,
                )
                history = await self.conv_mgr.get_messages(conversation_id=conversation_id, limit=8)
            except Exception as e:
                logger.warning("Conversation session retrieval failed: %s", e)

        # 2. Retrieve relevant memories (for context disambiguation only)
        memories = []
        if conversation_id:
            try:
                memories = await self.mem_mgr.retrieve_memories(
                    query=question,
                    conversation_id=conversation_id,
                    limit=3,
                )
            except Exception as e:
                logger.debug("Memory retrieval skipped: %s", e)

        # 3. Assemble dynamic context with token budgeting
        assembled = self.ctx_mgr.assemble_context(
            question=question,
            system_prompt="SYSTEM_PROMPT",
            passages=evidence_passages,
            conversation_history=history[:-1] if history else [],  # Exclude current question turn from history list
            memories=memories,
            entities=entities or [],
            conflicts=conflicts or [],
            missing_aspects=missing_aspects or [],
        )

        # 4. Construct PromptContext
        history_dicts = [{"role": m.role.value, "content": m.content} for m in assembled.recent_messages]
        mem_strings = [f"{m.subject}: {m.content}" for m in assembled.relevant_memories]

        prompt_ctx = PromptContext(
            question=question,
            question_type="LIST" if is_list_query else "FACTOID",
            entities=assembled.entities,
            passages=assembled.passages,
            conflicts=assembled.conflicts,
            sufficiency_rating=sufficiency_status,
            missing_aspects=assembled.missing_aspects,
            conversation_history=history_dicts,
            conversation_summary=assembled.conversation_summary,
            memories=mem_strings,
        )

        num_predict = 512 if is_list_query else settings.LLM_NUM_PREDICT

        # 5. Execute bounded local inference
        async def _run_inference():
            t_llm = time.perf_counter()
            res = await self.llm_adapter.generate_answer(
                context=prompt_ctx,
                num_predict_override=num_predict,
            )
            trace.timings.llm_request_ms = round((time.perf_counter() - t_llm) * 1000, 2)
            trace.timings.llm_prompt_tokens = res.prompt_tokens
            trace.timings.llm_response_tokens = res.response_tokens
            trace.timings.llm_tokens_per_second = res.tokens_per_second
            trace.timings.llm_load_duration_ms = res.load_duration_ms
            trace.timings.llm_eval_duration_ms = res.eval_duration_ms
            return res

        llm_result = await self.resp_ctrl.execute_bounded(
            request_id=req_id,
            coro_func=_run_inference,
            timeout=float(settings.LLM_TIMEOUT_SECONDS),
        )

        # 6. Extract entity memory candidates if applicable
        if conversation_id and entities:
            for ent in entities[:2]:
                asyncio.create_task(
                    self.mem_mgr.store_memory(
                        subject=ent,
                        content=f"Discussed in query: '{question}'",
                        conversation_id=conversation_id,
                        generate_embedding=False,
                    )
                )

        primary_art = None
        if llm_result.answer_text and not llm_result.answer_text.startswith("INSUFFICIENT_EVIDENCE") and not llm_result.answer_text.startswith("LLM answer generation failed"):
            primary_art = PrimaryArticleSelector.select_primary_article(
                selected_chunks=assembled.passages,
                answer_text=llm_result.answer_text,
                question=question,
            )
        clean_formatted_answer = format_answer_with_primary_reference(
            answer_text=llm_result.answer_text,
            primary_article=primary_art,
        )

        return clean_formatted_answer, trace.timings.model_dump(), conversation_id

    async def execute_turn_stream(
        self,
        question: str,
        evidence_passages: List[Dict[str, Any]],
        conversation_id: Optional[UUID] = None,
        entities: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        missing_aspects: Optional[List[str]] = None,
        sufficiency_status: str = "SUFFICIENT",
        is_list_query: bool = False,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a grounded answer turn via Server-Sent Events (SSE)."""
        req_id = request_id or str(uuid4())
        trace = self.obs_mgr.start_trace(req_id, conversation_id=conversation_id)

        # Emit start event
        yield StreamingEngine.emit_start(
            request_id=req_id,
            conversation_id=conversation_id,
            model=settings.LLM_MODEL,
        )

        # 1. Resolve conversation history
        history: List[ChatMessage] = []
        if conversation_id:
            try:
                await self.conv_mgr.add_message(
                    conversation_id=conversation_id,
                    role=MessageRole.USER,
                    content=question,
                )
                history = await self.conv_mgr.get_messages(conversation_id=conversation_id, limit=8)
            except Exception as e:
                logger.warning("Conversation session retrieval failed: %s", e)

        yield StreamingEngine.emit_progress(
            request_id=req_id,
            stage="retrieval",
            message="Evidence passages retrieved and packed",
        )

        # 2. Retrieve memories
        memories = []
        if conversation_id:
            try:
                memories = await self.mem_mgr.retrieve_memories(
                    query=question,
                    conversation_id=conversation_id,
                    limit=3,
                )
            except Exception as e:
                logger.debug("Memory retrieval skipped: %s", e)

        # 3. Assemble dynamic context with token budgeting
        assembled = self.ctx_mgr.assemble_context(
            question=question,
            system_prompt="SYSTEM_PROMPT",
            passages=evidence_passages,
            conversation_history=history[:-1] if history else [],
            memories=memories,
            entities=entities or [],
            conflicts=conflicts or [],
            missing_aspects=missing_aspects or [],
        )

        history_dicts = [{"role": m.role.value, "content": m.content} for m in assembled.recent_messages]
        mem_strings = [f"{m.subject}: {m.content}" for m in assembled.relevant_memories]

        prompt_ctx = PromptContext(
            question=question,
            question_type="LIST" if is_list_query else "FACTOID",
            entities=assembled.entities,
            passages=assembled.passages,
            conflicts=assembled.conflicts,
            sufficiency_rating=sufficiency_status,
            missing_aspects=assembled.missing_aspects,
            conversation_history=history_dicts,
            conversation_summary=assembled.conversation_summary,
            memories=mem_strings,
        )

        num_predict = 512 if is_list_query else settings.LLM_NUM_PREDICT

        yield StreamingEngine.emit_progress(
            request_id=req_id,
            stage="inference",
            message=f"Starting generation with {settings.LLM_MODEL}...",
        )

        # 4. Stream tokens progressively
        raw_tokens: List[str] = []
        final_gen_result = None

        try:
            stream_gen = self.llm_adapter.generate_answer_stream(
                context=prompt_ctx,
                num_predict_override=num_predict,
            )
            async for event_type, chunk in stream_gen:
                if self.resp_ctrl.is_cancelled(req_id):
                    yield StreamingEngine.emit_cancelled(req_id, "Generation cancelled by user")
                    return

                if event_type == "token":
                    trace.mark_first_token()
                    raw_tokens.append(chunk)
                    yield StreamingEngine.emit_token(req_id, chunk)
                elif event_type == "done":
                    final_gen_result = chunk

        except RequestCancelledError:
            yield StreamingEngine.emit_cancelled(req_id, "Request cancelled")
            return
        except Exception as e:
            logger.error("Streaming generation failure: %s", e)
            yield StreamingEngine.emit_error(req_id, "LLM_STREAM_ERROR", str(e))
            return

        full_raw_text = "".join(raw_tokens).strip()

        # Select primary article and format final clean answer (no inline [N] citations, single trailing reference)
        primary_art = None
        if full_raw_text and not full_raw_text.startswith("INSUFFICIENT_EVIDENCE") and not full_raw_text.startswith("LLM answer generation failed"):
            primary_art = PrimaryArticleSelector.select_primary_article(
                selected_chunks=assembled.passages,
                answer_text=full_raw_text,
                question=question,
            )
        formatted_answer = format_answer_with_primary_reference(
            answer_text=full_raw_text,
            primary_article=primary_art,
        )

        saved_msg_id = None
        if conversation_id:
            try:
                ast_msg = await self.conv_mgr.add_message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=formatted_answer,
                    token_count=len(raw_tokens),
                    message_status=MessageStatus.COMPLETED if formatted_answer else MessageStatus.FAILED,
                    source_references=evidence_passages,
                )
                saved_msg_id = ast_msg.id
            except Exception as e:
                logger.warning("Failed to persist assistant streaming message: %s", e)

        # 5. Emit completed event
        yield StreamingEngine.emit_complete(
            request_id=req_id,
            conversation_id=conversation_id,
            message_id=saved_msg_id,
            answer_text=formatted_answer,
            answer_status="ANSWERED" if formatted_answer else "FAILED",
            source_references=evidence_passages,
            diagnostics={
                "tokens_generated": len(raw_tokens),
                "first_token_latency_ms": trace.first_token_latency_ms,
            },
            timings=trace.timings.model_dump(),
        )
        self.obs_mgr.finish_trace(req_id, status="COMPLETED")


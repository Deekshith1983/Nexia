"""
ConversationCapability — handles general conversational responses.

Uses the existing conversational generation path (CONVERSATIONAL_SYSTEM_PROMPT)
through the Language Model Interface. Does NOT invoke Wikipedia retrieval.
"""
import logging
import time
from typing import List, Optional

from app.agent.capability import Capability, CapabilityRequest, CapabilityResult, CapabilityStatus
from app.core.grounded_generator import grounded_generator, GenerationInput
from app.core.conversation_manager import conversation_manager
from app.core.source_attribution import SourceMetadata
from app.services.ollama_service import ollama_service

logger = logging.getLogger("nexia.agent.conversation")

CAPABILITY_ID = "conversation"


class ConversationCapability(Capability):
    """
    Handles casual, general, and non-factual conversational responses.

    This capability does not trigger Wikipedia retrieval, answer validation,
    or source attribution — it is a direct conversational path through the LLM.

    The CapabilityRequest.input is expected to contain:
        plan      — ResponsePlan from DecisionPolicy
        qu        — QueryUnderstanding result
        resolution — ContextResolution result
    """

    @property
    def id(self) -> str:
        return CAPABILITY_ID

    @property
    def description(self) -> str:
        return (
            "Handles general conversational responses. "
            "Does not retrieve external knowledge."
        )

    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        ctx = request.context
        state = ctx.conversation_state
        cid = ctx.conversation_id
        t_start = time.monotonic()

        relevant_memories = ctx.metadata.get("relevant_memories", [])
        memory_contents = [m.content if hasattr(m, "content") else str(m) for m in relevant_memories]

        effective_profile = request.input.get("effective_profile")
        pers_guidance = effective_profile.to_guidance_text() if effective_profile else None

        gen_input = GenerationInput(
            original_query=ctx.original_query,
            resolved_query=ctx.resolved_query,
            conversation_context=state.messages,
            retrieved_evidence=None,
            relevant_memories=memory_contents if memory_contents else None,
            personalization_guidance=pers_guidance,
            is_knowledge_query=False,
        )

        # Record user message before streaming starts
        conversation_manager.add_message(cid, "user", ctx.original_query)

        sources: List[SourceMetadata] = []

        async def stream_gen():
            accumulated = ""
            gen_start = time.monotonic()
            try:
                prompt = grounded_generator.build_prompt(gen_input)
                async for token in ollama_service.stream_response(prompt):
                    accumulated += token
                    yield token
            finally:
                gen_sec = time.monotonic() - gen_start
                gen_ms = int(gen_sec * 1000)
                total_ms = int((time.monotonic() - t_start) * 1000)
                from app.optimization.telemetry import telemetry_tracker
                telemetry_tracker.record_generation(gen_sec)
                if accumulated.strip():
                    conversation_manager.add_message(cid, "assistant", accumulated)
                logger.debug(
                    "[CONV CAP] cid=%s gen_ms=%d total_ms=%d chars=%d",
                    cid[:8], gen_ms, total_ms, len(accumulated),
                )

        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            stream=stream_gen(),
            sources=sources,
            metadata={
                "capability": CAPABILITY_ID,
                "conversation_id": cid,
            },
        )

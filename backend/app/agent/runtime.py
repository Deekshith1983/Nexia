"""
NexiaAgentRuntime — the central orchestrator for the Nexia Agent.

Responsibilities:
    1. Establish execution context (conversation state, request ID)
    2. Run Query Understanding via query_analyzer
    3. Run Context & Coreference Resolution via context_manager
    4. Run Decision Policy via decision_policy
    5. Map ResponsePlan → capability_id → AgentPlan
    6. Look up capability from CapabilityRegistry
    7. Build ExecutionContext + CapabilityRequest
    8. Execute capability and return structured CapabilityResult
    9. Handle errors, cancellation, and missing capabilities safely

The runtime does NOT implement pipeline stages directly.
It coordinates Stage 2 subsystems (steps 2-4) and delegates to
capabilities (steps 5-8) for the remaining execution.

Architecture:
    FastAPI → NexiaAgentRuntime → CapabilityRegistry → Capability
                                                      → Stage 2 subsystems
"""
import asyncio
import logging
import time
import uuid
from typing import Optional, Tuple

from app.agent.capability import CapabilityRequest, CapabilityResult, CapabilityStatus
from app.agent.execution_context import AgentPlan, ExecutionContext
from app.agent.registry import capability_registry
from app.core.conversation_manager import conversation_manager
from app.core.context_manager import context_manager
from app.core.query_understanding import query_analyzer
from app.core.decision_policy import decision_policy, ResponseMode
from app.services.ollama_service import ollama_service

logger = logging.getLogger("nexia.agent.runtime")

from app.memory.commands import memory_command_parser
from app.memory.manager import memory_manager
from app.sessions.commands import session_command_parser, SessionCommandType
from app.sessions.manager import research_session_manager

# ── Capability selection mapping ──────────────────────────────────────────────
# Maps ResponseMode values to registered capability IDs.
# All unknown modes fall back to "conversation" (safe default).
_CAPABILITY_MAP: dict = {
    ResponseMode.GROUNDED_KNOWLEDGE: "wikipedia_knowledge",
    ResponseMode.UNSUPPORTED_KNOWLEDGE: "wikipedia_knowledge",
    ResponseMode.CONVERSATIONAL: "conversation",
}


def _mode_to_capability(response_mode: str) -> str:
    """Map a ResponseMode string to a capability ID. Defaults to conversation."""
    return _CAPABILITY_MAP.get(response_mode, "conversation")


class NexiaAgentRuntime:
    """
    Nexia Agent Runtime.

    Entry point for all user requests in Stage 3+.
    Replaces the direct `nexia_core.process_chat_stream()` call in chat.py.

    NexiaCore continues to exist and its subsystems are shared;
    the runtime uses the same module-level singletons.

    Stage 4 enhancements:
        - Explicit memory command routing (REMEMBER, FORGET, RECALL)
        - Memory relevance detection and contextual injection for conversational requests
        - Strict isolation between user memory and external Wikipedia knowledge
        - Multi-user isolation support via user_id

    Stage 10 enhancements:
        - Explicit research session command routing (LIST, PAUSE, CANCEL, ARCHIVE, DELETE, STATUS, SAVE)
        - Research session continuity & resume handling with ambiguity protection
        - Strict isolation between Conversation State, User Memory, and Research Sessions
    """

    async def run(
        self,
        user_message: str,
        conversation_id: Optional[str] = None,
        user_id: str = "default",
    ) -> Tuple[CapabilityResult, str]:
        """
        Execute a single agent run for one user message.

        Returns:
            (CapabilityResult, conversation_id)

            CapabilityResult.stream — async generator for streaming responses.
            CapabilityResult.sources — source metadata for response header.
            CapabilityResult.status — SUCCESS / FAILED / UNAVAILABLE / CANCELLED.
        """
        run_id = str(uuid.uuid4())
        t_start = time.monotonic()

        # ── 1. Conversation state ──────────────────────────────────────────────
        state = conversation_manager.get_or_create_conversation(conversation_id)
        cid = state.conversation_id

        logger.debug(
            "[RUNTIME] run_id=%s cid=%s user_id=%s query=%r",
            run_id[:8], cid[:8], user_id, user_message[:100],
        )

        try:
            # ── 1.4. Explicit Session Command Check ─────────────────────────────
            session_cmd = session_command_parser.parse(user_message)
            if session_cmd:
                logger.info(
                    "[RUNTIME] run_id=%s cid=%s Detected session command: %s",
                    run_id[:8], cid[:8], session_cmd.command_type.value
                )
                if session_cmd.command_type == SessionCommandType.RESUME:
                    matched, candidates, match_type = research_session_manager.find_matching_session(
                        session_cmd.target_identifier or user_message, user_id=user_id
                    )
                    if match_type == "AMBIGUOUS":
                        cand_str = ", ".join(f"'{c.title}' (ID: `{c.session_id}`)" for c in candidates)
                        response_text = f"Multiple active research sessions found: {cand_str}. Which one would you like to continue?"
                        async def _ambiguous_gen():
                            conversation_manager.add_message(cid, "assistant", response_text)
                            yield response_text
                        conversation_manager.add_message(cid, "user", user_message)
                        return CapabilityResult(
                            status=CapabilityStatus.SUCCESS,
                            stream=_ambiguous_gen(),
                            sources=[],
                            metadata={"ambiguous_sessions": [c.session_id for c in candidates]},
                        ), cid
                    elif match_type in ("NO_MATCH", "NO_ACTIVE_SESSIONS"):
                        response_text = "I couldn't find an active research session matching your request."
                        async def _no_match_gen():
                            conversation_manager.add_message(cid, "assistant", response_text)
                            yield response_text
                        conversation_manager.add_message(cid, "user", user_message)
                        return CapabilityResult(
                            status=CapabilityStatus.SUCCESS,
                            stream=_no_match_gen(),
                            sources=[],
                            metadata={},
                        ), cid
                    else:
                        # Unambiguous matched session to resume
                        matched = research_session_manager.resume_session(matched.session_id)
                        qu = await query_analyzer.analyze(matched.original_request, ollama_service)
                        resolution = await context_manager.resolve(qu, state, ollama_service)
                        plan = decision_policy.create_plan(qu, resolution)
                        cap = capability_registry.get("wikipedia_knowledge")
                        if cap is None:
                            return _make_unavailable("Wikipedia capability is not registered.", run_id), cid

                        exec_ctx = ExecutionContext(
                            request_id=run_id,
                            conversation_id=cid,
                            original_query=user_message,
                            resolved_query=matched.resolved_request,
                            resolved_entity=resolution.resolved_entity,
                            conversation_state=state,
                            metadata={"user_id": user_id, "session_id": matched.session_id},
                        )
                        cap_req = CapabilityRequest(
                            capability_id="wikipedia_knowledge",
                            input={
                                "qu": qu,
                                "resolution": resolution,
                                "plan": plan,
                                "resumed_session": matched,
                            },
                            context=exec_ctx,
                            metadata={"run_id": run_id, "user_id": user_id},
                        )
                        result = await cap.execute(cap_req)
                        return result, cid

                else:
                    # Non-resume session operations (LIST, PAUSE, CANCEL, ARCHIVE, DELETE, STATUS, SAVE)
                    cap = capability_registry.get("session_operation")
                    if cap is None:
                        return _make_unavailable("Session capability is not registered.", run_id), cid

                    sess_exec_ctx = ExecutionContext(
                        request_id=run_id,
                        conversation_id=cid,
                        original_query=user_message,
                        resolved_query=user_message,
                        resolved_entity=None,
                        conversation_state=state,
                        metadata={"user_id": user_id, "session_cmd": session_cmd.model_dump()},
                    )
                    sess_req = CapabilityRequest(
                        capability_id="session_operation",
                        input={"session_cmd": session_cmd},
                        context=sess_exec_ctx,
                        metadata={"run_id": run_id, "user_id": user_id},
                    )
                    result = await cap.execute(sess_req)
                    return result, cid

            # ── 1.5. Explicit Memory Command Check ──────────────────────────────
            mem_cmd = memory_command_parser.parse(user_message)
            if mem_cmd:
                logger.info(
                    "[RUNTIME] run_id=%s cid=%s Detected memory command: %s",
                    run_id[:8], cid[:8], mem_cmd.command_type.value
                )
                cap = capability_registry.get("memory_operation")
                if cap is None:
                    return _make_unavailable("Memory capability is not registered.", run_id), cid

                mem_exec_ctx = ExecutionContext(
                    request_id=run_id,
                    conversation_id=cid,
                    original_query=user_message,
                    resolved_query=user_message,
                    resolved_entity=None,
                    conversation_state=state,
                    metadata={"user_id": user_id, "mem_cmd": mem_cmd.model_dump()},
                )
                mem_req = CapabilityRequest(
                    capability_id="memory_operation",
                    input={"mem_cmd": mem_cmd},
                    context=mem_exec_ctx,
                    metadata={"run_id": run_id, "user_id": user_id},
                )
                result = await cap.execute(mem_req)
                return result, cid

            # ── 2. Query Understanding ─────────────────────────────────────────
            t = time.monotonic()
            qu = await query_analyzer.analyze(user_message, ollama_service)
            qu_ms = int((time.monotonic() - t) * 1000)
            state.metadata["latest_query_understanding"] = qu.model_dump()

            # ── 3. Context & Coreference Resolution ───────────────────────────
            t = time.monotonic()
            resolution = await context_manager.resolve(qu, state, ollama_service)
            ctx_ms = int((time.monotonic() - t) * 1000)
            state.metadata["latest_context_resolution"] = resolution.model_dump()

            # ── 4. Decision Policy ────────────────────────────────────────────
            t = time.monotonic()
            plan = decision_policy.create_plan(qu, resolution)
            dec_ms = int((time.monotonic() - t) * 1000)
            state.metadata["latest_response_plan"] = plan.model_dump()

            # ── 4.5. Memory Relevance Search ───────────────────────────────────
            relevant_memories = memory_manager.get_relevant_memories(
                resolution.resolved_query, user_id=user_id, limit=3
            )

            # ── 5. Agent Plan (capability selection) ──────────────────────────
            capability_id = _mode_to_capability(plan.response_mode)
            agent_plan = AgentPlan(
                response_mode=plan.response_mode,
                selected_capability_id=capability_id,
                metadata={
                    "intent": qu.intent,
                    "deterministic": qu.deterministic,
                    "entity": qu.entity,
                    "relevant_memory_count": len(relevant_memories),
                },
            )
            state.metadata["latest_agent_plan"] = {
                "response_mode": agent_plan.response_mode,
                "selected_capability_id": agent_plan.selected_capability_id,
                "metadata": agent_plan.metadata,
            }

            logger.debug(
                "[RUNTIME] run_id=%s qu_ms=%d ctx_ms=%d dec_ms=%d "
                "mode=%s → capability=%s (memories=%d)",
                run_id[:8], qu_ms, ctx_ms, dec_ms,
                plan.response_mode, capability_id, len(relevant_memories),
            )

            # ── 6. Capability lookup ───────────────────────────────────────────
            capability = capability_registry.get(capability_id)
            if capability is None:
                logger.error(
                    "[RUNTIME] run_id=%s capability not found: %r  registered=%s",
                    run_id[:8], capability_id, capability_registry.list(),
                )
                return _make_unavailable(
                    f"Capability '{capability_id}' is not registered in Nexia.", run_id
                ), cid

            # ── 7. Execution Context ───────────────────────────────────────────
            exec_ctx = ExecutionContext(
                request_id=run_id,
                conversation_id=cid,
                original_query=user_message,
                resolved_query=resolution.resolved_query,
                resolved_entity=resolution.resolved_entity,
                conversation_state=state,
                metadata={
                    "user_id": user_id,
                    "qu": qu.model_dump(),
                    "resolution": resolution.model_dump(),
                    "plan": plan.model_dump(),
                    "relevant_memories": relevant_memories,
                    "agent_plan": {
                        "response_mode": agent_plan.response_mode,
                        "selected_capability_id": agent_plan.selected_capability_id,
                    },
                },
            )

            # ── 8. Capability Request ──────────────────────────────────────────
            cap_request = CapabilityRequest(
                capability_id=capability_id,
                input={
                    "qu": qu,
                    "resolution": resolution,
                    "plan": plan,
                    "agent_plan": agent_plan,
                    "relevant_memories": relevant_memories,
                },
                context=exec_ctx,
                metadata={"run_id": run_id, "user_id": user_id},
            )

            # ── 9. Execute capability ──────────────────────────────────────────
            result = await capability.execute(cap_request)

            overhead_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(
                "[RUNTIME] run_id=%s cid=%s capability=%s status=%s overhead_ms=%d",
                run_id[:8], cid[:8], capability_id,
                result.status.value, overhead_ms,
            )
            return result, cid

        except asyncio.CancelledError:
            logger.info("[RUNTIME] run_id=%s cid=%s CANCELLED", run_id[:8], cid[:8])
            return _make_cancelled(run_id), cid

        except Exception as exc:
            logger.exception(
                "[RUNTIME] run_id=%s UNEXPECTED ERROR: %s", run_id[:8], exc
            )
            return _make_failed(str(exc), run_id), cid


# ── Helper constructors for error CapabilityResults ───────────────────────────

def _make_unavailable(reason: str, run_id: str) -> CapabilityResult:
    """Capability not found in registry."""
    async def _gen():
        yield f"[Nexia] This capability is not available. {reason}"
    return CapabilityResult(
        status=CapabilityStatus.UNAVAILABLE,
        stream=_gen(),
        error=reason,
        metadata={"run_id": run_id},
    )


def _make_cancelled(run_id: str) -> CapabilityResult:
    """Request was cancelled (client disconnect / Stop Generation)."""
    async def _gen():
        # Yields nothing — the stream was stopped
        return
        yield  # pragma: no cover — makes this an async generator
    return CapabilityResult(
        status=CapabilityStatus.CANCELLED,
        stream=_gen(),
        error="Request was cancelled.",
        metadata={"run_id": run_id},
    )


def _make_failed(error: str, run_id: str) -> CapabilityResult:
    """Unexpected runtime exception."""
    async def _gen():
        yield f"[Nexia] An error occurred: {error}"
    return CapabilityResult(
        status=CapabilityStatus.FAILED,
        stream=_gen(),
        error=error,
        metadata={"run_id": run_id},
    )


# Global singleton — imported by chat.py and tests
agent_runtime = NexiaAgentRuntime()

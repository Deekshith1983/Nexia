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
from app.personalization.manager import personalization_manager
from app.optimization import telemetry_tracker, profile_classifier, ComplexityLevel

# ── Capability selection mapping ──────────────────────────────────────────────
# Maps ResponseMode values to registered capability IDs.
# All unknown modes fall back to "conversation" (safe default).
_CAPABILITY_MAP: dict = {
    ResponseMode.GROUNDED_KNOWLEDGE: "wikipedia_knowledge",
    ResponseMode.UNSUPPORTED_KNOWLEDGE: "wikipedia_knowledge",
    ResponseMode.CONVERSATIONAL: "conversation",
}


# ── Phase 4: Personal / Memory-recall routing guard ────────────────────────

import re as _re

_PERSONAL_RECALL_RE = _re.compile(
    r"^\s*(?:"
    # "do you remember my [name/...]" / "remember me" / "remember who I am"
    r"(?:do\s+you\s+)?(?:remember|recall)\s+(?:my\b.*|me\b|who\s+i\s+am|what\s+i(?:'ve|\s+have)?\s+told.*)|"
    # "what is my name" / "what's my [X]"
    r"what(?:'s|\s+is)\s+my\b.*|"
    # "who am I"
    r"who\s+am\s+i|"
    # "what did I tell you"
    r"what\s+did\s+i\s+tell\s+you.*|"
    # "what do you know about me"
    r"what\s+do\s+you\s+know\s+about\s+me.*|"
    # "tell me what you know about me" / "tell me something about me"
    r"tell\s+me\s+(?:what\s+you\s+know|something)\s+about\s+me.*|"
    # "what have I shared with you"
    r"what\s+have\s+i\s+shared\s+with\s+you.*|"
    # "have you saved/stored my [info]" / "have you saved anything"
    r"have\s+you\s+(?:saved|stored|remembered|kept)\s+(?:my\b.*|anything.*)|"
    # "do you have / know my [X]" / "do you know any info about me"
    r"do\s+you\s+(?:have|know)\s+(?:my\b.*|any\s+info(?:rmation)?\s+about\s+me.*)"
    r")[?!.\s]*$",
    _re.IGNORECASE,
)



def _is_personal_recall_query(text: str) -> bool:
    """Return True if query is clearly asking Nexia to recall personal/user information."""
    return bool(_PERSONAL_RECALL_RE.match(text))


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
        telemetry_tracker.start_request(run_id)

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
                    from app.sessions.manager import research_session_manager
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

            # ── 2. Query Understanding (Route-First Deterministic Fast Path) ───
            t = time.monotonic()
            qu = query_analyzer.understand(user_message)
            qu_ms = int((time.monotonic() - t) * 1000)
            state.metadata["latest_query_understanding"] = qu.model_dump()

            # ── 2.1. Out-of-Scope Polite Decline ──────────────────────────────
            if qu.route == "out_of_scope":
                decline_msg = (
                    "I am Nexia, an informational assistant focused on Wikipedia knowledge. "
                    "I cannot perform mathematical calculations, real-time weather forecasts, or other live external operations."
                )
                conversation_manager.add_message(cid, "user", user_message)
                async def _out_of_scope_gen():
                    conversation_manager.add_message(cid, "assistant", decline_msg)
                    yield decline_msg
                telemetry_tracker.end_request(run_id)
                return CapabilityResult(
                    status=CapabilityStatus.SUCCESS,
                    stream=_out_of_scope_gen(),
                    sources=[],
                    metadata={"routing": "out_of_scope"},
                ), cid

            # ── 2.2. User Statement (Self-introduction / facts) ───────────────
            if qu.route == "user_statement":
                m_name = _re.search(r"\b(?:i\s+am|i'm|my\s+name\s+is)\s+([A-Za-z0-9]+)\b", user_message, _re.I)
                name_val = m_name.group(1).strip() if m_name and m_name.group(1).lower() not in ("a", "an", "the", "here") else None
                if name_val:
                    memory_manager.add_memory(f"User's name is {name_val.capitalize()}", user_id=user_id)
                    reply = f"Nice to meet you, {name_val.capitalize()}! I've remembered your name. How can I assist you with Wikipedia knowledge today?"
                else:
                    memory_manager.add_memory(user_message, user_id=user_id)
                    reply = "Thank you for sharing! I've saved that to memory. How can I help you today?"
                conversation_manager.add_message(cid, "user", user_message)
                async def _user_stmt_gen():
                    conversation_manager.add_message(cid, "assistant", reply)
                    yield reply
                telemetry_tracker.end_request(run_id)
                return CapabilityResult(
                    status=CapabilityStatus.SUCCESS,
                    stream=_user_stmt_gen(),
                    sources=[],
                    metadata={"routing": "user_statement"},
                ), cid

            # ── 2.3. Personal Recall / Memory Query ───────────────────────────
            if qu.route == "memory_recall" or _is_personal_recall_query(user_message):
                logger.info(
                    "[RUNTIME] run_id=%s cid=%s Personal recall query \u2014 routing to memory",
                    run_id[:8], cid[:8],
                )
                relevant = memory_manager.get_relevant_memories(
                    user_message, user_id=user_id, limit=5
                )
                if relevant:
                    mem_lines = "\n".join(f"- {m}" for m in relevant)
                    reply = f"Here's what I remember about you:\n{mem_lines}"
                else:
                    reply = (
                        "I don't have any personal information stored about you yet. "
                        "You can tell me things like 'My name is ...' and I'll remember them."
                    )
                conversation_manager.add_message(cid, "user", user_message)
                async def _recall_gen():
                    conversation_manager.add_message(cid, "assistant", reply)
                    yield reply
                telemetry_tracker.end_request(run_id)
                return CapabilityResult(
                    status=CapabilityStatus.SUCCESS,
                    stream=_recall_gen(),
                    sources=[],
                    metadata={"routing": "memory_recall"},
                ), cid

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

            # ── 4.5. Memory Relevance Search & Personalization Profile ─────────
            # Check for permanent preference change statements like "From now on, give me detailed explanations"
            if "from now on" in user_message.lower():
                extracted = personalization_manager.parse_preference_from_query(user_message)
                for cat, key, val in extracted:
                    personalization_manager.store_preference(category=cat, key=key, value=val, user_id=user_id)

            relevant_memories = memory_manager.get_relevant_memories(
                resolution.resolved_query, user_id=user_id, limit=3
            )
            effective_profile = personalization_manager.resolve_effective_profile(
                user_message, user_id=user_id, conversation_state=state
            )
            state.metadata["latest_effective_profile"] = effective_profile.model_dump()

            # ── 4.8. Computational Profile (Stage 13) ─────────────────────────
            computational_profile = profile_classifier.classify(
                qu=qu,
                resolution=resolution,
                plan=plan,
                relevant_memories=relevant_memories,
            )
            state.metadata["latest_computational_profile"] = computational_profile.model_dump()

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
                    "effective_profile": effective_profile.model_dump(),
                    "computational_profile": computational_profile.model_dump(),
                },
            )
            state.metadata["latest_agent_plan"] = {
                "response_mode": agent_plan.response_mode,
                "selected_capability_id": agent_plan.selected_capability_id,
                "metadata": agent_plan.metadata,
            }

            logger.debug(
                "[RUNTIME] run_id=%s qu_ms=%d ctx_ms=%d dec_ms=%d "
                "mode=%s → capability=%s (complexity=%s, memories=%d, profile=%s)",
                run_id[:8], qu_ms, ctx_ms, dec_ms,
                plan.response_mode, capability_id, computational_profile.complexity.value,
                len(relevant_memories),
                effective_profile.applied_preferences,
            )

            # ── 6. Capability lookup ───────────────────────────────────────────
            capability = capability_registry.get(capability_id)
            if capability is None:
                logger.error(
                    "[RUNTIME] run_id=%s capability not found: %r  registered=%s",
                    run_id[:8], capability_id, capability_registry.list(),
                )
                telemetry_tracker.end_request(run_id)
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
                    "effective_profile": effective_profile.model_dump(),
                    "computational_profile": computational_profile.model_dump(),
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
                    "effective_profile": effective_profile,
                    "computational_profile": computational_profile,
                },
                context=exec_ctx,
                metadata={"run_id": run_id, "user_id": user_id},
            )

            # ── 9. Execute capability ──────────────────────────────────────────
            result = await capability.execute(cap_request)

            overhead_ms = int((time.monotonic() - t_start) * 1000)
            perf_metrics = telemetry_tracker.end_request(run_id)
            if result.metadata is not None:
                result.metadata["performance_metrics"] = perf_metrics.model_dump()
                result.metadata["computational_profile"] = computational_profile.model_dump()

            logger.info(
                "[RUNTIME] run_id=%s cid=%s capability=%s status=%s overhead_ms=%d llm_calls=%d wiki_calls=%d",
                run_id[:8], cid[:8], capability_id,
                result.status.value, overhead_ms,
                perf_metrics.llm_call_count, perf_metrics.wikipedia_request_count,
            )
            return result, cid

        except asyncio.CancelledError:
            logger.info("[RUNTIME] run_id=%s cid=%s CANCELLED", run_id[:8], cid[:8])
            telemetry_tracker.end_request(run_id)
            return _make_cancelled(run_id), cid

        except Exception as exc:
            logger.exception(
                "[RUNTIME] run_id=%s UNEXPECTED ERROR: %s", run_id[:8], exc
            )
            telemetry_tracker.end_request(run_id)
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

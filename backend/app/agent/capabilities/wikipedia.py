"""
WikipediaKnowledgeCapability — Stage 3 agentic Wikipedia research and generation.

Replaces the Stage 3.1 single-retrieval implementation with the full
agentic research loop:

    WikipediaResearchAgent (plan → execute → observe → evaluate → repeat)
    ↓
    Evidence accumulation (one or more Wikipedia articles)
    ↓
    Grounded Generation    (Stage 2 grounded_generator + ollama_service)
    ↓
    Answer Validation      (Stage 2 answer_validator — heuristic)
    ↓
    Source Attribution     (Stage 2 source_attributor, all retrieved pages)
    ↓
    Streaming CapabilityResult

Stage 2 subsystems (grounded_generator, answer_validator, source_attributor,
ollama_service, conversation_manager) are reused without modification.
"""
import logging
import re
import time
import unicodedata
import uuid
from typing import List, Optional, Set

from app.agent.capability import Capability, CapabilityRequest, CapabilityResult, CapabilityStatus
from app.agent.reflection import reflection_engine, ReflectionResult, ReflectionStatus
from app.agent.research import wikipedia_research_agent
from app.agent.research.action import ResearchAction, ResearchActionType
from app.agent.research.evidence import CompletionStatus
from app.knowledge import GroundingStatus, KnowledgeWorkspace, knowledge_extractor, reasoning_engine
from app.core.grounded_generator import (
    grounded_generator, GenerationInput, INSUFFICIENT_EVIDENCE_RESPONSE,
)
from app.core.answer_validator import answer_validator, ValidationResult
from app.core.query_understanding import IntentCategory
from app.core.source_attribution import SourceMetadata
from app.core.conversation_manager import conversation_manager
from app.core.decision_policy import ResponseMode
from app.services.ollama_service import ollama_service

logger = logging.getLogger("nexia.agent.wikipedia")

CAPABILITY_ID = "wikipedia_knowledge"


def _norm(text: str) -> str:
    """Normalize text for duplicate check."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().lower()


class WikipediaKnowledgeCapability(Capability):
    """
    Agentic Wikipedia knowledge capability with Stage 5/6 Structured Knowledge & Reflection.

    Orchestrates the full research, structured knowledge extraction, multi-hop reasoning,
    reflection, repair, and grounded generation pipeline:
      1. Initial research planning & execution
      2. Structured Knowledge extraction (KnowledgeWorkspace: entities, facts, relationships)
      3. Multi-hop and multi-entity reasoning
      4. Grounded candidate generation
      5. Answer validation (Stage 2 answer_validator)
      6. Self-reflection (Stage 5 reflection_engine)
      7. Bounded repair loop (targeted Wikipedia research or generation retry)
      8. Final source attribution and streaming delivery
    """

    @property
    def id(self) -> str:
        return CAPABILITY_ID

    @property
    def description(self) -> str:
        return (
            "Performs agentic Wikipedia research with structured knowledge representation, "
            "multi-hop reasoning, adaptive self-reflection, and grounded factual response generation."
        )

    async def execute(self, request: CapabilityRequest) -> CapabilityResult:
        ctx = request.context
        state = ctx.conversation_state
        cid = ctx.conversation_id
        plan = request.input["plan"]
        qu = request.input["qu"]
        user_id = ctx.metadata.get("user_id", "default_user")
        t_start = time.monotonic()

        # ── Check for Resumed Session or New Research Session ─────────────────
        from app.sessions.manager import research_session_manager
        from app.sessions.models import ResearchSession, SessionStatus

        resumed_session: Optional[ResearchSession] = request.input.get("resumed_session")
        existing_workspace: Optional[KnowledgeWorkspace] = None
        existing_research_ctx: Optional[ResearchContext] = None
        active_session: Optional[ResearchSession] = None

        if resumed_session:
            existing_research_ctx, existing_workspace = research_session_manager.restore_research_context(resumed_session)
            active_session = resumed_session
        else:
            # Determine if this represents meaningful long-horizon research
            # Simple factual queries ("What is photosynthesis?") stay lightweight.
            is_deep_research = (
                request.input.get("create_session", False)
                or qu.intent == IntentCategory.COMPARISON
                or getattr(plan, "is_multi_entity", False)
                or "research" in ctx.original_query.lower()
                or "compare" in ctx.original_query.lower()
                or "investigate" in ctx.original_query.lower()
            )
            if is_deep_research:
                active_session = research_session_manager.create_session(
                    objective=ctx.resolved_query,
                    original_request=ctx.original_query,
                    resolved_request=ctx.resolved_query,
                    user_id=user_id,
                    conversation_id=cid,
                    title=ctx.resolved_query[:60],
                )

        # ── Phase 1: Agentic Wikipedia research ───────────────────────────────
        research_ctx = await wikipedia_research_agent.run(
            original_query=ctx.original_query,
            resolved_query=ctx.resolved_query,
            resolved_entity=ctx.resolved_entity,
            qu=qu,
            conversation_id=cid,
            request_id=ctx.request_id,
            existing_context=existing_research_ctx,
        )
        ret_ms = int((time.monotonic() - t_start) * 1000)
        state.metadata["latest_research_context"] = research_ctx.to_summary_dict()

        # Record user message in conversation history
        conversation_manager.add_message(cid, "user", ctx.original_query)

        # ── Phase 2: Adaptive Reflection, Structured Knowledge & Repair Loop ──
        max_repair_attempts = 2
        repair_attempt = 0
        executed_repair_queries: Set[str] = set()
        final_answer = ""
        val_res: Optional[ValidationResult] = None
        refl_res: Optional[ReflectionResult] = None

        while True:
            # ── 2a. Structured Knowledge Extraction & Reasoning ────────────────
            workspace = KnowledgeWorkspace(run_id=ctx.request_id)
            if existing_workspace:
                for k, ent in existing_workspace.entities.items():
                    workspace.entities[k] = ent
                workspace.facts.extend(existing_workspace.facts)
                workspace.relationships.extend(existing_workspace.relationships)
                workspace.claims.extend(existing_workspace.claims)
                workspace.contradictions.extend(existing_workspace.contradictions)
                workspace.evidence_references.extend(existing_workspace.evidence_references)

            for ev_item in research_ctx.evidence:
                knowledge_extractor.extract_from_evidence(ev_item, workspace)

            structured_summary = reasoning_engine.build_reasoned_knowledge_summary(
                workspace, ctx.resolved_query, qu
            )
            state.metadata["latest_knowledge_workspace"] = workspace.to_summary_dict()

            combined_retrieval = wikipedia_research_agent.build_combined_retrieval(research_ctx)
            augmented_query = wikipedia_research_agent.augment_query_for_intent(
                ctx.resolved_query, qu, len(research_ctx.evidence)
            )
            effective_profile = request.input.get("effective_profile")
            pers_guidance = effective_profile.to_guidance_text() if effective_profile else None

            gen_input = GenerationInput(
                original_query=ctx.original_query,
                resolved_query=augmented_query,
                conversation_context=state.messages,
                retrieved_evidence=combined_retrieval,
                structured_knowledge=structured_summary if structured_summary else None,
                personalization_guidance=pers_guidance,
                is_knowledge_query=True,
            )

            has_evidence = grounded_generator.has_sufficient_evidence(gen_input)
            if not has_evidence or research_ctx.completion_status == CompletionStatus.INSUFFICIENT_KNOWLEDGE:
                # Reflect on missing evidence
                refl_res = reflection_engine.reflect(
                    original_request=ctx.original_query,
                    resolved_request=ctx.resolved_query,
                    qu=qu,
                    research_ctx=research_ctx,
                    generated_answer=None,
                    validation_result=None,
                    repair_attempt=repair_attempt,
                    max_repair_attempts=max_repair_attempts,
                    executed_repair_queries=executed_repair_queries,
                )
                state.metadata["latest_reflection"] = refl_res.model_dump()

                if (
                    refl_res.status == ReflectionStatus.NEEDS_RESEARCH_REPAIR
                    and refl_res.corrective_action
                ):
                    repair_attempt += 1
                    act_data = refl_res.corrective_action
                    query_str = act_data.get("query", refl_res.suggested_query or "")
                    executed_repair_queries.add(_norm(query_str))
                    repair_action = ResearchAction(
                        action_id=str(uuid.uuid4())[:8],
                        action_type=ResearchActionType(
                            act_data.get("type", ResearchActionType.WIKIPEDIA_RETRIEVE.value)
                        ),
                        objective=act_data.get("objective", f"Targeted repair for {query_str}"),
                        query=query_str,
                        entity=act_data.get("entity", refl_res.target_entity),
                    )
                    success = await wikipedia_research_agent.execute_repair_action(
                        repair_action, research_ctx
                    )
                    if success:
                        continue  # retry with newly accumulated evidence

                final_answer = INSUFFICIENT_EVIDENCE_RESPONSE
                break

            # ── Generate candidate answer ─────────────────────────────────────
            prompt = grounded_generator.build_prompt(gen_input)
            candidate_answer = await ollama_service.generate_response(prompt)
            candidate_answer = candidate_answer.strip()

            # ── Validate candidate answer ─────────────────────────────────────
            if plan.validation_required:
                try:
                    val_res = await answer_validator.validate(
                        answer=candidate_answer,
                        evidence=combined_retrieval,
                        llm=ollama_service,
                        workspace=workspace,
                        evidence_items=research_ctx.evidence,
                        information_needs=getattr(research_ctx, "information_needs", None),
                    )
                    state.metadata["latest_validation_result"] = val_res.model_dump()
                    if val_res.claims:
                        state.metadata["latest_claims"] = [c.model_dump() for c in val_res.claims]
                except Exception as exc:
                    logger.warning("[WIKI CAP] validation exception: %s", exc)
                    val_res = None

            # ── Reflect on validation result & state ──────────────────────────
            refl_res = reflection_engine.reflect(
                original_request=ctx.original_query,
                resolved_request=ctx.resolved_query,
                qu=qu,
                research_ctx=research_ctx,
                generated_answer=candidate_answer,
                validation_result=val_res,
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
                executed_repair_queries=executed_repair_queries,
            )
            state.metadata["latest_reflection"] = refl_res.model_dump()

            if refl_res.status == ReflectionStatus.COMPLETE:
                final_answer = candidate_answer
                break

            elif refl_res.status == ReflectionStatus.NEEDS_RESEARCH_REPAIR:
                repair_attempt += 1
                act_data = refl_res.corrective_action or {}
                query_str = act_data.get("query", refl_res.suggested_query or "")
                executed_repair_queries.add(_norm(query_str))
                repair_action = ResearchAction(
                    action_id=str(uuid.uuid4())[:8],
                    action_type=ResearchActionType(
                        act_data.get("type", ResearchActionType.WIKIPEDIA_RETRIEVE.value)
                    ),
                    objective=act_data.get("objective", f"Targeted research for {query_str}"),
                    query=query_str,
                    entity=act_data.get("entity", refl_res.target_entity),
                )
                await wikipedia_research_agent.execute_repair_action(
                    repair_action, research_ctx
                )
                continue

            elif refl_res.status == ReflectionStatus.GENERATION_REPAIR_REQUIRED:
                repair_attempt += 1
                # Regenerate with existing evidence
                continue

            else:
                # UNRECOVERABLE, CANCELLED, or exhausted budget
                final_answer = INSUFFICIENT_EVIDENCE_RESPONSE
                break

        # ── Phase 3: Final Source Attribution ─────────────────────────────────
        final_sources: List[SourceMetadata] = []
        if plan.attribution_required and research_ctx.evidence:
            final_sources = wikipedia_research_agent.build_multi_sources(
                research_ctx, val_res
            )
            state.metadata["latest_sources"] = [s.model_dump() for s in final_sources]

        # ── Persist Research Session Checkpoint ───────────────────────────────
        if active_session:
            if refl_res and refl_res.status == ReflectionStatus.COMPLETE and final_answer != INSUFFICIENT_EVIDENCE_RESPONSE:
                research_session_manager.complete_session(
                    active_session.session_id,
                    research_ctx=research_ctx,
                    workspace=workspace,
                    claims=val_res.claims if val_res else None,
                    sources=final_sources,
                )
            else:
                research_session_manager.snapshot_research_context(
                    active_session,
                    research_ctx=research_ctx,
                    workspace=workspace,
                    claims=val_res.claims if val_res else None,
                    sources=final_sources,
                )
                research_session_manager.save_session(active_session)

        # ── Phase 4: Streaming generator ──────────────────────────────────────
        async def stream_gen():
            conversation_manager.add_message(cid, "assistant", final_answer)
            total_ms = int((time.monotonic() - t_start) * 1000)
            logger.info(
                "[WIKI CAP] cid=%s actions=%d repairs=%d evidence=%d "
                "refl=%s total_ms=%d chars=%d pages=%s",
                cid[:8],
                len(research_ctx.completed_actions),
                repair_attempt,
                len(research_ctx.evidence),
                refl_res.status.value if refl_res else "NONE",
                total_ms,
                len(final_answer),
                [e.page_title for e in research_ctx.evidence],
            )
            # Yield in word chunks to support SSE streaming
            words = re.findall(r"\S+|\s+", final_answer)
            for w in words:
                yield w

        return CapabilityResult(
            status=CapabilityStatus.SUCCESS,
            stream=stream_gen(),
            sources=final_sources,
            metadata={
                "capability": CAPABILITY_ID,
                "conversation_id": cid,
                "session_id": active_session.session_id if active_session else None,
                "repair_attempts": repair_attempt,
                "reflection_status": refl_res.status.value if refl_res else None,
                "research": {
                    "actions": len(research_ctx.completed_actions),
                    "evidence_items": len(research_ctx.evidence),
                    "completion_status": research_ctx.completion_status.value,
                    "evidence_pages": [e.page_title for e in research_ctx.evidence],
                },
                "claims": {
                    "count": len(val_res.claims) if val_res else 0,
                    "supported": len([c for c in val_res.claims if c.grounding_status == GroundingStatus.SUPPORTED]) if val_res else 0,
                    "unsupported": len(val_res.unsupported_claims) if val_res else 0,
                },
            },
        )


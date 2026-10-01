import time
import uuid
import logging
from typing import AsyncGenerator, List, Optional, Tuple

from app.core.llm_interface import BaseLanguageModel
from app.core.conversation_manager import ConversationManager, ConversationState, conversation_manager
from app.core.query_understanding import QueryAnalyzer, QueryUnderstanding, IntentCategory, query_analyzer
from app.core.context_manager import ContextManager, ContextResolution, ResolutionStatus, context_manager
from app.core.decision_policy import DecisionPolicy, ResponsePlan, ResponseMode, decision_policy
from app.services.wikipedia_retriever import WikipediaRetriever, RetrievalResult, RetrievalStatus, wikipedia_retriever
from app.core.grounded_generator import GroundedGenerator, GenerationInput, INSUFFICIENT_EVIDENCE_RESPONSE, grounded_generator
from app.core.answer_validator import AnswerValidator, ValidationResult, ValidationStatus, answer_validator
from app.core.source_attribution import SourceAttributor, SourceMetadata, FinalResponse, source_attributor
from app.services.ollama_service import ollama_service

logger = logging.getLogger("nexia.core")

logger_debug = logging.getLogger("nexia.pipeline")

UNVERIFIED_ANSWER_RESPONSE = "I couldn't verify that answer against the available Wikipedia evidence."


def _ms(start: float) -> int:
    """Return elapsed milliseconds since `start`."""
    return int((time.monotonic() - start) * 1000)


class NexiaCore:
    """
    Nexia Core Application AI Orchestration Layer.

    Owns application-level AI capabilities:
    - Conversation State & History Management
    - Query Understanding & Intent Parsing  (deterministic fast-path + LLM fallback)
    - Context & Coreference Resolution
    - Decision Policy & Response Orchestration
    - Grounded Knowledge Retrieval (Wikipedia)
    - Grounded Answer Generation
    - Answer Validation & Verification
    - Source Attribution & Citations
    - Pipeline Observability (per-stage timing)
    - LLM Provider Abstraction
    """

    def __init__(
        self,
        llm: BaseLanguageModel = None,
        conv_mgr: ConversationManager = None,
        analyzer: QueryAnalyzer = None,
        context_mgr: ContextManager = None,
        policy: DecisionPolicy = None,
        wiki_retriever: WikipediaRetriever = None,
        generator: GroundedGenerator = None,
        validator: AnswerValidator = None,
        attributor: SourceAttributor = None,
    ):
        self.llm = llm or ollama_service
        self.conversation_manager = conv_mgr or conversation_manager
        self.query_analyzer = analyzer or query_analyzer
        self.context_manager = context_mgr or context_manager
        self.decision_policy = policy or decision_policy
        self.wikipedia_retriever = wiki_retriever or wikipedia_retriever
        self.grounded_generator = generator or grounded_generator
        self.answer_validator = validator or answer_validator
        self.source_attributor = attributor or source_attributor

    async def check_health(self) -> bool:
        """Check if the underlying AI runtime infrastructure is healthy."""
        return await self.llm.check_health()

    async def process_chat_stream(
        self, user_message: str, conversation_id: Optional[str] = None
    ) -> Tuple[AsyncGenerator[str, None], str, List[SourceMetadata]]:
        """
        Orchestrate streaming conversational processing.

        Pipeline: QU → Context → Decision → Retrieval → Generation → Validation → Attribution
        Returns: (stream_generator, conversation_id, initial_sources)
        """
        req_id = str(uuid.uuid4())
        req_start = time.monotonic()
        timing: dict = {}

        # 1. Conversation state
        state = self.conversation_manager.get_or_create_conversation(conversation_id)
        cid = state.conversation_id

        # 2. Query Understanding
        t = time.monotonic()
        qu: QueryUnderstanding = await self.query_analyzer.analyze(user_message, self.llm)
        timing["query_understanding_ms"] = _ms(t)
        state.metadata["latest_query_understanding"] = qu.model_dump()

        # 3. Context & Coreference Resolution
        t = time.monotonic()
        resolution: ContextResolution = await self.context_manager.resolve(qu, state, self.llm)
        timing["context_resolution_ms"] = _ms(t)
        state.metadata["latest_context_resolution"] = resolution.model_dump()

        logger_debug.debug(
            "[PIPELINE/stream] qu: intent=%s entity=%r info=%r deterministic=%s | "
            "ctx: original=%r resolved=%r refs=%r status=%s",
            qu.intent, qu.entity, qu.information_needed, qu.deterministic,
            resolution.original_query, resolution.resolved_query,
            resolution.resolved_references, resolution.resolution_status,
        )

        # 4. Decision Policy
        t = time.monotonic()
        plan: ResponsePlan = self.decision_policy.create_plan(qu, resolution)
        timing["decision_policy_ms"] = _ms(t)
        state.metadata["latest_response_plan"] = plan.model_dump()

        logger_debug.debug(
            "[PIPELINE/stream] decision: mode=%s retrieval=%s generation=%s validation=%s attribution=%s",
            plan.response_mode, plan.retrieval_required, plan.generation_required,
            plan.validation_required, plan.attribution_required,
        )

        # 5. Wikipedia Retrieval (if required)
        retrieval: Optional[RetrievalResult] = None
        timing["retrieval_ms"] = 0
        if plan.retrieval_required:
            t = time.monotonic()
            retrieval = await self.wikipedia_retriever.retrieve(
                query=resolution.resolved_query,
                entity=resolution.resolved_entity,
            )
            timing["retrieval_ms"] = _ms(t)
            state.metadata["latest_retrieval_result"] = retrieval.model_dump()

            logger_debug.debug(
                "[PIPELINE/stream] retrieval: status=%s page=%r url=%s content_len=%d",
                retrieval.status,
                retrieval.page_title,
                retrieval.page_url or "-",
                len(retrieval.content or ""),
            )

        # 6. Assemble Generation Input
        is_knowledge = (plan.response_mode == ResponseMode.GROUNDED_KNOWLEDGE)
        gen_input = GenerationInput(
            original_query=user_message,
            resolved_query=resolution.resolved_query,
            conversation_context=state.messages,
            retrieved_evidence=retrieval,
            answer_contract=getattr(qu, "answer_contract", None),
            is_knowledge_query=is_knowledge,
        )

        # Downgrade to UNSUPPORTED if evidence is empty
        if is_knowledge and not self.grounded_generator.has_sufficient_evidence(gen_input):
            plan.response_mode = ResponseMode.UNSUPPORTED_KNOWLEDGE

        # 7. Record user message
        self.conversation_manager.add_message(cid, "user", user_message)

        sources: List[SourceMetadata] = []

        # ── Stream generator (inner coroutine) ────────────────────────────
        async def stream_generator():
            nonlocal sources
            accumulated_text = ""
            val_res: Optional[ValidationResult] = None
            gen_start = time.monotonic()
            try:
                if plan.response_mode == ResponseMode.UNSUPPORTED_KNOWLEDGE:
                    accumulated_text = INSUFFICIENT_EVIDENCE_RESPONSE
                    yield INSUFFICIENT_EVIDENCE_RESPONSE
                elif plan.generation_required:
                    grounded_prompt = self.grounded_generator.build_prompt(gen_input)
                    async for token in self.llm.stream_response(grounded_prompt):
                        accumulated_text += token
                        yield token
            finally:
                timing["generation_ms"] = _ms(gen_start)

                if accumulated_text.strip():
                    # Validation
                    t_val = time.monotonic()
                    if (
                        plan.validation_required
                        and self.grounded_generator.has_sufficient_evidence(gen_input)
                        and accumulated_text != INSUFFICIENT_EVIDENCE_RESPONSE
                    ):
                        try:
                            val_res = await self.answer_validator.validate(
                                accumulated_text,
                                retrieval,
                                self.llm,
                                answer_contract=getattr(qu, "answer_contract", None),
                            )
                            state.metadata["latest_validation_result"] = val_res.model_dump()
                        except Exception as e:
                            val_res = ValidationResult(
                                status=ValidationStatus.VALIDATION_ERROR,
                                reason=f"Validation exception: {str(e)}",
                                metadata={"error": str(e)},
                            )
                            state.metadata["latest_validation_result"] = val_res.model_dump()
                    timing["validation_ms"] = _ms(t_val)

                    # Source Attribution
                    t_attr = time.monotonic()
                    verified_sources = []
                    if plan.attribution_required:
                        verified_sources = self.source_attributor.build_sources(retrieval, val_res)
                        state.metadata["latest_sources"] = [s.model_dump() for s in verified_sources]
                    timing["attribution_ms"] = _ms(t_attr)

                    timing["total_ms"] = _ms(req_start)

                    # Execution metadata
                    execution_record = {
                        "request_id": req_id,
                        "intent": qu.intent,
                        "deterministic_qu": qu.deterministic,
                        "resolved_query": resolution.resolved_query,
                        "response_mode": plan.response_mode,
                        "retrieval_used": plan.retrieval_required and retrieval is not None,
                        "retrieval_success": (retrieval.status == RetrievalStatus.SUCCESS) if retrieval else False,
                        "generation_used": plan.generation_required,
                        "validation_status": val_res.status if val_res else "SKIPPED",
                        "sources": [s.model_dump() for s in verified_sources],
                        "timing": timing,
                    }
                    state.metadata["latest_execution"] = execution_record

                    logger.info(
                        "[NEXIA TRACE] req=%s mode=%s qu=%dms ctx=%dms dec=%dms ret=%dms gen=%dms val=%dms attr=%dms total=%dms",
                        req_id[:8],
                        plan.response_mode,
                        timing.get("query_understanding_ms", 0),
                        timing.get("context_resolution_ms", 0),
                        timing.get("decision_policy_ms", 0),
                        timing.get("retrieval_ms", 0),
                        timing.get("generation_ms", 0),
                        timing.get("validation_ms", 0),
                        timing.get("attribution_ms", 0),
                        timing.get("total_ms", 0),
                    )

                    self.conversation_manager.add_message(cid, "assistant", accumulated_text)

        # Pre-build initial sources for response header (before streaming starts)
        if is_knowledge and self.grounded_generator.has_sufficient_evidence(gen_input):
            sources = self.source_attributor.build_sources(retrieval, None)

        return stream_generator(), cid, sources

    async def process_chat(
        self, user_message: str, conversation_id: Optional[str] = None
    ) -> Tuple[FinalResponse, str]:
        """
        Orchestrate non-streaming conversational processing.

        Pipeline: QU → Context → Decision → Retrieval → Generation → Validation → Attribution
        """
        req_id = str(uuid.uuid4())
        req_start = time.monotonic()
        timing: dict = {}

        state = self.conversation_manager.get_or_create_conversation(conversation_id)
        cid = state.conversation_id

        # Query Understanding
        t = time.monotonic()
        qu: QueryUnderstanding = await self.query_analyzer.analyze(user_message, self.llm)
        timing["query_understanding_ms"] = _ms(t)
        state.metadata["latest_query_understanding"] = qu.model_dump()

        # Context & Coreference Resolution
        t = time.monotonic()
        resolution: ContextResolution = await self.context_manager.resolve(qu, state, self.llm)
        timing["context_resolution_ms"] = _ms(t)
        state.metadata["latest_context_resolution"] = resolution.model_dump()

        logger_debug.debug(
            "[PIPELINE] qu: intent=%s entity=%r info=%r deterministic=%s | "
            "ctx: original=%r resolved=%r refs=%r status=%s",
            qu.intent, qu.entity, qu.information_needed, qu.deterministic,
            resolution.original_query, resolution.resolved_query,
            resolution.resolved_references, resolution.resolution_status,
        )

        # Decision Policy
        t = time.monotonic()
        plan: ResponsePlan = self.decision_policy.create_plan(qu, resolution)
        timing["decision_policy_ms"] = _ms(t)
        state.metadata["latest_response_plan"] = plan.model_dump()

        logger_debug.debug(
            "[PIPELINE] decision: mode=%s retrieval=%s generation=%s validation=%s attribution=%s",
            plan.response_mode, plan.retrieval_required, plan.generation_required,
            plan.validation_required, plan.attribution_required,
        )

        # Wikipedia Retrieval
        retrieval: Optional[RetrievalResult] = None
        timing["retrieval_ms"] = 0
        if plan.retrieval_required:
            t = time.monotonic()
            retrieval = await self.wikipedia_retriever.retrieve(
                query=resolution.resolved_query,
                entity=resolution.resolved_entity,
            )
            timing["retrieval_ms"] = _ms(t)
            state.metadata["latest_retrieval_result"] = retrieval.model_dump()

            logger_debug.debug(
                "[PIPELINE] retrieval: status=%s page=%r url=%s content_len=%d",
                retrieval.status,
                retrieval.page_title,
                retrieval.page_url or "-",
                len(retrieval.content or ""),
            )

        is_knowledge = (plan.response_mode == ResponseMode.GROUNDED_KNOWLEDGE)
        gen_input = GenerationInput(
            original_query=user_message,
            resolved_query=resolution.resolved_query,
            conversation_context=state.messages,
            retrieved_evidence=retrieval,
            answer_contract=getattr(qu, "answer_contract", None),
            is_knowledge_query=is_knowledge,
        )

        if is_knowledge and not self.grounded_generator.has_sufficient_evidence(gen_input):
            plan.response_mode = ResponseMode.UNSUPPORTED_KNOWLEDGE

        self.conversation_manager.add_message(cid, "user", user_message)

        val_res: Optional[ValidationResult] = None

        # Generation
        t = time.monotonic()
        if plan.response_mode == ResponseMode.UNSUPPORTED_KNOWLEDGE:
            response_text = INSUFFICIENT_EVIDENCE_RESPONSE
        elif plan.generation_required:
            grounded_prompt = self.grounded_generator.build_prompt(gen_input)
            response_text = await self.llm.generate_response(grounded_prompt)
        else:
            response_text = ""
        timing["generation_ms"] = _ms(t)

        # Validation
        t = time.monotonic()
        if (
            plan.validation_required
            and self.grounded_generator.has_sufficient_evidence(gen_input)
            and response_text not in (INSUFFICIENT_EVIDENCE_RESPONSE, UNVERIFIED_ANSWER_RESPONSE, "")
        ):
            try:
                val_res = await self.answer_validator.validate(
                    response_text,
                    retrieval,
                    self.llm,
                    answer_contract=getattr(qu, "answer_contract", None),
                )
                state.metadata["latest_validation_result"] = val_res.model_dump()
                if val_res.status == ValidationStatus.UNSUPPORTED:
                    response_text = UNVERIFIED_ANSWER_RESPONSE
            except Exception as e:
                val_res = ValidationResult(
                    status=ValidationStatus.VALIDATION_ERROR,
                    reason=f"Validation exception: {str(e)}",
                    metadata={"error": str(e)},
                )
                state.metadata["latest_validation_result"] = val_res.model_dump()
        timing["validation_ms"] = _ms(t)

        # Source Attribution
        t = time.monotonic()
        verified_sources = []
        if plan.attribution_required:
            verified_sources = self.source_attributor.build_sources(retrieval, val_res)
            state.metadata["latest_sources"] = [s.model_dump() for s in verified_sources]
        timing["attribution_ms"] = _ms(t)

        timing["total_ms"] = _ms(req_start)

        execution_record = {
            "request_id": req_id,
            "intent": qu.intent,
            "deterministic_qu": qu.deterministic,
            "resolved_query": resolution.resolved_query,
            "response_mode": plan.response_mode,
            "retrieval_used": plan.retrieval_required and retrieval is not None,
            "retrieval_success": (retrieval.status == RetrievalStatus.SUCCESS) if retrieval else False,
            "generation_used": plan.generation_required,
            "validation_status": val_res.status if val_res else "SKIPPED",
            "sources": [s.model_dump() for s in verified_sources],
            "timing": timing,
        }
        state.metadata["latest_execution"] = execution_record

        logger.info(
            "[NEXIA TRACE] req=%s mode=%s qu=%dms ctx=%dms dec=%dms ret=%dms gen=%dms val=%dms attr=%dms total=%dms",
            req_id[:8],
            plan.response_mode,
            timing.get("query_understanding_ms", 0),
            timing.get("context_resolution_ms", 0),
            timing.get("decision_policy_ms", 0),
            timing.get("retrieval_ms", 0),
            timing.get("generation_ms", 0),
            timing.get("validation_ms", 0),
            timing.get("attribution_ms", 0),
            timing.get("total_ms", 0),
        )

        self.conversation_manager.add_message(cid, "assistant", response_text)

        final_resp = FinalResponse(
            content=response_text,
            sources=verified_sources,
            metadata={"conversation_id": cid, "latest_execution": execution_record},
        )
        return final_resp, cid


# Global Nexia Core instance
nexia_core = NexiaCore()

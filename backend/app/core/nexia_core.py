from typing import AsyncGenerator, Optional, Tuple
from app.core.llm_interface import BaseLanguageModel
from app.core.conversation_manager import ConversationManager, ConversationState, conversation_manager
from app.core.query_understanding import QueryAnalyzer, QueryUnderstanding, IntentCategory, query_analyzer
from app.core.context_manager import ContextManager, ContextResolution, ResolutionStatus, context_manager
from app.services.wikipedia_retriever import WikipediaRetriever, RetrievalResult, wikipedia_retriever
from app.core.grounded_generator import GroundedGenerator, GenerationInput, INSUFFICIENT_EVIDENCE_RESPONSE, grounded_generator
from app.services.ollama_service import ollama_service

class NexiaCore:
    """
    Nexia Core Application AI Orchestration Layer.

    Owns application-level AI capabilities:
    - Conversation State & History Management
    - Query Understanding & Intent Parsing
    - Context & Coreference Resolution
    - Grounded Knowledge Retrieval (Wikipedia)
    - Grounded Answer Generation
    - LLM Provider Abstraction
    """

    def __init__(
        self,
        llm: BaseLanguageModel = None,
        conv_mgr: ConversationManager = None,
        analyzer: QueryAnalyzer = None,
        context_mgr: ContextManager = None,
        wiki_retriever: WikipediaRetriever = None,
        generator: GroundedGenerator = None,
    ):
        self.llm = llm or ollama_service
        self.conversation_manager = conv_mgr or conversation_manager
        self.query_analyzer = analyzer or query_analyzer
        self.context_manager = context_mgr or context_manager
        self.wikipedia_retriever = wiki_retriever or wikipedia_retriever
        self.grounded_generator = generator or grounded_generator

    async def check_health(self) -> bool:
        """Check if the underlying AI runtime infrastructure is healthy."""
        return await self.llm.check_health()

    def _is_knowledge_query(self, qu: QueryUnderstanding, resolution: ContextResolution) -> bool:
        """Determine whether Wikipedia retrieval & factual grounding are required."""
        if qu.intent in [
            IntentCategory.FACTUAL_LOOKUP,
            IntentCategory.ENTITY_INFORMATION,
            IntentCategory.EXPLANATION,
            IntentCategory.COMPARISON,
            IntentCategory.DEFINITION,
            IntentCategory.SUMMARIZATION,
        ]:
            return True
        if qu.entity or resolution.resolved_entity:
            return True
        if resolution.resolution_status == ResolutionStatus.RESOLVED:
            return True
        return False

    async def process_chat_stream(
        self, user_message: str, conversation_id: Optional[str] = None
    ) -> Tuple[AsyncGenerator[str, None], str]:
        """
        Orchestrate conversational processing: understand query, resolve context, retrieve evidence, ground prompt, and stream answer tokens.
        
        Returns:
            Tuple of (stream_generator, conversation_id)
        """
        # 1. Retrieve or initialize conversation state
        state = self.conversation_manager.get_or_create_conversation(conversation_id)
        cid = state.conversation_id

        # 2. Execute Query Understanding
        qu: QueryUnderstanding = await self.query_analyzer.analyze(user_message, self.llm)
        state.metadata["latest_query_understanding"] = qu.model_dump()

        # 3. Execute Context & Coreference Resolution
        resolution: ContextResolution = await self.context_manager.resolve(qu, state, self.llm)
        state.metadata["latest_context_resolution"] = resolution.model_dump()

        # 4. Determine knowledge retrieval necessity
        is_knowledge = self._is_knowledge_query(qu, resolution)
        retrieval: Optional[RetrievalResult] = None

        if is_knowledge:
            retrieval = await self.wikipedia_retriever.retrieve(resolution.resolved_query)
            state.metadata["latest_retrieval_result"] = retrieval.model_dump()

        # 5. Assemble Generation Input
        gen_input = GenerationInput(
            original_query=user_message,
            resolved_query=resolution.resolved_query,
            conversation_context=state.messages,
            retrieved_evidence=retrieval,
            is_knowledge_query=is_knowledge,
        )

        # 6. Record user message in Nexia conversation state
        self.conversation_manager.add_message(cid, "user", user_message)

        # 7. Stream generator
        async def stream_generator():
            accumulated_text = ""
            try:
                # Handle knowledge query with missing/insufficient evidence cleanly
                if is_knowledge and not self.grounded_generator.has_sufficient_evidence(gen_input):
                    accumulated_text = INSUFFICIENT_EVIDENCE_RESPONSE
                    yield INSUFFICIENT_EVIDENCE_RESPONSE
                else:
                    grounded_prompt = self.grounded_generator.build_prompt(gen_input)
                    async for token in self.llm.stream_response(grounded_prompt):
                        accumulated_text += token
                        yield token
            finally:
                # 8. Finalize exactly ONE assistant message in conversation state
                if accumulated_text.strip():
                    self.conversation_manager.add_message(cid, "assistant", accumulated_text)

        return stream_generator(), cid

    async def process_chat(self, user_message: str, conversation_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Orchestrate conversational processing for non-streaming queries.
        """
        state = self.conversation_manager.get_or_create_conversation(conversation_id)
        cid = state.conversation_id
        
        qu: QueryUnderstanding = await self.query_analyzer.analyze(user_message, self.llm)
        state.metadata["latest_query_understanding"] = qu.model_dump()

        resolution: ContextResolution = await self.context_manager.resolve(qu, state, self.llm)
        state.metadata["latest_context_resolution"] = resolution.model_dump()

        is_knowledge = self._is_knowledge_query(qu, resolution)
        retrieval: Optional[RetrievalResult] = None

        if is_knowledge:
            retrieval = await self.wikipedia_retriever.retrieve(resolution.resolved_query)
            state.metadata["latest_retrieval_result"] = retrieval.model_dump()

        gen_input = GenerationInput(
            original_query=user_message,
            resolved_query=resolution.resolved_query,
            conversation_context=state.messages,
            retrieved_evidence=retrieval,
            is_knowledge_query=is_knowledge,
        )

        self.conversation_manager.add_message(cid, "user", user_message)

        if is_knowledge and not self.grounded_generator.has_sufficient_evidence(gen_input):
            response_text = INSUFFICIENT_EVIDENCE_RESPONSE
        else:
            grounded_prompt = self.grounded_generator.build_prompt(gen_input)
            response_text = await self.llm.generate_response(grounded_prompt)

        self.conversation_manager.add_message(cid, "assistant", response_text)
        return response_text, cid

# Global Nexia Core instance
nexia_core = NexiaCore()

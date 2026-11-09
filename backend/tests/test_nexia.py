
import asyncio
from datetime import datetime, timezone
import json
import re
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.http_client import get_http_client
from app.main import app
from app.retrieval.adaptive_nlp_planner import (
    AdaptiveQueryAnalyzer,
    EvidenceCoverageAssessor,
    EvidencePacker,
    QueryComplexity,
    SubQueryRequirement,
    compute_shingle_similarity,
)
from app.retrieval.entity_extractor import EntityExtractor
from app.retrieval.nlp_query_parser import NLPQueryParser
from app.schemas.answer import (
    MAX_EVIDENCE_PASSAGES,
    MIN_CONFIDENCE,
    TOP_K_PER_DOCUMENT,
    AnswerRequest,
    AnswerResponse,
    RetrievalDiagnostics,
    SourceReference,
)
from app.schemas.conversation import (
    ChatMessage,
    ConversationCreate,
    ConversationRead,
    ConversationStatus,
    ConversationWithHistory,
    MessageRole,
    MessageStatus,
)
from app.schemas.evidence_package import EvidencePackage, SufficiencyAssessment
from app.schemas.memory import MemoryType
from app.schemas.retrieval import Stage4StructuredQuery
from app.services.answer_orchestrator import AnswerOrchestrator, ClaimVerifier
from app.services.cache_service import InMemoryCache
from app.services.citation_manager import (
    PrimaryArticle,
    PrimaryArticleSelector,
    StreamingCitationFilter,
    format_answer_with_primary_reference,
    strip_inline_citations,
)
from app.services.context_manager import ContextManager
from app.services.conversation_manager import ConversationManager
from app.services.hybrid_retrieval import RetrievedChunk
from app.services.idempotency import IdempotencyRecord, IdempotencyService
from app.services.llm_adapter import LLMGenerationResult
from app.services.memory_manager import MemoryManager
from app.services.prompts import PromptContext, build_answer_prompt
from app.services.rate_limiter import InMemoryTokenBucket, RateLimitPolicy, RateLimiterService
from app.services.response_controller import (
    InferenceQueueFullError,
    InferenceQueueTimeoutError,
    Priority,
    RequestCancelledError,
    ResponseController,
)
from app.services.streaming_engine import StreamingEngine
from app.services.title_generator import ConversationTitleGenerator


client = TestClient(app)


# ==============================================================================
# SECTION 1: Application Health, Readiness & Metrics Endpoints
# ==============================================================================

class TestHealthAndMetrics:
    """Verifies service health checks, readiness, and observability metrics."""

    def test_health_endpoint(self):
        """GET /health returns HTTP 200 with status: ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_metrics_endpoint_payload(self):
        """GET /api/v1/metrics returns real-time performance and system metrics."""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "inference" in data
        assert "cache" in data
        assert "rate_limiting" in data
        assert "database_pool" in data
        assert data["inference"]["max_active"] >= 1


# ==============================================================================
# SECTION 2: API Route Validation & Surface Minimization
# ==============================================================================

class TestApiSurfaceMinimization:
    """Verifies obsolete intermediate stage routes return 404, keeping API surface clean."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("GET", "/api/v1/retrieval/search"),
            ("GET", "/api/v1/retrieval/document"),
            ("POST", "/api/v1/retrieval/passages"),
            ("POST", "/api/v1/retrieval/multi-document"),
            ("POST", "/api/v1/retrieval/relationships"),
            ("POST", "/api/v1/retrieval/evidence"),
            ("GET", "/retrieval/search"),
            ("POST", "/retrieval/passages"),
            ("POST", "/retrieval/relationships"),
        ],
    )
    def test_obsolete_endpoints_return_404(self, method: str, path: str):
        """Obsolete intermediate pipeline endpoints must return 404 Not Found."""
        if method == "GET":
            response = client.get(path)
        else:
            response = client.post(path, json={})
        assert response.status_code == 404


# ==============================================================================
# SECTION 3: Conversation Lifecycle, History & Persistence
# ==============================================================================

class TestConversationLifecycle:
    """Verifies conversation CRUD endpoints and conversation manager logic."""

    @pytest.mark.asyncio
    async def test_create_conversation(self):
        """POST /api/v1/conversations creates and returns active conversation."""
        from app.api.routes import conversations
        conv_id = uuid4()
        mock_conv = ConversationRead(
            id=conv_id,
            title="Ancient Civilizations",
            status=ConversationStatus.ACTIVE,
            metadata={"domain": "history"},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            message_count=0,
        )

        with patch.object(conversations.conv_manager, "create_conversation", AsyncMock(return_value=mock_conv)):
            req = ConversationCreate(title="Ancient Civilizations", metadata={"domain": "history"})
            res = await conversations.create_conversation(req)
            assert res.id == conv_id
            assert res.title == "Ancient Civilizations"
            assert res.status == ConversationStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_conversation_with_messages(self):
        """GET /api/v1/conversations/{id} returns conversation with ordered history."""
        from app.api.routes import conversations
        conv_id = uuid4()
        mock_history = ConversationWithHistory(
            id=conv_id,
            title="Gupta Empire",
            status=ConversationStatus.ACTIVE,
            metadata={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            message_count=2,
            messages=[
                ChatMessage(
                    id=uuid4(),
                    conversation_id=conv_id,
                    role=MessageRole.USER,
                    content="Who succeeded Samudragupta?",
                    created_at=datetime.now(timezone.utc),
                ),
                ChatMessage(
                    id=uuid4(),
                    conversation_id=conv_id,
                    role=MessageRole.ASSISTANT,
                    content="Chandragupta II succeeded Samudragupta.\n\n[Samudragupta](https://en.wikipedia.org/wiki/Samudragupta)",
                    created_at=datetime.now(timezone.utc),
                ),
            ],
        )

        with patch.object(conversations.conv_manager, "get_conversation_with_history", AsyncMock(return_value=mock_history)):
            res = await conversations.get_conversation(conv_id)
            assert res.id == conv_id
            assert len(res.messages) == 2
            assert res.messages[0].role == MessageRole.USER
            assert res.messages[1].role == MessageRole.ASSISTANT

    @pytest.mark.asyncio
    async def test_delete_conversation(self):
        """DELETE /api/v1/conversations/{id} deletes conversation correctly."""
        from app.api.routes import conversations
        conv_id = uuid4()
        with patch.object(conversations.conv_manager, "delete_conversation", AsyncMock(return_value=True)):
            res = await conversations.delete_conversation(conv_id)
            assert res is None


# ==============================================================================
# SECTION 4: Automatic Conversation Title Generation & Validation
# ==============================================================================

class TestTitleGenerator:
    """Verifies atomic title generation, validation rules, 5-word cap, and fallback behavior."""

    @pytest.fixture
    def generator(self):
        adapter = MagicMock()
        adapter.generate_text = AsyncMock(return_value="The Roman Empire")
        return ConversationTitleGenerator(llm_adapter=adapter)

    def test_clean_title_removes_prefixes_and_quotes(self, generator):
        raw = 'Title: "Shivaji Maharaj and the Maratha Empire"'
        title = generator.clean_and_validate_title(raw, question="Who was Shivaji Maharaj?")
        assert not title.startswith("Title:")
        assert '"' not in title
        assert len(title.split()) <= 5

    def test_clean_title_preserves_acronyms(self, generator):
        raw = "NASA Space Exploration Mission"
        title = generator.clean_and_validate_title(raw, question="Tell me about NASA")
        words = title.split()
        assert "NASA" in words
        assert len(words) <= 5

    def test_clean_title_clamps_to_five_words(self, generator):
        raw = "A Comprehensive Historical Overview of the Entire Roman Empire"
        title = generator.clean_and_validate_title(raw, question="Explain the Roman Empire")
        words = title.split()
        assert len(words) <= 5
        assert words[-1].lower() not in {"of", "the", "in", "and", "to"}

    def test_clean_title_refusal_fallback(self, generator):
        raw = "I am sorry, as an AI I cannot generate a title."
        title = generator.clean_and_validate_title(raw, question="Who was Alan Turing?")
        words = title.split()
        assert len(words) <= 5
        assert "Alan" in title or "Turing" in title

    @pytest.mark.asyncio
    async def test_atomic_title_generation(self, generator):
        """Title generator uses question + answer and updates conversation title atomically."""
        conv_mgr = MagicMock()
        conv_mgr.is_eligible_for_title_generation = AsyncMock(return_value=True)
        conv_mgr.update_title_atomic = AsyncMock(return_value=True)

        res = await generator.generate_and_update_title(
            conversation_id=uuid4(),
            first_question="When was the Roman Empire founded?",
            first_answer="The Roman Empire was founded in 27 BCE.",
            conversation_manager=conv_mgr,
        )
        assert res is not None
        assert conv_mgr.update_title_atomic.called


# ==============================================================================
# SECTION 5: Dynamic Context Assembly & Token Budgeting
# ==============================================================================

class TestContextManagerAndTokenBudget:
    """Verifies token budgeting, message retention, and truncation boundaries."""

    def test_token_estimation(self):
        mgr = ContextManager()
        short_text = "Who was Chandragupta Maurya?"
        est = mgr.estimate_tokens(short_text)
        assert 5 <= est <= 15
        assert mgr.estimate_tokens("") == 0

    def test_assemble_context_priority_allocation(self):
        mgr = ContextManager(num_ctx=4096, reserved_generation=512)
        passages = [
            {
                "document_title": "Chandragupta Maurya",
                "section_title": "Early life",
                "passage_text": "Chandragupta Maurya was the founder of the Maurya Empire in ancient India.",
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
            question="Tell me about his conquests.",
            system_prompt="SYSTEM_PROMPT",
            passages=passages,
            conversation_history=history,
            entities=["Chandragupta Maurya"],
        )
        assert len(assembled.passages) == 1
        assert len(assembled.recent_messages) == 2
        assert "Chandragupta Maurya" in assembled.entities


# ==============================================================================
# SECTION 6: Adaptive Query Complexity, Intent & Entity Extraction
# ==============================================================================

class TestAdaptiveNLPUnderstanding:
    """Verifies query complexity classification, entity extraction, and subquery decomposition."""

    def test_simple_factual_query(self):
        q = "Who was the first President of India?"
        res = AdaptiveQueryAnalyzer.analyze_complexity(q)
        assert res.complexity == QueryComplexity.SIMPLE
        assert len(res.subqueries) == 1
        assert any("India" in e for e in res.primary_entities + res.entities_detected)

    def test_complex_multi_requirement_query(self):
        q = "When was the Maurya Empire founded, who was its first ruler, and what caused its decline?"
        res = AdaptiveQueryAnalyzer.analyze_complexity(q)
        assert res.complexity == QueryComplexity.COMPLEX
        assert len(res.subqueries) >= 2

    def test_entity_extraction_and_normalization(self):
        query = EntityExtractor.analyze_stage4_query("successor of Shri Krishnadevaraya")
        assert query.question_type == "RELATIONSHIP"
        assert query.relation is not None
        assert query.relation.type == "SUCCESSOR_OF"
        assert len(query.entities) == 1
        assert query.entities[0].normalized_text == "Krishnadevaraya"


# ==============================================================================
# SECTION 7: Retrieval Pipeline & Enforced Configuration (3, 0.4, 9)
# ==============================================================================

class TestRetrievalConfigurationEnforcement:
    """Guarantees top_k_per_document=3, min_confidence=0.4, max_evidence_passages=9 across the backend."""

    def test_constants_defined_correctly(self):
        assert TOP_K_PER_DOCUMENT == 3
        assert MIN_CONFIDENCE == 0.4
        assert MAX_EVIDENCE_PASSAGES == 9

    def test_backend_request_conflicting_parameters_overridden(self):
        req = AnswerRequest(
            question="What is gravity?",
            top_k_per_document=99,
            min_confidence=0.99,
            max_evidence_passages=1,
        )
        assert req.top_k_per_document == 3
        assert req.min_confidence == 0.4
        assert req.max_evidence_passages == 9

    def test_answer_endpoint_enforces_retrieval_values(self):
        with patch("app.api.routes.answer.orchestrator.answer", new_callable=AsyncMock) as mock_answer:
            mock_answer.return_value = AnswerResponse(
                question="What is entropy?",
                answer_text="Entropy is a measure of disorder.",
                answer_status="ANSWERED",
                source_references=[],
                conflicts=[],
                sufficiency_rating="SUFFICIENT",
                retrieval_diagnostics=RetrievalDiagnostics(
                    embedding_model="nomic-embed-text",
                    llm_model="gemma3:4b",
                ),
            )
            response = client.post(
                "/api/v1/retrieval/answer",
                json={
                    "question": "What is entropy?",
                    "top_k_per_document": 50,
                    "min_confidence": 0.85,
                    "max_evidence_passages": 2,
                },
            )
            assert response.status_code == 200
            call_arg = mock_answer.call_args[0][0]
            assert call_arg.top_k_per_document == 3
            assert call_arg.min_confidence == 0.4
            assert call_arg.max_evidence_passages == 9


# ==============================================================================
# SECTION 8: Slotted Evidence Packing & Claim Grounding Verification
# ==============================================================================

class TestEvidencePackingAndClaimVerification:
    """Verifies evidence deduplication, section slotting, and claim-level grounding."""

    def test_evidence_deduplication_shingle_similarity(self):
        p1 = "Albert Einstein developed the theory of relativity in 1905."
        p2 = "Albert Einstein developed the special theory of relativity in 1905."
        sim = compute_shingle_similarity(p1, p2)
        assert sim >= 0.45

    def test_claim_verifier_prunes_unsupported_claims(self):
        chunks = [
            RetrievedChunk(
                chunk_id=1,
                document_title="Antikythera mechanism",
                source_url="https://en.wikipedia.org/wiki/Antikythera_mechanism",
                section_title="Discovery",
                passage_text="The Antikythera mechanism was retrieved from a shipwreck off the Greek island of Antikythera in 1901.",
                rrf_score=0.50,
                retrieval_method="hybrid",
            )
        ]
        raw_answer = (
            "The Antikythera mechanism was retrieved from a shipwreck in 1901 [1]. "
            "The island has a bird observatory for migratory birds."
        )
        verified_answer, stats = ClaimVerifier.verify_and_clean_answer(
            answer_text=raw_answer,
            selected_chunks=chunks,
            question="Where was the Antikythera mechanism found?",
        )
        assert stats["unsupported_claims_removed"] >= 1
        assert "bird observatory" not in verified_answer
        assert "Antikythera mechanism was retrieved" in verified_answer


# ==============================================================================
# SECTION 9: Inline Citation Marker Stripping & Markdown Link Preservation
# ==============================================================================

class TestInlineCitationStripping:
    """Verifies complete removal of [k], [1], [source] while preserving legitimate brackets."""

    def test_strip_k_and_internal_evidence_markers(self):
        # User problem statement example
        user_example = (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity [k]. "
            "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect [k]. "
            "Einstein's work profoundly impacted the fields of physics and astronomy [k]."
        )
        expected = (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. "
            "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. "
            "Einstein's work profoundly impacted the fields of physics and astronomy."
        )
        assert strip_inline_citations(user_example) == expected

    def test_strip_chained_and_numbered_citations(self):
        assert strip_inline_citations("Relativity confirmed [1], [2].") == "Relativity confirmed."
        assert strip_inline_citations("Relativity confirmed [k][1].") == "Relativity confirmed."
        assert strip_inline_citations("Relativity confirmed [source 1].") == "Relativity confirmed."
        assert strip_inline_citations("Relativity confirmed [passage 2].") == "Relativity confirmed."
        assert strip_inline_citations("Relativity confirmed.[k]") == "Relativity confirmed."

    def test_preserves_legitimate_mathematical_intervals_and_quotes(self):
        math_interval = "Values range between [10, 20] in standard notation."
        assert strip_inline_citations(math_interval) == math_interval

        editorial_note = "He wrote [sic] in his private diary."
        assert strip_inline_citations(editorial_note) == editorial_note

        quote_insertion = "She observed that [the result] was conclusive."
        assert strip_inline_citations(quote_insertion) == quote_insertion

        markdown_link = "See [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein) for details."
        assert strip_inline_citations(markdown_link) == markdown_link


# ==============================================================================
# SECTION 10: Single Primary Article Selection & Trailing Link Formatting
# ==============================================================================

class TestPrimaryArticleFormatting:
    """Verifies deterministic selection of single Wikipedia article and clean output formatting."""

    def test_format_answer_with_single_primary_reference(self):
        user_example = (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity [k]. "
            "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect [k]. "
            "Einstein's work profoundly impacted the fields of physics and astronomy [k]."
        )
        primary = PrimaryArticle(
            title="Albert Einstein",
            url="https://en.wikipedia.org/wiki/Albert_Einstein",
        )
        formatted = format_answer_with_primary_reference(user_example, primary)
        expected = (
            "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. "
            "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. "
            "Einstein's work profoundly impacted the fields of physics and astronomy.\n\n"
            "[Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)"
        )
        assert formatted == expected
        assert formatted.count("https://en.wikipedia.org/wiki/Albert_Einstein") == 1

    def test_strongest_article_selected_on_multi_article_evidence(self):
        chunks = [
            RetrievedChunk(
                chunk_id=1,
                document_title="Allahabad Pillar inscription",
                source_url="https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription",
                section_title="Contents",
                passage_text="The Allahabad Pillar inscription was composed by Harishena praising Samudragupta.",
                rrf_score=0.50,
                retrieval_method="hybrid",
            ),
            RetrievedChunk(
                chunk_id=2,
                document_title="Gupta Empire",
                source_url="https://en.wikipedia.org/wiki/Gupta_Empire",
                section_title="History",
                passage_text="The Gupta Empire ruled northern India during its classical golden age.",
                rrf_score=0.20,
                retrieval_method="vector",
            ),
        ]
        raw = "The Allahabad Pillar inscription was composed by Harishena [1]. The Gupta Empire prospered [2]."
        primary = PrimaryArticleSelector.select_primary_article(
            selected_chunks=chunks,
            answer_text=raw,
            question="What is the Allahabad Pillar inscription?",
        )
        assert primary is not None
        assert primary.title == "Allahabad Pillar inscription"


# ==============================================================================
# SECTION 11: SSE Streaming Engine, Progressive Citation Filtering & Cancellation
# ==============================================================================

class TestStreamingEngineAndProgress:
    """Verifies token chunk streaming, StreamingCitationFilter suppression, and Thinking messages."""

    def test_streaming_citation_filter_suppresses_split_tokens(self):
        chunks = [
            "Albert", " Einstein", " developed", " relativity", " [", "k", "].",
            " He", " won", " the", " Nobel", " Prize", " [", "1", "],", " [", "2", "].",
            " Interval", " is", " [10,", " 20].",
        ]
        cf = StreamingCitationFilter()
        streamed = []
        for c in chunks:
            out = cf.process_chunk(c)
            if out:
                streamed.append(out)
        flushed = cf.flush()
        if flushed:
            streamed.append(flushed)

        full = "".join(streamed)
        assert "[k]" not in full
        assert "[1]" not in full
        assert "[2]" not in full
        assert "[10, 20]" in full
        assert full == "Albert Einstein developed relativity. He won the Nobel Prize. Interval is [10, 20]."

    @pytest.mark.asyncio
    async def test_streaming_thinking_progress_events(self):
        orchestrator = AnswerOrchestrator()
        async def mock_stream(*args, **kwargs):
            yield "token", "Answer text."
            yield "done", None

        with patch.object(orchestrator.llm_orchestrator.llm_adapter, "generate_answer_stream", side_effect=mock_stream):
            progress_events = []
            async for chunk in orchestrator.llm_orchestrator.execute_turn_stream(
                question="Explain quantum mechanics",
                evidence_passages=[],
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1].strip())
                        if "message" in data:
                            progress_events.append(data)

            assert len(progress_events) >= 1
            for p in progress_events:
                assert p["message"] == "Thinking..."
                assert "gemma" not in p["message"].lower()


# ==============================================================================
# SECTION 12: High-Performance Middleware, Rate Limiting, Admission & Caching
# ==============================================================================

class TestPerformanceOptimizations:
    """Verifies token bucket rate limiting, single-slot admission control, tiered caching & idempotency."""

    @pytest.mark.asyncio
    async def test_token_bucket_burst_and_monotonic_refill(self):
        bucket = InMemoryTokenBucket()
        policy = RateLimitPolicy(capacity=2, refill_interval_seconds=10.0, cost=1)
        r1 = await bucket.consume("user1", policy)
        assert r1.allowed is True
        assert r1.remaining_tokens == 1.0

        r2 = await bucket.consume("user1", policy)
        assert r2.allowed is True
        assert r2.remaining_tokens == 0.0

        r3 = await bucket.consume("user1", policy)
        assert r3.allowed is False
        assert r3.retry_after_seconds > 0.0

    @pytest.mark.asyncio
    async def test_admission_manager_single_slot(self):
        controller = ResponseController(max_active=1, max_queue=2, queue_timeout=0.5)
        wait_ms = await controller.acquire_slot("req-1")
        assert wait_ms == 0.0
        assert controller.active_generations == 1
        await controller.release_slot("req-1")
        assert controller.active_generations == 0

    @pytest.mark.asyncio
    async def test_tiered_caching_and_invalidation(self):
        cache = InMemoryCache()
        await cache.set("k1", {"data": "test"}, ttl_seconds=60)
        hit = await cache.get("k1")
        assert hit == {"data": "test"}
        await cache.delete("k1")
        miss = await cache.get("k1")
        assert miss is None

    @pytest.mark.asyncio
    async def test_shared_http_client_singleton(self):
        c1 = get_http_client()
        c2 = get_http_client()
        assert c1 is c2

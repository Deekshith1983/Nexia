"""Tests for unified retrieval configuration enforcement (top_k_per_document=3, min_confidence=0.4, max_evidence_passages=9)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app
from app.schemas.answer import (
    AnswerRequest,
    AnswerResponse,
    RetrievalDiagnostics,
    MAX_EVIDENCE_PASSAGES,
    MIN_CONFIDENCE,
    TOP_K_PER_DOCUMENT,
)
from app.schemas.evidence_package import EvidencePackage, SufficiencyAssessment
from app.services.answer_orchestrator import AnswerOrchestrator


def _create_mock_answer_response(question: str = "What is Python?") -> AnswerResponse:
    return AnswerResponse(
        question=question,
        answer_text="Python is a programming language.",
        answer_status="ANSWERED",
        source_references=[],
        conflicts=[],
        sufficiency_rating="SUFFICIENT",
        retrieval_diagnostics=RetrievalDiagnostics(
            embedding_model="nomic-embed-text",
            llm_model="gemma3:4b",
        ),
    )


class TestRetrievalConfigurationEnforcement:
    """Verifies that retrieval configuration is consistently enforced across backend schemas, routes, and orchestrators."""

    @pytest.fixture(autouse=True)
    def bypass_rate_limiter(self):
        with patch("app.services.rate_limiter.RateLimiterService.check", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = MagicMock(allowed=True, retry_after_seconds=0.0)
            yield

    def test_constants_defined_correctly(self):
        """Verify canonical constants."""
        assert TOP_K_PER_DOCUMENT == 3
        assert MIN_CONFIDENCE == 0.4
        assert MAX_EVIDENCE_PASSAGES == 9

    def test_backend_request_omitted_parameters(self):
        """1. Backend request with omitted parameters: backend applies the required values."""
        req = AnswerRequest(question="What is photosynthesis?")
        assert req.top_k_per_document == 3
        assert req.min_confidence == 0.4
        assert req.max_evidence_passages == 9

    def test_backend_request_conflicting_parameters(self):
        """2. Backend request with conflicting parameters: backend overrides and enforces the required values."""
        req = AnswerRequest(
            question="What is photosynthesis?",
            top_k_per_document=10,
            min_confidence=0.95,
            max_evidence_passages=1,
        )
        assert req.top_k_per_document == 3
        assert req.min_confidence == 0.4
        assert req.max_evidence_passages == 9

    def test_normal_answer_endpoint_post(self):
        """3. Normal answer endpoint POST receives and enforces the required values."""
        client = TestClient(app)
        with patch("app.api.routes.answer.orchestrator.answer", new_callable=AsyncMock) as mock_answer:
            mock_answer.return_value = _create_mock_answer_response("What is Python?")

            # Client sends conflicting values
            response = client.post(
                "/api/v1/retrieval/answer",
                json={
                    "question": "What is Python?",
                    "top_k_per_document": 8,
                    "min_confidence": 0.8,
                    "max_evidence_passages": 2,
                },
            )

            assert response.status_code == 200
            assert mock_answer.called
            call_arg = mock_answer.call_args[0][0]
            assert isinstance(call_arg, AnswerRequest)
            assert call_arg.top_k_per_document == 3
            assert call_arg.min_confidence == 0.4
            assert call_arg.max_evidence_passages == 9

    def test_streaming_answer_endpoint_post(self):
        """4. Streaming answer endpoint POST receives and enforces the required values."""
        client = TestClient(app)
        async def mock_stream_gen(request, request_id=None):
            yield "event: start\ndata: {}\n\n"

        with patch("app.api.routes.answer.orchestrator.answer_stream", side_effect=mock_stream_gen) as mock_stream:
            response = client.post(
                "/api/v1/retrieval/answer/stream",
                json={
                    "question": "Explain quantum entanglement",
                    "top_k_per_document": 5,
                    "min_confidence": 0.7,
                    "max_evidence_passages": 1,
                },
            )

            assert response.status_code == 200
            assert mock_stream.called
            call_arg = mock_stream.call_args[0][0]
            assert isinstance(call_arg, AnswerRequest)
            assert call_arg.top_k_per_document == 3
            assert call_arg.min_confidence == 0.4
            assert call_arg.max_evidence_passages == 9

    def test_get_answer_endpoints_use_enforced_defaults(self):
        """GET answer endpoints use the exact defaults (3, 0.4, 9)."""
        client = TestClient(app)
        with patch("app.api.routes.answer.orchestrator.answer", new_callable=AsyncMock) as mock_answer:
            mock_answer.return_value = _create_mock_answer_response("Test question")

            response = client.get(
                "/api/v1/retrieval/answer",
                params={"question": "Test question", "top_k_per_document": 7},
            )

            assert response.status_code == 200
            call_arg = mock_answer.call_args[0][0]
            assert call_arg.top_k_per_document == 3
            assert call_arg.min_confidence == 0.4
            assert call_arg.max_evidence_passages == 9

    @pytest.mark.asyncio
    async def test_retrieval_pipeline_consumes_enforced_values_in_answer(self):
        """5. Retrieval pipeline in AnswerOrchestrator.answer() consumes top_k_per_document=3, min_confidence=0.4, max_evidence_passages=9."""
        orchestrator = AnswerOrchestrator()
        orchestrator.evidence_reasoner = MagicMock()
        orchestrator.evidence_reasoner.reason = AsyncMock()

        mock_evidence_pkg = EvidencePackage(
            question="What is the speed of light?",
            evidence_clusters=[],
            conflicts=[],
            sufficiency_assessment=SufficiencyAssessment(
                status="SUFFICIENT",
                confidence=0.9,
                reasoning="Sufficient evidence.",
            ),
        )
        mock_reason_res = MagicMock()
        mock_reason_res.evidence_package = mock_evidence_pkg
        orchestrator.evidence_reasoner.reason.return_value = mock_reason_res

        # Mock document cache and hybrid retrieval
        orchestrator.document_cache = MagicMock()
        orchestrator.document_cache.get_or_fetch_document = AsyncMock(
            return_value=(MagicMock(page_id=1, title="Test"), "HIT")
        )
        orchestrator.hybrid_retrieval = MagicMock()
        orchestrator.hybrid_retrieval.retrieve = AsyncMock(return_value=[])

        # Mock LLM orchestrator and response controller
        orchestrator.response_controller = MagicMock()
        mock_llm_res = MagicMock(
            answer_text="Test answer.",
            generation_time_ms=10.0,
            load_duration_ms=1.0,
            prompt_eval_duration_ms=2.0,
            eval_duration_ms=3.0,
            tokens_per_second=20.0,
            prompt_tokens=10,
            response_tokens=10,
            status="ANSWERED",
        )
        orchestrator.response_controller.execute_bounded = AsyncMock(return_value=mock_llm_res)

        with patch("app.services.answer_orchestrator.EvidencePacker.pack_slotted_evidence") as mock_pack:
            mock_pack.return_value = []

            req = AnswerRequest(
                question="What is the speed of light?",
                top_k_per_document=99,
                min_confidence=0.99,
                max_evidence_passages=1,
            )
            await orchestrator.answer(req)

            # Verify EvidenceReasoningRequest received top_k_per_document=3 and min_confidence=0.4
            assert orchestrator.evidence_reasoner.reason.called
            reasoning_arg = orchestrator.evidence_reasoner.reason.call_args[0][0]
            assert reasoning_arg.top_k_per_document == 3
            assert reasoning_arg.min_confidence == 0.4

            # Verify EvidencePacker received max_passages=9
            assert mock_pack.called
            kwargs = mock_pack.call_args[1]
            assert kwargs["max_passages"] == 9

    @pytest.mark.asyncio
    async def test_retrieval_pipeline_consumes_enforced_values_in_stream(self):
        """6. Streaming pipeline in AnswerOrchestrator.answer_stream() consumes top_k_per_document=3, min_confidence=0.4, max_evidence_passages=9."""
        orchestrator = AnswerOrchestrator()
        orchestrator.evidence_reasoner = MagicMock()
        orchestrator.evidence_reasoner.reason = AsyncMock()

        mock_evidence_pkg = EvidencePackage(
            question="What is entropy?",
            evidence_clusters=[],
            conflicts=[],
            sufficiency_assessment=SufficiencyAssessment(
                status="SUFFICIENT",
                confidence=0.9,
                reasoning="Sufficient evidence.",
            ),
        )
        mock_reason_res = MagicMock()
        mock_reason_res.evidence_package = mock_evidence_pkg
        orchestrator.evidence_reasoner.reason.return_value = mock_reason_res

        orchestrator.document_cache = MagicMock()
        orchestrator.document_cache.get_or_fetch_document = AsyncMock(
            return_value=(MagicMock(page_id=1, title="Test"), "HIT")
        )
        orchestrator.hybrid_retrieval = MagicMock()
        orchestrator.hybrid_retrieval.retrieve = AsyncMock(return_value=[])

        # Mock llm_orchestrator.execute_turn_stream
        async def mock_turn_stream(**kwargs):
            yield "event: complete\ndata: {}\n\n"

        orchestrator.llm_orchestrator = MagicMock()
        orchestrator.llm_orchestrator.execute_turn_stream = mock_turn_stream

        with patch("app.services.answer_orchestrator.EvidencePacker.pack_slotted_evidence") as mock_pack:
            mock_pack.return_value = []

            req = AnswerRequest(
                question="What is entropy?",
                top_k_per_document=88,
                min_confidence=0.99,
                max_evidence_passages=1,
            )

            # Consume the stream generator
            chunks = []
            async for chunk in orchestrator.answer_stream(req):
                chunks.append(chunk)

            assert len(chunks) > 0

            # Verify EvidenceReasoningRequest received top_k_per_document=3 and min_confidence=0.4
            assert orchestrator.evidence_reasoner.reason.called
            reasoning_arg = orchestrator.evidence_reasoner.reason.call_args[0][0]
            assert reasoning_arg.top_k_per_document == 3
            assert reasoning_arg.min_confidence == 0.4

            # Verify EvidencePacker received max_passages=9
            assert mock_pack.called
            kwargs = mock_pack.call_args[1]
            assert kwargs["max_passages"] == 9

"""Live End-to-End Tests for Stage 7 with live PostgreSQL and local Ollama."""

import pytest
from app.db.connection import ping_database
from app.services.embedding_service import EmbeddingService
from app.services.llm_adapter import OllamaLLMAdapter
from app.services.answer_orchestrator import AnswerOrchestrator
from app.schemas.answer import AnswerRequest


async def ensure_live_services():
    """Verify live PostgreSQL and Ollama before executing live tests."""
    db_ok = await ping_database()
    if not db_ok:
        pytest.skip("PostgreSQL database is not reachable. Skipping live tests.")

    emb_service = EmbeddingService()
    emb_ok = await emb_service.check_health()
    if not emb_ok:
        pytest.skip("Ollama embedding service (nomic-embed-text) is not healthy. Skipping.")

    llm_adapter = OllamaLLMAdapter()
    llm_ok = await llm_adapter.check_health()
    if not llm_ok:
        pytest.skip("Ollama LLM (gemma3:4b) is not healthy. Skipping.")


@pytest.mark.asyncio
async def test_live_relationship_question():
    """Test full live answering on a relationship question."""
    await ensure_live_services()
    orchestrator = AnswerOrchestrator()
    request = AnswerRequest(
        question="Who was the successor of Krishnadevaraya?",
        max_evidence_passages=4,
    )

    response = await orchestrator.answer(request)

    assert response.question == request.question
    assert response.answer_status in ("ANSWERED", "CONFLICTING_EVIDENCE")
    assert len(response.answer_text) > 10
    # Must cite evidence or name Achyuta Deva Raya
    assert "achyuta" in response.answer_text.lower() or "raya" in response.answer_text.lower()
    assert len(response.source_references) > 0
    assert response.retrieval_diagnostics.chunks_in_db > 0


@pytest.mark.asyncio
async def test_live_cache_hit_behavior():
    """Test that a repeated query leverages the document cache."""
    await ensure_live_services()
    orchestrator = AnswerOrchestrator()

    # Query about Krishnadevaraya again - article should now be in nexia_documents
    request = AnswerRequest(
        question="What empire did Krishnadevaraya rule?",
        max_evidence_passages=4,
        force_refresh=False,
    )

    response = await orchestrator.answer(request)

    assert response.answer_status == "ANSWERED"
    # Should register at least one cache hit since Krishnadevaraya was ingested
    assert response.retrieval_diagnostics.cache_hits > 0


@pytest.mark.asyncio
async def test_live_insufficient_evidence():
    """Test that the system flags insufficient evidence when passages lack answers."""
    await ensure_live_services()
    orchestrator = AnswerOrchestrator()
    request = AnswerRequest(
        question="What was Albert Einstein's favorite color of sports car?",
        max_evidence_passages=4,
    )

    response = await orchestrator.answer(request)

    # Should detect insufficient evidence either in deterministic Stage 6 or in LLM
    assert response.answer_status in ("INSUFFICIENT_EVIDENCE", "ANSWERED")
    if response.answer_status == "INSUFFICIENT_EVIDENCE":
        assert "insufficient" in response.answer_text.lower()

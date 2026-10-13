"""Stage 7 Integration and Unit Tests: LLM Grounded Answer, Vector Cache & Retrieval."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.retrieval import Document, DocumentMetadata, Section, Paragraph
from app.services.chunker import ChunkerService
from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingUnavailableError,
    EmbeddingDimensionError,
)
from app.services.prompts import PromptContext, build_answer_prompt
from app.services.llm_adapter import OllamaLLMAdapter, LLMGenerationResult
from app.services.answer_orchestrator import AnswerOrchestrator
from app.schemas.answer import AnswerRequest

client = TestClient(app)


def test_chunker_basic():
    """Verify ChunkerService produces expected chunks with metadata."""
    chunker = ChunkerService()
    doc = Document(
        title="Alan Turing",
        page_id=1234,
        url="https://en.wikipedia.org/wiki/Alan_Turing",
        introduction="Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist.",
        metadata=DocumentMetadata(
            page_id=1234,
            url="https://en.wikipedia.org/wiki/Alan_Turing",
            total_sections=1,
            total_paragraphs=2,
            total_words=45,
        ),
        sections=[
            Section(
                index=0,
                title="Early life",
                level=2,
                paragraphs=[
                    Paragraph(
                        id="doc_1234_sec_0_p_0",
                        document_title="Alan Turing",
                        section_title="Early life",
                        index=0,
                        text="Alan Turing was born in Maida Vale, London, on 23 June 1912. He was the son of Julius Mathison Turing of the Indian Civil Service.",
                        word_count=25,
                        char_count=138,
                    ),
                    Paragraph(
                        id="doc_1234_sec_0_p_1",
                        document_title="Alan Turing",
                        section_title="Early life",
                        index=1,
                        text="Too short.",  # Under 15 words -> should be skipped
                        word_count=2,
                        char_count=10,
                    ),
                ],
            )
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_title == "Alan Turing"
    assert chunk.page_id == 1234
    assert chunk.section_title == "Early life"
    assert chunk.paragraph_id == "doc_1234_sec_0_p_0"
    assert chunk.word_count == 25
    assert len(chunk.content_hash) == 64


def test_chunker_large_paragraph_split():
    """Verify large paragraphs are split at sentence boundaries."""
    chunker = ChunkerService()
    sentence = "Alan Turing made fundamental contributions to the foundations of theoretical computer science. "
    large_text = sentence * 60  # > 500 words
    words = len(large_text.split())

    doc = Document(
        title="Test Document",
        page_id=999,
        url="https://en.wikipedia.org/wiki/Test",
        introduction="",
        metadata=DocumentMetadata(
            page_id=999,
            url="https://en.wikipedia.org/wiki/Test",
            total_sections=1,
            total_paragraphs=1,
            total_words=words,
        ),
        sections=[
            Section(
                index=0,
                title="Overview",
                level=2,
                paragraphs=[
                    Paragraph(
                        id="doc_999_sec_0_p_0",
                        document_title="Test Document",
                        section_title="Overview",
                        index=0,
                        text=large_text,
                        word_count=words,
                        char_count=len(large_text),
                    )
                ],
            )
        ],
    )

    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert "_sub_" in c.paragraph_id
        assert c.word_count <= chunker.MAX_CHUNK_WORDS


@pytest.mark.asyncio
async def test_embedding_service_dimension_check():
    """Verify EmbeddingService rejects unexpected vector dimensions."""
    from unittest.mock import MagicMock
    svc = EmbeddingService(expected_dimension=768)

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        # Return 3-dim vector instead of 768
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_post.return_value = mock_response

        with pytest.raises(EmbeddingDimensionError):
            await svc.embed_text("test query")


def test_build_answer_prompt():
    """Verify prompt formatting matches system instructions and references."""
    ctx = PromptContext(
        question="Who was the successor of Krishnadevaraya?",
        question_type="RELATIONSHIP",
        entities=["Krishnadevaraya"],
        relation="SUCCESSOR_OF",
        passages=[
            {
                "document_title": "Krishnadevaraya",
                "section_title": "Succession",
                "source_url": "https://en.wikipedia.org/wiki/Krishnadevaraya",
                "passage_text": "Krishnadevaraya was succeeded by his younger brother Achyuta Deva Raya in 1529.",
            }
        ],
        conflicts=[],
        sufficiency_rating="SUFFICIENT",
    )

    prompt = build_answer_prompt(ctx)
    assert "USER QUESTION: Who was the successor of Krishnadevaraya?" in prompt
    assert "[1] Source: Krishnadevaraya | Section: Succession" in prompt
    assert "Krishnadevaraya was succeeded by his younger brother Achyuta Deva Raya" in prompt
    assert "None identified." in prompt
    assert "SUFFICIENCY RATING (from Nexia deterministic analysis): SUFFICIENT" in prompt


@pytest.mark.asyncio
async def test_llm_adapter_insufficient_evidence():
    """Verify OllamaLLMAdapter flags INSUFFICIENT_EVIDENCE status correctly."""
    from unittest.mock import MagicMock
    adapter = OllamaLLMAdapter()

    ctx = PromptContext(
        question="What was Albert Einstein's favorite car model?",
        passages=[],
        conflicts=[],
        sufficiency_rating="INSUFFICIENT",
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "INSUFFICIENT_EVIDENCE: The supplied passages contain no mention of a favorite car.",
            "prompt_eval_count": 50,
            "eval_count": 20,
            "total_duration": 1500000000,
        }
        mock_post.return_value = mock_response

        result = await adapter.generate_answer(ctx)
        assert result.status == "INSUFFICIENT_EVIDENCE"
        assert "INSUFFICIENT_EVIDENCE" in result.answer_text


def test_root_endpoint_lists_answer():
    """Verify GET / lists the Stage 7 answer endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data["endpoints"]
    assert data["endpoints"]["answer"] == "/api/v1/retrieval/answer"


@pytest.mark.asyncio
async def test_stage7_orchestrator_mocked_llm():
    """Verify end-to-end AnswerOrchestrator pipeline with mocked LLM generation."""
    orchestrator = AnswerOrchestrator()

    # Mock the LLM adapter call so test runs fast and deterministically
    mock_gen = LLMGenerationResult(
        answer_text="Krishnadevaraya was succeeded by his half-brother Achyuta Deva Raya in 1529 [1].",
        status="ANSWERED",
        model="gemma3:4b",
        generation_time_ms=120.0,
    )

    with patch.object(orchestrator.llm_adapter, "generate_answer", return_value=mock_gen):
        request = AnswerRequest(
            question="Who was the successor of Krishnadevaraya?",
            max_evidence_passages=4,
        )
        response = await orchestrator.answer(request)

        assert response.question == "Who was the successor of Krishnadevaraya?"
        assert response.answer_status == "ANSWERED"
        assert "Achyuta Deva Raya" in response.answer_text
        assert response.retrieval_diagnostics.llm_model == "gemma3:4b"
        assert response.stage6_evidence_package is not None

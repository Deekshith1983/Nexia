"""Comprehensive Unit & Regression Tests for Single Primary Article Citation Selection & Formatting."""

import pytest
from unittest.mock import AsyncMock, patch

from app.schemas.answer import AnswerRequest, SourceReference
from app.services.citation_manager import (
    PrimaryArticle,
    PrimaryArticleSelector,
    format_answer_with_primary_reference,
    strip_inline_citations,
)
from app.services.answer_orchestrator import AnswerOrchestrator, ClaimVerifier
from app.services.hybrid_retrieval import RetrievedChunk
from app.services.llm_adapter import LLMGenerationResult


# =====================================================================
# Case A: Single Article
# =====================================================================
def test_case_a_single_article():
    """Verify single supporting article generates a clean trailing hyperlink."""
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Samudragupta",
            source_url="https://en.wikipedia.org/wiki/Samudragupta",
            section_title="Military campaigns",
            passage_text="Samudragupta expanded the Gupta Empire through extensive military conquests across India.",
            rrf_score=0.45,
            retrieval_method="hybrid",
        )
    ]
    raw_answer = "Samudragupta expanded his empire through extensive military conquests [1]."
    question = "How did Samudragupta expand his empire?"

    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=raw_answer,
        question=question,
    )
    assert primary is not None
    assert primary.title == "Samudragupta"
    assert primary.url == "https://en.wikipedia.org/wiki/Samudragupta"
    assert primary.supported_claims == 1

    formatted = format_answer_with_primary_reference(raw_answer, primary)
    assert "[1]" not in formatted
    assert formatted.endswith("[Samudragupta](https://en.wikipedia.org/wiki/Samudragupta)")
    assert formatted == "Samudragupta expanded his empire through extensive military conquests.\n\n[Samudragupta](https://en.wikipedia.org/wiki/Samudragupta)"


# =====================================================================
# Case B: Multiple Articles with Strongest Verified Coverage
# =====================================================================
def test_case_b_multiple_articles_strongest_selected():
    """Verify that among multiple articles, the one with strongest verified claim coverage is chosen."""
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
            document_title="Allahabad Pillar inscription",
            source_url="https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription",
            section_title="Political geography",
            passage_text="It describes Samudragupta's treatment of frontier kings and reinstated southern rulers.",
            rrf_score=0.48,
            retrieval_method="hybrid",
        ),
        RetrievedChunk(
            chunk_id=3,
            document_title="Gupta Empire",
            source_url="https://en.wikipedia.org/wiki/Gupta_Empire",
            section_title="History",
            passage_text="The Gupta Empire ruled northern India during its classical golden age.",
            rrf_score=0.20,
            retrieval_method="vector",
        ),
    ]
    raw_answer = (
        "The Allahabad Pillar inscription was composed by the court poet Harishena [1]. "
        "It details Samudragupta's treatment of frontier kings and southern rulers [2]. "
        "The Gupta Empire experienced a classical golden age [3]."
    )
    question = "What does the Allahabad Pillar inscription describe?"

    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=raw_answer,
        question=question,
    )
    assert primary is not None
    # Allahabad Pillar inscription supports 2 out of 3 claims with high RRF and query match
    assert primary.title == "Allahabad Pillar inscription"
    assert primary.url == "https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription"

    formatted = format_answer_with_primary_reference(raw_answer, primary)
    assert "[1]" not in formatted
    assert "[2]" not in formatted
    assert "[3]" not in formatted
    assert "[Gupta Empire]" not in formatted
    assert formatted.endswith("[Allahabad Pillar inscription](https://en.wikipedia.org/wiki/Allahabad_Pillar_inscription)")


# =====================================================================
# Case C: Repeated Passages from Same Article
# =====================================================================
def test_case_c_repeated_passages_deduplicated():
    """Verify that multiple chunks from the same article do not produce duplicate links."""
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Shivaji",
            source_url="https://en.wikipedia.org/wiki/Shivaji",
            section_title="Early life",
            passage_text="Shivaji was born in the hill-fort of Shivneri.",
            rrf_score=0.40,
            retrieval_method="hybrid",
        ),
        RetrievedChunk(
            chunk_id=2,
            document_title="Shivaji",
            source_url="https://en.wikipedia.org/wiki/Shivaji",
            section_title="Coronation",
            passage_text="Shivaji was formally crowned Chhatrapati at Raigad in 1674.",
            rrf_score=0.42,
            retrieval_method="hybrid",
        ),
        RetrievedChunk(
            chunk_id=3,
            document_title="Shivaji",
            source_url="https://en.wikipedia.org/wiki/Shivaji",
            section_title="Military",
            passage_text="He pioneered guerrilla warfare tactics known as ganimi kava.",
            rrf_score=0.45,
            retrieval_method="hybrid",
        ),
    ]
    raw_answer = (
        "Shivaji was born at Shivneri [1]. "
        "He was crowned Chhatrapati at Raigad in 1674 [2]. "
        "He utilized guerrilla warfare methods [3]."
    )
    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=raw_answer,
        question="Who was Shivaji?",
    )
    assert primary is not None
    assert primary.title == "Shivaji"

    formatted = format_answer_with_primary_reference(raw_answer, primary)
    # Ensure the link appears exactly once
    assert formatted.count("[Shivaji](https://en.wikipedia.org/wiki/Shivaji)") == 1
    assert "[1]" not in formatted
    assert "[2]" not in formatted
    assert "[3]" not in formatted


# =====================================================================
# Case D: Weak Primary Coverage Fallback
# =====================================================================
def test_case_d_weak_coverage_fallback():
    """Verify that if no single article provides sufficient coverage, no misleading link is added."""
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Unrelated Topic A",
            source_url="https://en.wikipedia.org/wiki/Unrelated_Topic_A",
            section_title="Section 1",
            passage_text="Random text that has nothing to do with quantum physics.",
            rrf_score=0.05,
            retrieval_method="keyword",
        ),
    ]
    # Answer text has zero verified support in chunks
    raw_answer = "Quantum entanglement occurs when pairs of particles interact in ways such that their physical states become linked."
    question = "Explain quantum entanglement in computing."

    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=raw_answer,
        question=question,
        coverage_threshold=0.35,
    )
    # Should safely abstain from attaching an unrelated article link
    assert primary is None

    formatted = format_answer_with_primary_reference(raw_answer, primary)
    assert formatted == raw_answer
    assert "https://" not in formatted


# =====================================================================
# Case E: Unsupported Claim Verification Preserved
# =====================================================================
def test_case_e_claim_verifier_operates_normally():
    """Verify ClaimVerifier continues pruning ungrounded sentences before formatting."""
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
    # Sentence 2 mentions ungrounded bird observatory off-topic content
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

    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=verified_answer,
        question="Where was the Antikythera mechanism found?",
    )
    formatted = format_answer_with_primary_reference(verified_answer, primary)
    assert "[1]" not in formatted
    assert formatted.endswith("[Antikythera mechanism](https://en.wikipedia.org/wiki/Antikythera_mechanism)")


# =====================================================================
# Case F: Long Answer with Multiple Paragraphs
# =====================================================================
def test_case_f_long_answer_multiple_paragraphs():
    """Verify that multi-paragraph answers place the hyperlink only at the very end of the final paragraph."""
    chunks = [
        RetrievedChunk(
            chunk_id=1,
            document_title="Aryabhata",
            source_url="https://en.wikipedia.org/wiki/Aryabhata",
            section_title="Astronomy",
            passage_text="Aryabhata proposed that the earth rotates on its axis daily.",
            rrf_score=0.48,
            retrieval_method="hybrid",
        ),
        RetrievedChunk(
            chunk_id=2,
            document_title="Aryabhata",
            source_url="https://en.wikipedia.org/wiki/Aryabhata",
            section_title="Mathematics",
            passage_text="He approximated the value of pi to 3.1416.",
            rrf_score=0.46,
            retrieval_method="hybrid",
        ),
    ]
    multi_para_answer = (
        "Aryabhata was an ancient Indian mathematician and astronomer [1].\n\n"
        "In astronomy, he proposed that the earth rotates on its axis daily [1].\n\n"
        "In mathematics, he provided an accurate approximation of pi [2]."
    )
    primary = PrimaryArticleSelector.select_primary_article(
        selected_chunks=chunks,
        answer_text=multi_para_answer,
        question="Who was Aryabhata and what did he discover?",
    )
    formatted = format_answer_with_primary_reference(multi_para_answer, primary)

    paragraphs = formatted.split("\n\n")
    assert len(paragraphs) == 4
    # Check paragraph 1 does NOT have the link
    assert "https://" not in paragraphs[0]
    assert "[1]" not in paragraphs[0]
    # Check paragraph 2 does NOT have the link
    assert "https://" not in paragraphs[1]
    assert "[1]" not in paragraphs[1]
    # Check paragraph 3 does NOT have the link
    assert "https://" not in paragraphs[2]
    assert "[2]" not in paragraphs[2]
    # Check paragraph 4 is the final article link
    assert paragraphs[3] == "[Aryabhata](https://en.wikipedia.org/wiki/Aryabhata)"


# =====================================================================
# Case G: Inline Citation Stripping Utility
# =====================================================================
def test_strip_inline_citations():
    """Verify inline citation removal removes [k], [1], [source], etc. while preserving math intervals, editorial quotes, and markdown links."""
    text1 = "In 1529 [1], Krishnadevaraya died [2]. He ruled for 20 years [1]."
    assert strip_inline_citations(text1) == "In 1529, Krishnadevaraya died. He ruled for 20 years."

    # Mathematical intervals must be preserved
    text2 = "Values range between [10, 20] in standard notation."
    assert strip_inline_citations(text2) == "Values range between [10, 20] in standard notation."

    # Editorial notes in quotes must be preserved
    text_editorial1 = "He wrote [sic] in his notebook."
    assert strip_inline_citations(text_editorial1) == "He wrote [sic] in his notebook."

    text_editorial2 = "The biographer noted that [she] was exceptionally gifted."
    assert strip_inline_citations(text_editorial2) == "The biographer noted that [she] was exceptionally gifted."

    # Markdown links must be preserved
    text_link = "More details at [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)."
    assert strip_inline_citations(text_link) == "More details at [Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)."


def test_strip_inline_citations_k_and_internal_markers():
    """Verify [k], [1], [2], [source], [passage 1] and chained markers are cleanly removed."""
    # The exact user-provided problem statement
    user_example = (
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity [k]. "
        "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect [k]. "
        "Einstein's work profoundly impacted the fields of physics and astronomy [k]."
    )
    expected_clean = (
        "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity. "
        "He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. "
        "Einstein's work profoundly impacted the fields of physics and astronomy."
    )
    assert strip_inline_citations(user_example) == expected_clean

    # Internal markers: [source], [passage 1], [doc 2], [evidence 1], [1, 2], [1], [2]
    assert strip_inline_citations("Relativity was confirmed [source].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [passage 1].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [doc 2].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [evidence 1].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [1], [2].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [1, 2].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed [k][1].") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed. [k]") == "Relativity was confirmed."
    assert strip_inline_citations("Relativity was confirmed.[k]") == "Relativity was confirmed."

    # List items with [k]
    list_text = "1. General relativity [k]\n2. Special relativity [k]"
    assert strip_inline_citations(list_text) == "1. General relativity\n2. Special relativity"


def test_format_answer_with_primary_reference_user_example():
    """Verify single final Wikipedia article citation appended to clean text with no [k]."""
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


def test_streaming_citation_filter():
    """Verify StreamingCitationFilter suppresses [k], [1], [source] across chunk boundaries."""
    from app.services.citation_manager import StreamingCitationFilter

    # Simulation where [k]. is split into 3 chunks: [' [', 'k', '].']
    chunks = [
        "Albert", " Einstein", " developed", " relativity", " [", "k", "].",
        " He", " received", " the", " Nobel", " Prize", " [", "1", "],", " [", "2", "].",
        " Values", " in", " [10,", " 20].",
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

    full_text = "".join(streamed)
    assert "[k]" not in full_text
    assert "[1]" not in full_text
    assert "[2]" not in full_text
    assert "[10, 20]" in full_text
    assert full_text == "Albert Einstein developed relativity. He received the Nobel Prize. Values in [10, 20]."


# =====================================================================
# Case H: AnswerOrchestrator End-to-End Integration
# =====================================================================
@pytest.mark.asyncio
async def test_case_h_orchestrator_integration():
    """Verify AnswerOrchestrator outputs clean answer with single trailing primary article."""
    orchestrator = AnswerOrchestrator()

    mock_gen = LLMGenerationResult(
        answer_text="Krishnadevaraya was succeeded by his half-brother Achyuta Deva Raya in 1529 [1].",
        status="ANSWERED",
        model="gemma3:4b",
        generation_time_ms=100.0,
    )

    with patch.object(orchestrator.llm_adapter, "generate_answer", return_value=mock_gen):
        request = AnswerRequest(
            question="Who was the successor of Krishnadevaraya?",
            max_evidence_passages=3,
        )
        response = await orchestrator.answer(request)

        assert response.answer_status == "ANSWERED"
        assert "[1]" not in response.answer_text
        assert response.answer_text.endswith("[Krishnadevaraya](https://en.wikipedia.org/wiki/Krishnadevaraya)")
        # Internal provenance preserved
        assert len(response.source_references) >= 1
        assert response.source_references[0].document_title == "Krishnadevaraya"
        assert response.retrieval_diagnostics.claims_verified >= 1


# =====================================================================
# Case I: Streaming End-to-End Delivery & Parity
# =====================================================================
@pytest.mark.asyncio
async def test_case_i_streaming_parity():
    """Verify streaming complete event produces identical primary article formatting."""
    import json
    from app.services.llm_orchestrator import LLMOrchestrator

    orchestrator = AnswerOrchestrator()
    question = "Who was the successor of Krishnadevaraya?"
    evidence = [
        {
            "document_title": "Krishnadevaraya",
            "section_title": "Succession",
            "source_url": "https://en.wikipedia.org/wiki/Krishnadevaraya",
            "passage_text": "Krishnadevaraya was succeeded by his half-brother Achyuta Deva Raya in 1529.",
            "rrf_score": 0.48,
        }
    ]

    async def mock_stream_gen(*args, **kwargs):
        tokens = ["Krishnadevaraya ", "was ", "succeeded ", "by ", "his ", "half-brother ", "Achyuta ", "Deva ", "Raya ", "in ", "1529 ", "[1]."]
        for t in tokens:
            yield "token", t
        yield "done", None

    with patch.object(orchestrator.llm_orchestrator.llm_adapter, "generate_answer_stream", side_effect=mock_stream_gen):
        events = []
        async for sse_chunk in orchestrator.llm_orchestrator.execute_turn_stream(
            question=question,
            evidence_passages=evidence,
        ):
            for line in sse_chunk.strip().split("\n"):
                if line.startswith("data:"):
                    events.append(json.loads(line.split(":", 1)[1].strip()))

        complete_event = next(e for e in events if "answer_text" in e)
        assert complete_event["answer_status"] == "ANSWERED"
        assert "[1]" not in complete_event["answer_text"]
        assert complete_event["answer_text"].endswith("[Krishnadevaraya](https://en.wikipedia.org/wiki/Krishnadevaraya)")
        assert len(complete_event["source_references"]) == 1


# =====================================================================
# Case J: Model-Generated Sources/References Section Stripping
# =====================================================================
def test_case_j_sources_section_stripped():
    """Verify that model-generated Sources or References headings and lists are stripped."""
    raw = (
        "The Gupta Empire was an ancient Indian empire [1]. It existed from the mid-to-late 3rd century to 543 CE [2].\n\n"
        "Sources:\n"
        "- https://en.wikipedia.org/wiki/Gupta_Empire\n"
        "- [Gupta Empire](https://en.wikipedia.org/wiki/Gupta_Empire)"
    )
    primary = PrimaryArticle(
        title="Gupta Empire",
        url="https://en.wikipedia.org/wiki/Gupta_Empire",
    )
    formatted = format_answer_with_primary_reference(raw, primary)

    assert "Sources:" not in formatted
    assert "[1]" not in formatted
    assert "[2]" not in formatted
    assert formatted.count("https://en.wikipedia.org/wiki/Gupta_Empire") == 1
    assert formatted.endswith("[Gupta Empire](https://en.wikipedia.org/wiki/Gupta_Empire)")


# =====================================================================
# Case K: Duplicate Final Hyperlink Deduplicated
# =====================================================================
def test_case_k_duplicate_final_hyperlink_deduplicated():
    """Verify that if the model output already ends with a markdown link, no duplicate is created."""
    raw = "The Mauryan Empire was founded by Chandragupta Maurya in 322 BCE.\n\n[Mauryan Empire](https://en.wikipedia.org/wiki/Mauryan_Empire)"
    primary = PrimaryArticle(
        title="Mauryan Empire",
        url="https://en.wikipedia.org/wiki/Mauryan_Empire",
    )
    formatted = format_answer_with_primary_reference(raw, primary)

    assert formatted.count("[Mauryan Empire](https://en.wikipedia.org/wiki/Mauryan_Empire)") == 1
    assert formatted.endswith("[Mauryan Empire](https://en.wikipedia.org/wiki/Mauryan_Empire)")


# =====================================================================
# Case L: Streaming Thinking Progress Events
# =====================================================================
@pytest.mark.asyncio
async def test_case_l_streaming_thinking_progress():
    """Verify that streaming pipeline emits 'Thinking...' in progress events without technical jargon."""
    import json
    orchestrator = AnswerOrchestrator()

    async def mock_stream_gen(*args, **kwargs):
        yield "token", "Answer "
        yield "token", "text."
        yield "done", None

    with patch.object(orchestrator.llm_orchestrator.llm_adapter, "generate_answer_stream", side_effect=mock_stream_gen):
        progress_events = []
        async for chunk in orchestrator.llm_orchestrator.execute_turn_stream(
            question="What is physics?",
            evidence_passages=[],
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if "message" in data:
                        progress_events.append(data)

        assert len(progress_events) >= 2
        for p in progress_events:
            assert p["message"] == "Thinking..."
            assert "gemma" not in p["message"].lower()
            assert "knowledge base" not in p["message"].lower()


# =====================================================================
# Case M: Streaming with Split [k] Tokens Does Not Leak to User
# =====================================================================
@pytest.mark.asyncio
async def test_case_m_streaming_with_k_tokens():
    """Verify streaming suppresses [k] across token chunk boundaries and appends single final article."""
    import json
    orchestrator = AnswerOrchestrator()
    question = "Who was Albert Einstein?"
    evidence = [
        {
            "document_title": "Albert Einstein",
            "section_title": "Early life and career",
            "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
            "passage_text": "Albert Einstein was a German-born theoretical physicist who developed relativity.",
            "rrf_score": 0.55,
        }
    ]

    # Split [k]. across 3 chunks: " [", "k", "]."
    async def mock_stream_gen(*args, **kwargs):
        tokens = [
            "Albert ", "Einstein ", "was ", "a ", "physicist", " [", "k", "].",
            " He ", "developed ", "relativity", " [", "k", "].",
        ]
        for t in tokens:
            yield "token", t
        yield "done", None

    with patch.object(orchestrator.llm_orchestrator.llm_adapter, "generate_answer_stream", side_effect=mock_stream_gen):
        delivered_tokens = []
        complete_events = []
        async for chunk in orchestrator.llm_orchestrator.execute_turn_stream(
            question=question,
            evidence_passages=evidence,
        ):
            for line in chunk.strip().split("\n"):
                if line.startswith("data:"):
                    data = json.loads(line.split(":", 1)[1].strip())
                    if "token" in data:
                        delivered_tokens.append(data["token"])
                    if "answer_text" in data:
                        complete_events.append(data)

        # 1. No delivered token during the live stream ever contained [k]
        combined_stream = "".join(delivered_tokens)
        assert "[k]" not in combined_stream
        assert "Albert Einstein was a physicist. He developed relativity." in combined_stream

        # 2. Complete event has clean text without [k] and exactly one Wikipedia link at the end
        assert len(complete_events) == 1
        final_answer = complete_events[0]["answer_text"]
        assert "[k]" not in final_answer
        assert final_answer.endswith("[Albert Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)")
        assert final_answer.count("https://en.wikipedia.org/wiki/Albert_Einstein") == 1



"""Unit and integration tests for automatic conversation title generation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.connection import get_session_factory
from app.db.models import NexiaConversation, NexiaConversationMessage
from app.main import app
from app.schemas.answer import AnswerRequest
from app.schemas.conversation import MessageRole, MessageStatus
from app.schemas.evidence_package import EvidencePackage, SufficiencyAssessment
from app.services.answer_orchestrator import AnswerOrchestrator
from app.services.conversation_manager import ConversationManager
from app.services.hybrid_retrieval import RetrievedChunk
from app.services.llm_adapter import LLMTimeoutError, LLMUnavailableError, OllamaLLMAdapter
from app.services.llm_orchestrator import LLMOrchestrator
from app.services.title_generator import ConversationTitleGenerator


@pytest.fixture
def mock_llm_adapter():
    """Create a mock OllamaLLMAdapter with generate_text."""
    adapter = MagicMock(spec=OllamaLLMAdapter)
    adapter.model = "gemma3:4b"
    adapter.generate_text = AsyncMock(return_value="The Reign of Shivaji Maharaj")
    return adapter


@pytest.fixture
def title_generator(mock_llm_adapter):
    return ConversationTitleGenerator(llm_adapter=mock_llm_adapter)


@pytest_asyncio.fixture(autouse=True)
async def cleanup_created_conversations():
    created_ids = set()
    original_create = ConversationManager.create_conversation
    original_add = ConversationManager.add_message

    async def tracking_create(self, *args, **kwargs):
        conv = await original_create(self, *args, **kwargs)
        created_ids.add(conv.id)
        return conv

    async def tracking_add(self, conversation_id, *args, **kwargs):
        created_ids.add(conversation_id)
        return await original_add(self, conversation_id, *args, **kwargs)

    with patch.object(ConversationManager, "create_conversation", tracking_create), \
         patch.object(ConversationManager, "add_message", tracking_add):
        yield

    # Teardown: delete all tracked conversations from database
    if created_ids:
        mgr = ConversationManager()
        for cid in list(created_ids):
            try:
                await mgr.delete_conversation(cid)
            except Exception:
                pass

    from app.db.connection import get_engine
    try:
        await get_engine().dispose()
    except Exception:
        pass


class TestTitleGeneratorCleaningAndValidation:
    """Tests for title cleaning, formatting, and truncation rules."""

    def test_strip_prefixes_and_quotes(self):
        generator = ConversationTitleGenerator()
        raw = 'Title: "Shivaji Maharaj and the Maratha Empire"'
        title = generator.clean_and_validate_title(raw, question="Who was Shivaji Maharaj?")
        assert title == "Shivaji Maharaj and Maratha Empire" or len(title.split()) <= 5
        assert not title.startswith("Title:")
        assert '"' not in title
        assert len(title.split()) <= 5

    def test_strip_markdown_and_punctuation(self):
        generator = ConversationTitleGenerator()
        raw = '**Quantum Computing Fundamentals**.'
        title = generator.clean_and_validate_title(raw, question="What is quantum computing?")
        assert title == "Quantum Computing Fundamentals"
        assert "**" not in title
        assert not title.endswith(".")

    def test_acronym_preservation(self):
        generator = ConversationTitleGenerator()
        raw = 'NASA Space Exploration Mission'
        title = generator.clean_and_validate_title(raw, question="Tell me about NASA")
        words = title.split()
        assert "NASA" in words
        assert len(words) <= 5

    def test_clamp_to_five_words(self):
        generator = ConversationTitleGenerator()
        raw = "A Comprehensive Historical Overview of the Entire Roman Empire"
        title = generator.clean_and_validate_title(raw, question="Explain the Roman Empire")
        words = title.split()
        assert len(words) <= 5
        # Ensure trailing dangling preposition was trimmed if applicable
        assert words[-1].lower() not in {"of", "the", "in", "and", "to"}

    def test_refusal_fallback(self):
        generator = ConversationTitleGenerator()
        raw = "I am sorry, as an AI I cannot generate a title."
        title = generator.clean_and_validate_title(raw, question="Who was Alan Turing?")
        words = title.split()
        assert len(words) <= 5
        assert "Alan" in title or "Turing" in title

    def test_empty_output_fallback(self):
        generator = ConversationTitleGenerator()
        raw = "   \n\t  "
        title = generator.clean_and_validate_title(raw, question="What is photosynthesis in biology?")
        words = title.split()
        assert len(words) <= 5
        assert "Photosynthesis" in title or "Biology" in title


class TestTitleGenerationPrompting:
    """Tests verifying prompt construction and LLM invocation."""

    @pytest.mark.asyncio
    async def test_uses_both_question_and_answer(self, title_generator, mock_llm_adapter):
        mock_llm_adapter.generate_text.return_value = "Maratha Empire Origins"

        question = "Who was Shivaji Maharaj?"
        answer = "Shivaji Maharaj was an Indian warrior king who founded the Maratha Empire. [Chhatrapati Shivaji Maharaj](https://en.wikipedia.org/wiki/Shivaji)"

        title = await title_generator.generate_title(question, answer)
        assert title == "Maratha Empire Origins"

        mock_llm_adapter.generate_text.assert_called_once()
        call_kwargs = mock_llm_adapter.generate_text.call_args[1]
        prompt = call_kwargs["prompt"]

        assert "Who was Shivaji Maharaj?" in prompt
        assert "Maratha Empire" in prompt
        # Ensure markdown link url was stripped from summary
        assert "https://en.wikipedia.org" not in prompt

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback(self, title_generator, mock_llm_adapter):
        mock_llm_adapter.generate_text.side_effect = LLMTimeoutError("Request timed out")

        question = "Explain Black Hole Thermodynamics"
        answer = "Black hole thermodynamics describes the laws governing black holes."

        title = await title_generator.generate_title(question, answer)
        words = title.split()
        assert len(words) <= 5
        assert "Black" in title or "Hole" in title or "Thermodynamics" in title


class TestConversationEligibilityAndAtomicUpdate:
    """Tests for eligibility checks and atomic title updates in PostgreSQL."""

    @pytest.mark.asyncio
    async def test_eligibility_lifecycle(self):
        conv_mgr = ConversationManager()
        # 1. Create a new conversation
        conv = await conv_mgr.create_conversation()
        assert conv.title == "New Chat"

        # Not eligible before any messages
        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is False

        # Add user message (1 message total)
        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Who was Ashoka?",
        )
        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is False

        # Add assistant message (2 messages total - turn 1 complete)
        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="Ashoka was an Indian emperor of the Maurya Dynasty.",
        )
        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is True

        # Atomically rename
        success = await conv_mgr.update_title_atomic(conv.id, "Reign of Ashoka")
        assert success is True

        # Fetch and verify title updated
        updated_conv = await conv_mgr.get_conversation(conv.id)
        assert updated_conv.title == "Reign of Ashoka"

        # After title is no longer 'New Chat' or 'New Conversation', should no longer be eligible
        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is False

        # Second update attempt fails (atomic guard)
        second_attempt = await conv_mgr.update_title_atomic(conv.id, "Different Title")
        assert second_attempt is False

        # Add turn 2 (message 3 and 4)
        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Where did he rule?",
        )
        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="He ruled almost all of the Indian subcontinent.",
        )
        # Never eligible on turn 2+
        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is False

    @pytest.mark.asyncio
    async def test_custom_user_title_is_never_overwritten(self):
        conv_mgr = ConversationManager()
        # User creates conversation with a custom title
        conv = await conv_mgr.create_conversation(title="My Custom Research")
        assert conv.title == "My Custom Research"

        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.USER,
            content="Tell me about algorithms.",
        )
        await conv_mgr.add_message(
            conversation_id=conv.id,
            role=MessageRole.ASSISTANT,
            content="An algorithm is a finite sequence of instructions.",
        )

        eligible = await conv_mgr.is_eligible_for_title_generation(conv.id)
        assert eligible is False

        # Atomic update will return False and leave custom title untouched
        success = await conv_mgr.update_title_atomic(conv.id, "Algorithms Guide")
        assert success is False

        fetched = await conv_mgr.get_conversation(conv.id)
        assert fetched.title == "My Custom Research"


class TestStreamingTitleDelivery:
    """Test SSE event stream title emission."""

    @pytest.mark.asyncio
    async def test_streaming_emits_title_event_on_turn_1(self):
        conv_mgr = ConversationManager()
        conv = await conv_mgr.create_conversation()

        mock_llm = MagicMock(spec=OllamaLLMAdapter)
        mock_llm.model = "gemma3:4b"

        # Stream answer tokens
        async def fake_stream(*args, **kwargs):
            yield ("token", "Shivaji ")
            yield ("token", "was a king.")
            yield ("done", MagicMock(
                answer_text="Shivaji was a king.",
                prompt_tokens=10,
                response_tokens=5,
                load_duration_ms=1.0,
                prompt_eval_duration_ms=2.0,
                eval_duration_ms=3.0,
            ))

        mock_llm.generate_answer_stream = fake_stream
        mock_llm.generate_text = AsyncMock(return_value="Shivaji The Warrior King")

        title_gen = ConversationTitleGenerator(llm_adapter=mock_llm)
        orchestrator = LLMOrchestrator(
            conversation_manager=conv_mgr,
            llm_adapter=mock_llm,
            title_generator=title_gen,
        )

        events = []
        async for chunk in orchestrator.execute_turn_stream(
            question="Who was Shivaji?",
            evidence_passages=[],
            conversation_id=conv.id,
        ):
            events.append(chunk)

        # Verify complete event came before title event
        complete_idx = next(i for i, ev in enumerate(events) if "event: complete" in ev)
        title_idx = next(i for i, ev in enumerate(events) if "event: title" in ev)

        assert complete_idx < title_idx, "Complete event must be yielded before title event!"
        assert "Shivaji The Warrior King" in events[title_idx]

        # Verify DB was updated
        updated_conv = await conv_mgr.get_conversation(conv.id)
        assert updated_conv.title == "Shivaji The Warrior King"


class TestNonStreamingTitleGeneration:
    """Test non-streaming answer orchestrator background title updates."""

    @pytest.mark.asyncio
    async def test_non_streaming_triggers_title_generation_background(self):
        conv_mgr = ConversationManager()
        conv = await conv_mgr.create_conversation()

        mock_llm = MagicMock(spec=OllamaLLMAdapter)
        mock_llm.model = "gemma3:4b"
        mock_llm.generate_answer = AsyncMock(return_value=MagicMock(
            answer_text="Shivaji Maharaj was a great warrior king.",
            status="ANSWERED",
            model="gemma3:4b",
            prompt_tokens=15,
            response_tokens=10,
            load_duration_ms=1.0,
            prompt_eval_duration_ms=2.0,
            eval_duration_ms=3.0,
            tokens_per_second=20.0,
        ))
        mock_llm.generate_text = AsyncMock(return_value="Shivaji Maharaj Reign")

        title_gen = ConversationTitleGenerator(llm_adapter=mock_llm)
        orchestrator = AnswerOrchestrator(
            conversation_manager=conv_mgr,
            llm_adapter=mock_llm,
            title_generator=title_gen,
        )

        req = AnswerRequest(
            question="Who was Shivaji Maharaj?",
            conversation_id=conv.id,
        )

        with patch.object(orchestrator.evidence_reasoner, "reason", new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = MagicMock(
                evidence_package=EvidencePackage(
                    question="Who was Shivaji Maharaj?",
                    evidence_clusters=[],
                    conflicts=[],
                    sufficiency_assessment=SufficiencyAssessment(
                        status="SUFFICIENT",
                        confidence=1.0,
                        missing_aspects=[],
                        reasoning="Evidence verified",
                    ),
                )
            )
            with patch.object(orchestrator.document_cache, "get_or_fetch_document", new_callable=AsyncMock) as mock_cache:
                mock_cache.return_value = (MagicMock(page_id=1, title="Shivaji"), False)
                with patch.object(orchestrator.hybrid_retrieval, "retrieve", new_callable=AsyncMock) as mock_hybrid:
                    mock_hybrid.return_value = [
                        RetrievedChunk(
                            chunk_id=1,
                            document_title="Shivaji",
                            section_title="Early Life",
                            source_url="https://en.wikipedia.org/wiki/Shivaji",
                            passage_text="Shivaji was a warrior king.",
                            bm25_rank=1,
                            vector_rank=1,
                            rrf_score=0.9,
                            retrieval_method="hybrid",
                        )
                    ]
                    resp = await orchestrator.answer(req)
                    assert resp.answer_status == "ANSWERED"
                    assert resp.conversation_id == conv.id

        # Give background task event loop cycle to complete
        await asyncio.sleep(0.1)

        # Verify title was updated in DB
        updated_conv = await conv_mgr.get_conversation(conv.id)
        assert updated_conv.title == "Shivaji Maharaj Reign"


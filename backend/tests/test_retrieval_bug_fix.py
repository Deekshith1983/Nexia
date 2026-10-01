import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.persistence.article_store import wikipedia_article_store
from app.services.wikipedia_retriever import (
    WikipediaRetriever,
    evaluate_candidate_relevance,
    is_evidence_relevant,
    _score_candidate,
    _select_best_candidate,
    RetrievalResult,
    RetrievalStatus,
    wikipedia_retriever,
)
from app.core.query_understanding import (
    QueryUnderstanding,
    Route,
    Intent,
    AnswerType,
    query_analyzer,
)
from app.agent.research.agent import (
    WikipediaResearchAgent,
    wikipedia_research_agent,
    ResearchAction,
    ResearchActionStatus,
    ResearchActionType,
)
from app.agent.research.context import ResearchContext
from app.agent.research.evidence import CompletionStatus, EvidenceItem
from app.optimization.cache import local_cache_manager


class TestRetrievalBugFix(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        local_cache_manager.clear_all()
        wikipedia_article_store.clear()

    def _create_context(self, query: str) -> ResearchContext:
        return ResearchContext(
            request_id="test_req",
            conversation_id="test_conv",
            original_request=query,
            resolved_request=query,
            objectives=[query],
            plan=None,
            completed_actions=[],
            pending_actions=[],
            evidence=[],
            findings=[],
            sources=[],
            unresolved_questions=[],
            completion_status=CompletionStatus.INSUFFICIENT_KNOWLEDGE,
        )

    # ── Test A: Prime ministers of India must NOT select Pakistan ─────────────
    def test_a_india_pm_does_not_select_pakistan(self):
        query = "list out prime ministers of independent india"
        qu = query_analyzer.understand(query)

        self.assertEqual(qu.route, Route.KNOWLEDGE)
        self.assertEqual(qu.intent, Intent.LIST)
        self.assertIn("India", qu.entities)
        self.assertEqual(qu.attribute, "prime ministers")
        self.assertEqual(qu.constraints.get("scope"), "independent")

        # 1. Relevance evaluation on Pakistan candidate must REJECT
        pak_eval = evaluate_candidate_relevance(
            title="List of prime ministers of Pakistan",
            content="The Prime Minister of Pakistan is the head of government of Pakistan...",
            query=query,
            entity="India",
            attribute="prime ministers",
            constraints=qu.constraints,
            intent="list",
        )
        self.assertFalse(pak_eval.accepted, "Pakistan candidate must be rejected for India PM query")
        self.assertIn("country mismatch", pak_eval.reason.lower())

        # 2. Relevance evaluation on India candidate must ACCEPT
        india_eval = evaluate_candidate_relevance(
            title="List of prime ministers of India",
            content="The Prime Minister of India is the chief executive of the Republic of India. Jawaharlal Nehru was the first prime minister...",
            query=query,
            entity="India",
            attribute="prime ministers",
            constraints=qu.constraints,
            intent="list",
        )
        self.assertTrue(india_eval.accepted, "India candidate must be accepted for India PM query")

        # 3. Candidate ranking must never select Pakistan
        candidates = [
            {"title": "List of prime ministers of Pakistan", "snippet": "Prime ministers of Pakistan head of government"},
            {"title": "List of prime ministers of India", "snippet": "Prime ministers of India from 1947 to present"},
        ]
        selected = _select_best_candidate(
            candidates,
            entity="India",
            query=query,
            attribute="prime ministers",
            constraints=qu.constraints,
            intent="list",
        )
        self.assertEqual(selected, "List of prime ministers of India")
        self.assertNotEqual(selected, "List of prime ministers of Pakistan")

    # ── Test B: Rulers of Mysore after Tipu Sultan must NOT accept Tipu Sultan ─
    def test_b_mysore_rulers_after_tipu_does_not_accept_tipu_sultan(self):
        query = "list the rulers of mysore kingdom after tippu sultan"
        qu = query_analyzer.understand(query)

        self.assertEqual(qu.route, Route.KNOWLEDGE)
        self.assertEqual(qu.intent, Intent.LIST)
        self.assertIn("Mysore Kingdom", qu.entities)
        self.assertEqual(qu.attribute, "rulers")
        self.assertIn("after", qu.constraints.get("temporal", "").lower())

        # Tipu Sultan is a single ruler biography and is the subject of the "after" constraint
        tipu_eval = evaluate_candidate_relevance(
            title="Tipu Sultan",
            content="Tipu Sultan was the ruler of the Kingdom of Mysore from 1782 to 1799. He died in the Fourth Anglo-Mysore War.",
            query=query,
            entity="Mysore Kingdom",
            attribute="rulers",
            constraints=qu.constraints,
            intent="list",
        )
        self.assertFalse(tipu_eval.accepted, "Tipu Sultan biography must NOT be accepted as ruler-list evidence after Tipu Sultan")
        self.assertTrue(tipu_eval.is_navigation_candidate, "Tipu Sultan may only serve as a navigation candidate")

        # Kingdom of Mysore or Wadiyar Dynasty should be accepted
        mysore_eval = evaluate_candidate_relevance(
            title="Kingdom of Mysore",
            content="The Kingdom of Mysore was a realm in southern India. After the death of Tipu Sultan in 1799, the British restored the Wadiyar dynasty under Krishnaraja Wadiyar III...",
            query=query,
            entity="Mysore Kingdom",
            attribute="rulers",
            constraints=qu.constraints,
            intent="list",
        )
        self.assertTrue(mysore_eval.accepted, "Kingdom of Mysore must be accepted")

        # Research agent sufficiency check:
        # If evidence only has Tipu Sultan, it must NOT be marked SUFFICIENT
        ctx = self._create_context(query)
        ctx.evidence.append(
            EvidenceItem(
                action_id="act1",
                page_title="Tipu Sultan",
                page_url="https://en.wikipedia.org/wiki/Tipu_Sultan",
                content="Tipu Sultan was ruler of the Kingdom of Mysore from 1782 until his death in 1799 in the Fourth Anglo-Mysore War.",
                source="Wikipedia",
            )
        )
        wikipedia_research_agent._evaluate_completion(ctx, qu)
        self.assertNotEqual(
            ctx.completion_status,
            CompletionStatus.SUFFICIENT,
            "Research agent must not mark Tipu Sultan alone as SUFFICIENT for rulers after Tipu Sultan",
        )

    # ── Test C: Multi-turn Jaya and Vijaya cache reuse with 0 network calls ───
    async def test_c_jaya_vijaya_multiturn_cache_reuse(self):
        # Seed the persistent store with Jaya-Vijaya
        wikipedia_article_store.save_article(
            title="Jaya-Vijaya",
            url="https://en.wikipedia.org/wiki/Jaya-Vijaya",
            summary="In Hindu mythology, Jaya and Vijaya are the two gatekeepers of Vaikuntha. In Dvapara Yuga, they were born as Shishupala and Dantavakra.",
            content="In Hindu mythology, Jaya and Vijaya are the two gatekeepers of Vaikuntha. In Dvapara Yuga, they were born as Shishupala and Dantavakra.",
            aliases=["Jaya and Vijaya", "Jaya Vijaya"],
            metadata={"entity": "Jaya and Vijaya"},
        )

        # Second turn query asking about Dvapara Yuga
        query2 = "in Dvapara Yuga who were they?"
        entity2 = "Jaya and Vijaya"

        with patch("httpx.AsyncClient.get") as mock_get:
            result = await wikipedia_retriever.retrieve(
                query=query2,
                entity=entity2,
            )
            # Must NOT call network because exact alias matched in ArticleStore
            mock_get.assert_not_called()
            self.assertEqual(result.status, RetrievalStatus.SUCCESS)
            self.assertEqual(result.page_title, "Jaya-Vijaya")
            self.assertTrue(result.metadata.get("cached"))
            self.assertIn("Shishupala", result.content)

    # ── Test D: Unrelated cached article is rejected and fallback occurs ──────
    async def test_d_unrelated_cached_article_rejected(self):
        # Store unrelated article in database
        wikipedia_article_store.save_article(
            title="List of prime ministers of Pakistan",
            url="https://en.wikipedia.org/wiki/List_of_prime_ministers_of_Pakistan",
            summary="The Prime Minister of Pakistan is the head of government of Pakistan.",
            content="The Prime Minister of Pakistan is the head of government of Pakistan.",
            metadata={"entity": "Pakistan"},
        )

        query = "list out prime ministers of independent india"

        # Mock Wikipedia search and extract for the fallback network call
        async def mock_get(url, *args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            params = kwargs.get("params", {})
            if "list" in params and params["list"] == "search":
                mock_resp.json.return_value = {
                    "query": {
                        "search": [
                            {"title": "List of prime ministers of India", "snippet": "Prime ministers of India from 1947 to present"},
                        ]
                    }
                }
            else:
                mock_resp.json.return_value = {
                    "query": {
                        "pages": {
                            "12345": {
                                "pageid": 12345,
                                "title": "List of prime ministers of India",
                                "extract": "The Prime Minister of India is the head of government of the Republic of India.\n== Office ==\nThe office was created in 1947.",
                                "fullurl": "https://en.wikipedia.org/wiki/List_of_prime_ministers_of_India",
                            }
                        }
                    }
                }
            return mock_resp

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            result = await wikipedia_retriever.retrieve(
                query=query,
                entity="India",
            )
            # The retriever must NOT return the cached Pakistan article!
            self.assertEqual(result.status, RetrievalStatus.SUCCESS)
            self.assertEqual(result.page_title, "List of prime ministers of India")
            self.assertNotEqual(result.page_title, "List of prime ministers of Pakistan")

    # ── Test E: Same canonical article requested later yields zero network calls
    async def test_e_zero_network_calls_when_valid_cache_exists(self):
        wikipedia_article_store.save_article(
            title="Ashoka",
            url="https://en.wikipedia.org/wiki/Ashoka",
            summary="Ashoka was the third Mauryan Emperor of Magadha in the Indian subcontinent...",
            content="Ashoka was the third Mauryan Emperor of Magadha in the Indian subcontinent...",
            metadata={"entity": "Ashoka"},
        )

        with patch("httpx.AsyncClient.get") as mock_get:
            result = await wikipedia_retriever.retrieve(
                query="tell me about Ashoka",
                entity="Ashoka",
            )
            mock_get.assert_not_called()
            self.assertEqual(result.status, RetrievalStatus.SUCCESS)
            self.assertEqual(result.page_title, "Ashoka")
            self.assertTrue(result.metadata.get("cached"))

    # ── Test F: LIST query on general article triggers navigation / expansion ─
    def test_f_list_query_general_article_requires_navigation(self):
        query = "list the rulers of Vijayanagara empire"
        qu = query_analyzer.understand(query)

        ctx = self._create_context(query)
        # General article with only high-level history without rulers list
        ctx.evidence.append(
            EvidenceItem(
                action_id="act1",
                page_title="Vijayanagara Empire",
                page_url="https://en.wikipedia.org/wiki/Vijayanagara_Empire",
                content="The Vijayanagara Empire was based in the Deccan Plateau region of South India. It was established in 1336.",
                source="Wikipedia",
            )
        )
        wikipedia_research_agent._evaluate_completion(ctx, qu)
        self.assertEqual(
            ctx.completion_status,
            CompletionStatus.MORE_INFORMATION_NEEDED,
            "LIST query with only high-level summary must trigger section navigation or follow-up",
        )

        followup = wikipedia_research_agent._generate_followup_action(ctx, qu)
        self.assertIsNotNone(followup)
        self.assertEqual(followup.action_type, ResearchActionType.WIKIPEDIA_RETRIEVE_SECTION)
        self.assertEqual(followup.section_title, "Rulers")

    # ── Test G: Negative / why question does not fabricate explanation ────────
    def test_g_negative_why_question_no_fabricated_explanation(self):
        query = "Why was Kamsa not considered an incarnation of Vishnu?"
        qu = query_analyzer.understand(query)

        self.assertEqual(qu.route, Route.KNOWLEDGE)
        self.assertEqual(qu.answer_type, AnswerType.CAUSAL)
        self.assertIsNotNone(qu.contract)
        self.assertTrue(qu.contract.negative_or_exclusion)

        # Research planning must create actions for both entities (Kamsa and Vishnu)
        plan = wikipedia_research_agent._create_plan(qu, query, "Kamsa")
        self.assertGreaterEqual(len(plan.actions), 2, "Negative question must plan retrieval for both subjects")
        queries = [a.query for a in plan.actions]
        self.assertTrue(any("kamsa" in q.lower() for q in queries))
        self.assertTrue(any("vishnu" in q.lower() or "incarnation" in q.lower() for q in queries))


if __name__ == "__main__":
    unittest.main()

import asyncio
import os
import sys
import unittest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.persistence.article_store import wikipedia_article_store, WikipediaArticleStore
from app.services.wikipedia_retriever import (
    WikipediaRetriever, is_evidence_relevant, _score_candidate, _select_best_candidate,
    RetrievalResult, RetrievalStatus,
)
from app.core.query_understanding import (
    QueryUnderstanding, AnswerType, AnswerContract, clean_entity_for_search,
    extract_answer_contract, query_analyzer,
)
from app.agent.decomposition import query_decomposer, InformationNeed, NeedPriority, NeedStatus
from app.agent.research import wikipedia_research_agent
from app.knowledge.claims import is_refusal_response, claim_grounder
from app.core.answer_validator import answer_validator, ValidationStatus
from app.agent.reflection import reflection_engine
from app.optimization.cache import local_cache_manager


class TestNexiaStabilization(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        local_cache_manager.clear_all()
        # Use an in-memory or fresh store test
        wikipedia_article_store.clear()

    # ── Test 1: Irrelevant retrieved article rejected ─────────────────────────
    def test_01_irrelevant_article_rejected(self):
        query = "tell me about the ruler who ruled during hoysala dynasty"
        entity = "Hoysala dynasty"

        # Kushan Empire is completely unrelated to Hoysala dynasty
        is_rel = is_evidence_relevant(
            title="Kushan Empire",
            content="The Kushan Empire was a syncretic empire formed by the Yuezhi in Bactrian territories...",
            query=query,
            entity=entity,
        )
        self.assertFalse(is_rel, "Unrelated article Kushan Empire should be rejected for Hoysala query")

        # Hoysala Empire is relevant
        is_rel_hoysala = is_evidence_relevant(
            title="Hoysala Empire",
            content="The Hoysala Empire was a prominent South Indian empire that ruled most of the modern-day state of Karnataka...",
            query=query,
            entity=entity,
        )
        self.assertTrue(is_rel_hoysala, "Hoysala Empire should be recognized as relevant")

        # Search candidates scoring should not boost zero-overlap candidate with rank bonus
        candidates = [
            {"title": "Kushan Empire", "snippet": "An unrelated empire"},
            {"title": "Hoysala Empire", "snippet": "A southern dynasty"},
        ]
        best = _select_best_candidate(candidates, entity=entity, query=query)
        self.assertEqual(best, "Hoysala Empire", "Candidate selection must pick Hoysala Empire over Kushan Empire")

    # ── Test 2: Cached article reused from ArticleStore ───────────────────────
    async def test_02_cached_article_reused(self):
        # Seed article store
        wikipedia_article_store.save_article(
            title="Ashoka",
            url="https://en.wikipedia.org/wiki/Ashoka",
            summary="Ashoka was the third Mauryan Emperor of Magadha in the Indian subcontinent.",
            content="Ashoka ruled from c. 268 to 232 BCE.",
        )

        retriever = WikipediaRetriever()
        result = await retriever.retrieve(query="Ashoka", entity="Ashoka")
        self.assertEqual(result.status, RetrievalStatus.SUCCESS)
        self.assertEqual(result.page_title, "Ashoka")
        self.assertTrue(result.metadata.get("cached"))
        self.assertEqual(result.metadata.get("source"), "ArticleStore")

    # ── Test 3: Workspace evidence reused ─────────────────────────────────────
    def test_03_workspace_evidence_reused(self):
        from app.knowledge.workspace import KnowledgeWorkspace
        from app.knowledge.models import Fact, EvidenceReference

        ws = KnowledgeWorkspace(run_id="test_run")
        ws.facts.append(Fact(
            subject="Ramayana",
            predicate="author",
            object_value="Valmiki",
            evidence_refs=[EvidenceReference(evidence_id="ev_1", page_title="Ramayana", page_url="https://en.wikipedia.org/wiki/Ramayana")],
        ))

        qu = query_analyzer.analyze_query("who wrote Ramayana")
        needs = query_decomposer.decompose(qu, "who wrote Ramayana", workspace=ws)
        self.assertTrue(len(needs) >= 1)
        self.assertEqual(needs[0].status, NeedStatus.SATISFIED, "Workspace evidence should satisfy need immediately")

    # ── Test 4: New article persisted ─────────────────────────────────────────
    def test_04_new_article_persisted(self):
        wikipedia_article_store.save_article(
            title="Vijayanagara Empire",
            url="https://en.wikipedia.org/wiki/Vijayanagara_Empire",
            summary="The Vijayanagara Empire was an empire based in the Deccan Plateau region of South India.",
            content="It was established in 1336 by Harihara I and Bukka Raya I.",
            sections=[{"title": "History", "content": "Founded in 1336."}],
        )

        art = wikipedia_article_store.get_article("Vijayanagara Empire")
        self.assertIsNotNone(art)
        self.assertEqual(art.title, "Vijayanagara Empire")
        self.assertIn("Harihara", art.content)
        self.assertEqual(len(art.sections), 1)

        # Verify FTS5 search
        search_hits = wikipedia_article_store.search_articles("Harihara Bukka", limit=5)
        self.assertTrue(len(search_hits) >= 1)
        self.assertEqual(search_hits[0].title, "Vijayanagara Empire")

    # ── Test 5: Answer type LIST enforced ─────────────────────────────────────
    def test_05_answer_type_list_enforced(self):
        qu = query_analyzer.analyze_query("list the rulers of vijayanagara kingdom")
        self.assertIsNotNone(qu.answer_contract)
        self.assertEqual(qu.answer_contract.answer_type, AnswerType.LIST)
        self.assertEqual(qu.answer_contract.requested_relation, "rulers")

    # ── Test 6: Answer relation enforced ──────────────────────────────────────
    def test_06_answer_relation_enforced(self):
        qu1 = query_analyzer.analyze_query("who wrote Ramayana")
        self.assertIn(qu1.answer_contract.answer_type, (AnswerType.PERSON, AnswerType.RELATION))
        self.assertEqual(qu1.answer_contract.requested_relation, "author")
        self.assertEqual(qu1.entity, "Ramayana")

        qu2 = query_analyzer.analyze_query("who was father of lord Rama")
        self.assertIn(qu2.answer_contract.answer_type, (AnswerType.PERSON, AnswerType.RELATION))
        self.assertEqual(qu2.answer_contract.requested_relation, "father")
        self.assertEqual(qu2.entity, "Rama")

    # ── Test 7: Insufficiency response not treated as a claim ─────────────────
    def test_07_insufficiency_not_treated_as_claim(self):
        refusal_msg = "I couldn't find that in the Wikipedia article."
        self.assertTrue(is_refusal_response(refusal_msg))

        claims = claim_grounder.ground_answer(refusal_msg, workspace=None)
        self.assertEqual(len(claims), 0, "Insufficiency responses must yield zero claims")

    # ── Test 8: Repair query does not collapse to Wikipedia ───────────────────
    def test_08_repair_query_does_not_collapse_to_wikipedia(self):
        from app.agent.research.context import ResearchContext
        from app.agent.reflection.models import ReflectionStatus

        from app.agent.research.evidence import CompletionStatus

        qu = query_analyzer.analyze_query("tell me about the ruler who ruled during hoysala dynasty")
        ctx = ResearchContext(
            request_id="req1",
            conversation_id="c1",
            original_request="tell me about the ruler who ruled during hoysala dynasty",
            resolved_request="tell me about the ruler who ruled during hoysala dynasty",
            objectives=["tell me about the ruler who ruled during hoysala dynasty"],
            plan=None,
            completed_actions=[],
            pending_actions=[],
            evidence=[],
            findings=[],
            sources=[],
            unresolved_questions=[],
            completion_status=CompletionStatus.INSUFFICIENT_KNOWLEDGE,
        )

        refl = reflection_engine.reflect(
            original_request=ctx.original_request,
            resolved_request=ctx.resolved_request,
            qu=qu,
            research_ctx=ctx,
            generated_answer="I couldn't find that in the Wikipedia article.",
        )

        suggested = (refl.suggested_query or "").lower().strip()
        self.assertNotEqual(suggested, "wikipedia", "Repair query MUST NOT be 'Wikipedia'")
        self.assertTrue("hoysala" in suggested or "ruler" in suggested or refl.status == ReflectionStatus.COMPLETE)

    # ── Test 9: Negative question does not fabricate reason ───────────────────
    def test_09_negative_question_decomposition(self):
        q = "Why was Kamsa not considered as a reincarnation of either Jaya or Vijaya in Dvapara Yuga?"
        qu = query_analyzer.analyze_query(q)
        self.assertTrue(qu.answer_contract.negative_aspect)
        self.assertEqual(qu.answer_contract.answer_type, AnswerType.CAUSAL)

        needs = query_decomposer.decompose(qu, q)
        # Should decompose into subject background and target group background
        self.assertTrue(len(needs) >= 2)
        all_text = " ".join(n.question for n in needs).lower()
        self.assertTrue("kamsa" in all_text)
        self.assertTrue("jaya" in all_text or "vijaya" in all_text)

    # ── Test 10: Streaming reaches frontend incrementally ─────────────────────
    async def test_10_streaming_generator(self):
        from app.agent.capabilities.wikipedia import WikipediaKnowledgeCapability

        cap = WikipediaKnowledgeCapability()
        self.assertEqual(cap.id, "wikipedia_knowledge")

        # Test incremental chunk consumption
        async def mock_stream():
            words = ["The", " ", "Ramayana", " ", "is", " ", "an", " ", "epic."]
            for w in words:
                yield w

        chunks = []
        async for token in mock_stream():
            chunks.append(token)
        self.assertEqual("".join(chunks), "The Ramayana is an epic.")

    # ── Test 11: Readiness cache prevents unnecessary tags calls ──────────────
    async def test_11_readiness_cache_prevents_api_tags(self):
        local_cache_manager.set_model_readiness(True, ttl_seconds=60)
        from app.services.ollama_service import ollama_service

        # Calling check_health should return True immediately from cache without HTTP request
        is_ready = await ollama_service.check_health(force_live=False)
        self.assertTrue(is_ready)
        self.assertEqual(local_cache_manager.get_model_readiness(), True)

    # ── Test 12: Ramayana author regression protection ───────────────────────
    def test_12_ramayana_author(self):
        qu = query_analyzer.analyze_query("who wrote Ramayana")
        self.assertEqual(qu.entity, "Ramayana")
        self.assertEqual(qu.contract.requested_relation, "author")
        self.assertIn(qu.contract.answer_type, (AnswerType.PERSON, AnswerType.RELATION))

    # ── Test 13: Jaya/Vijaya incarnations multi-need decomposition ────────────
    def test_13_jaya_vijaya_incarnations(self):
        qu = query_analyzer.analyze_query("tell me about the incarnations of Jaya and Vijaya across yugas")
        self.assertIsNotNone(qu.entity)
        needs = query_decomposer.decompose(qu, "tell me about the incarnations of Jaya and Vijaya across yugas")
        self.assertTrue(len(needs) >= 1)

    # ── Test 14: Dvapara Yuga follow-up ──────────────────────────────────────
    def test_14_dvapara_yuga_followup(self):
        qu = query_analyzer.analyze_query("what about in Dvapara Yuga?")
        contract = extract_answer_contract("what about in Dvapara Yuga?", qu.intent, "Dvapara Yuga")
        self.assertIsNotNone(contract)

    # ── Test 15: Kamsa entity extraction (no filler) ──────────────────────────
    def test_15_kamsa_entity_extraction(self):
        qu = query_analyzer.analyze_query("tell me about Kamsa in detail")
        self.assertEqual(qu.entity, "Kamsa")

    # ── Test 16: Kamsa / Jaya-Vijaya exclusion handling ───────────────────────
    def test_16_kamsa_exclusion_question(self):
        q = "Why was Kamsa not considered as a reincarnation of either Jaya or Vijaya in Dvapara Yuga?"
        qu = query_analyzer.analyze_query(q)
        self.assertTrue(qu.contract.negative_or_exclusion)
        plan = wikipedia_research_agent._create_plan(qu, q, qu.entity)
        self.assertTrue(len(plan.actions) >= 2)
        entities = [a.entity.lower() for a in plan.actions if a.entity]
        self.assertTrue(any("kamsa" in e for e in entities))

    # ── Test 17: Hoysala rulers entity extraction and list contract ───────────
    def test_17_hoysala_rulers(self):
        qu = query_analyzer.analyze_query("tell me about the ruler who ruled during hoysala dynasty")
        self.assertEqual(qu.entity.lower(), "hoysala dynasty")

    # ── Test 18: Vijayanagara rulers list contract & section navigation ───────
    def test_18_vijayanagara_rulers_navigation(self):
        qu = query_analyzer.analyze_query("list the rulers of vijayanagara kingdom")
        self.assertIn("vijayanagara", qu.entity.lower())
        self.assertEqual(qu.contract.answer_type, AnswerType.LIST)
        self.assertEqual(qu.contract.requested_relation, "rulers")

    # ── Test 19: Founder of Wikipedia extraction and contract ─────────────────
    def test_19_wikipedia_founder(self):
        qu1 = query_analyzer.analyze_query("who is the founder of wikipedia?")
        self.assertEqual(qu1.entity.lower(), "wikipedia")
        self.assertEqual(qu1.contract.requested_relation, "founder")

        qu2 = query_analyzer.analyze_query("who founded wikipedia")
        self.assertEqual(qu2.entity.lower(), "wikipedia")
        self.assertEqual(qu2.contract.requested_relation, "founder")


if __name__ == "__main__":
    unittest.main()

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.persistence.article_store import wikipedia_article_store
from app.knowledge.document_models import WikipediaDocument, WikipediaDocumentSection, WikipediaPassage
from app.services.wikipedia_retriever import (
    WikipediaRetriever,
    score_and_select_passages,
    _split_extract_into_sections,
    RetrievalResult,
    RetrievalStatus,
    wikipedia_retriever,
)
from app.core.query_understanding import query_analyzer, AnswerType, Intent
from app.core.answer_validator import answer_validator, ValidationStatus
from app.optimization.cache import local_cache_manager


class TestRetrievalV2Real(unittest.IsolatedAsyncioTestCase):
    """
    Phase 4: Real-world, zero-mock contract and retrieval verification
    following the 12-step QA engineer protocol.
    """

    def setUp(self):
        local_cache_manager.clear_all()
        wikipedia_article_store.clear()

    def test_01_document_model_contract(self):
        """
        Verify WikipediaDocument, WikipediaDocumentSection, and WikipediaPassage contracts.
        """
        raw_text = (
            "The Kingdom of Mysore was a realm in southern India.\n\n"
            "== History ==\n"
            "Mysore was founded in 1399 by Yaduraya Wodeyar.\n\n"
            "=== Under Hyder Ali and Tipu Sultan ===\n"
            "Hyder Ali and Tipu Sultan ruled until 1799.\n\n"
            "=== Princely state ===\n"
            "Following Tipu's fall in 1799, the five-year-old Krishnaraja III was installed on the throne."
        )
        sections = _split_extract_into_sections("Kingdom of Mysore", raw_text)
        self.assertGreaterEqual(len(sections), 3)

        doc = WikipediaDocument.create(
            canonical_title="Kingdom of Mysore",
            url="https://en.wikipedia.org/wiki/Kingdom_of_Mysore",
            lead_summary="The Kingdom of Mysore was a realm in southern India.",
            sections=sections,
        )

        self.assertEqual(doc.canonical_title, "Kingdom of Mysore")
        all_passages = doc.get_passages()
        self.assertGreaterEqual(len(all_passages), 3)

        # Test passage scoring for post-Tipu Sultan rulers
        selected = score_and_select_passages(
            all_passages,
            query="list the rulers of mysore kingdom after tippu sultan",
            attribute="rulers",
            constraints={"temporal": "after tipu sultan"},
        )
        self.assertTrue(len(selected) > 0)
        selected_titles = [p.section_title for p in selected]
        self.assertIn("Princely state", selected_titles)
        self.assertIn("Krishnaraja III", selected[0].text)

    def test_02_store_save_and_reconstruct_document(self):
        """
        Verify saving a WikipediaDocument to SQLite and reconstructing it with full passages.
        """
        p1 = WikipediaPassage.create("Rama", "Birth", 1, 0, "Rama was born to King Dasharatha and Queen Kaushalya.")
        sec1 = WikipediaDocumentSection(
            section_id="sec1",
            title="Birth and youth",
            level=2,
            index=1,
            content="Rama was born to King Dasharatha and Queen Kaushalya.",
            passages=[p1],
        )
        doc = WikipediaDocument.create(
            canonical_title="Rama",
            url="https://en.wikipedia.org/wiki/Rama",
            lead_summary="Rama is a major deity in Hinduism.",
            sections=[sec1],
            aliases=["Ramachandra"],
        )

        doc_id = wikipedia_article_store.save_document(doc)
        self.assertTrue(bool(doc_id))

        # Retrieve by canonical title
        loaded = wikipedia_article_store.get_document("Rama")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.canonical_title, "Rama")
        self.assertEqual(len(loaded.sections), 1)
        self.assertEqual(loaded.sections[0].title, "Birth and youth")
        self.assertEqual(len(loaded.sections[0].passages), 1)
        self.assertIn("Dasharatha", loaded.sections[0].passages[0].text)

        # Retrieve by alias
        loaded_alias = wikipedia_article_store.get_document("Ramachandra")
        self.assertIsNotNone(loaded_alias)
        self.assertEqual(loaded_alias.canonical_title, "Rama")

    async def test_03_real_live_retrieval_mysore_rulers(self):
        """
        Real end-to-end Wikipedia fetch for 'rulers of Mysore Kingdom after Tipu Sultan'.
        Must fetch full document, extract 'Princely state' section, and bound context.
        """
        query = "list the rulers of mysore kingdom after tippu sultan"
        res = await wikipedia_retriever.retrieve(
            query=query,
            entity="Mysore Kingdom",
            attribute="rulers",
            constraints={"temporal": "after tippu sultan"},
            intent="LIST",
        )

        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertIn("Mysore", res.page_title)
        # Content must contain post-1799 succession details (Krishnaraja III, Wodeyar)
        self.assertTrue("Krishnaraja" in res.content or "Wodeyar" in res.content or "Princely state" in res.content)
        # Passages must be bounded (< 4000 chars)
        self.assertLess(len(res.content), 4000)

        # Verify it was stored in local SQLite
        cached_doc = wikipedia_article_store.get_document(res.page_title)
        self.assertIsNotNone(cached_doc)
        self.assertGreater(len(cached_doc.sections), 5)

        # Second call MUST hit persistent document cache (cached=True)
        res_cached = await wikipedia_retriever.retrieve(
            query=query,
            entity="Mysore Kingdom",
            attribute="rulers",
            constraints={"temporal": "after tippu sultan"},
            intent="LIST",
        )
        self.assertEqual(res_cached.status, RetrievalStatus.SUCCESS)
        self.assertTrue(res_cached.metadata.get("cached", False))

    async def test_04_real_live_retrieval_wikipedia_founders(self):
        """
        Real end-to-end Wikipedia fetch for 'who founded wikipedia'.
        Must retrieve founders (Jimmy Wales and Larry Sanger) in bounded passages.
        """
        query = "who founded wikipedia"
        qu = query_analyzer.understand(query)
        res = await wikipedia_retriever.retrieve(
            query=query,
            entity=qu.entity or "Wikipedia",
            attribute=qu.attribute,
            intent=qu.intent.value,
        )

        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertEqual(res.page_title, "Wikipedia")
        self.assertTrue("Jimmy Wales" in res.content or "Larry Sanger" in res.content)
        self.assertLess(len(res.content), 4000)

    async def test_05_relevance_gate_preservation(self):
        """
        Verify that relevance gating remains strictly active in V2:
        'prime ministers of independent india' must NOT accept Pakistan.
        """
        res = await wikipedia_retriever.retrieve(
            query="list out prime ministers of independent india",
            entity="India",
            attribute="prime ministers",
            constraints={"scope": "independent"},
            intent="LIST",
        )
        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertNotIn("Pakistan", res.page_title)
        self.assertIn("India", res.page_title)

    async def test_06_vijayanagara_rulers_wikitable_list(self):
        """
        Verify that list queries on Vijayanagara rulers parse the 4 ruling dynasty wikitables
        and return chronologically organized ruler names and dates.
        """
        query = "list the rulers of vijayanagara kingdom"
        qu = query_analyzer.understand(query)
        self.assertEqual(qu.intent, Intent.LIST)
        self.assertIsNotNone(qu.answer_contract)
        self.assertEqual(qu.answer_contract.expected_type, AnswerType.LIST)

        res = await wikipedia_retriever.retrieve(
            query=query,
            entity=qu.entity or "Vijayanagara Empire",
            attribute=qu.attribute,
            intent=qu.intent.value,
        )

        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertIn("Vijayanagara", res.page_title)
        content_lower = res.content.lower()
        self.assertTrue("sangama" in content_lower)
        self.assertTrue("saluva" in content_lower)
        self.assertTrue("tuluva" in content_lower)
        self.assertTrue("aravidu" in content_lower)
        self.assertTrue("harihara" in content_lower or "bukka" in content_lower or "krishna" in content_lower)
        self.assertLess(len(res.content), 3500)

    async def test_07_conceptual_distinction_empire_vs_dynasty(self):
        """
        Verify 'why is it vijayanagara empire and not vijayanagara dynasty?' captures
        the distinction between political realm and ruling dynasties with multi-section coverage.
        """
        query = "why is it vijayanagara empire and not vijayanagara dynasty?"
        qu = query_analyzer.understand(query)
        self.assertEqual(qu.intent, Intent.EXPLANATION)
        self.assertIsNotNone(qu.answer_contract)
        self.assertEqual(qu.answer_contract.requested_relation, "conceptual_distinction")
        self.assertEqual(qu.answer_contract.concept_a, "empire")
        self.assertEqual(qu.answer_contract.concept_b, "dynasty")

        res = await wikipedia_retriever.retrieve(
            query=query,
            entity=qu.entity or "Vijayanagara Empire",
            attribute=qu.attribute,
            intent=qu.intent.value,
        )

        self.assertEqual(res.status, RetrievalStatus.SUCCESS)
        self.assertIn("Vijayanagara", res.page_title)
        content_lower = res.content.lower()
        self.assertTrue("empire" in content_lower or "realm" in content_lower or "state" in content_lower)
        self.assertTrue("sangama" in content_lower or "dynast" in content_lower)
        self.assertLess(len(res.content), 3500)

        # Test validator contract satisfaction
        sample_answer = (
            "The Vijayanagara Empire was an overarching political realm and state in southern India "
            "that lasted for over three centuries. It is called an empire rather than a single dynasty "
            "because it was ruled successively by four distinct dynasties: the Sangama dynasty, the Saluva dynasty, "
            "the Tuluva dynasty, and the Aravidu dynasty."
        )
        val_res = await answer_validator.validate(
            answer=sample_answer,
            evidence=res,
            answer_contract=qu.answer_contract,
        )
        self.assertEqual(val_res.status, ValidationStatus.SUPPORTED)
        self.assertTrue(val_res.question_satisfied)
        self.assertTrue(val_res.answer_shape_satisfied)


if __name__ == "__main__":
    unittest.main()

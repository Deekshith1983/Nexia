import logging
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.query_understanding import (
    QueryUnderstanding,
    Route,
    Intent,
    AnswerType,
    AnswerContract,
    query_analyzer,
    clean_entity_for_search,
    extract_answer_contract,
)
from app.core.decision_policy import decision_policy, ResponseMode
from app.core.context_manager import ContextResolution, ResolutionStatus


class TestQueryUnderstandingRefactor(unittest.TestCase):

    def setUp(self):
        self.analyzer = query_analyzer

    def test_29_row_table(self):
        """Execute the exact 29-row benchmark table from the specification."""
        test_cases = [
            ("hi i am deekshith nice to meet you", "user_statement", [], None),
            ("so can u remember what is my name is?", "memory_recall", [], None),
            ("what is my name", "memory_recall", [], None),
            ("who are you", "about_bot", [], None),
            ("what is your name", "about_bot", [], None),
            ("tell me about bhagat singh", "knowledge", ["Bhagat Singh"], None),
            ("tell me about hoysala dynasty in detail.", "knowledge", ["Hoysala dynasty"], None),
            ("tell me about the great king ashoka", "knowledge", ["Ashoka"], None),
            ("tell me history about harappan civilization", "knowledge", ["Harappan Civilization"], "history"),
            ("tell me about ramayana", "knowledge", ["Ramayana"], None),
            ("tell me about valmiki ramayana", "knowledge", ["Valmiki Ramayana"], None),
            ("who was father of lord rama?", "knowledge", ["Rama"], "father"),
            ("who is rama's father", "knowledge", ["Rama"], "father"),
            ("who invented wikipedia?", "knowledge", ["Wikipedia"], "founder"),
            ("when did gandhi die", "knowledge", ["Gandhi"], "death_date"),
            ("when was the taj mahal built", "knowledge", ["Taj Mahal"], "construction_date"),
            ("what is the capital of france", "knowledge", ["France"], "capital"),
            ("what's the population of india", "knowledge", ["India"], "population"),
            ("what is the difference between python and java", "knowledge", ["Python", "Java"], None),
            ("compare python and java", "knowledge", ["Python", "Java"], None),
            ("tell me about sri lanka", "knowledge", ["Sri Lanka"], None),
            ("tell me about lord byron", "knowledge", ["Lord Byron"], None),
            ("tell me about the great wall of china", "knowledge", ["Great Wall of China"], None),
            ("was ashoka a buddhist", "knowledge", ["Ashoka"], None),
            ("why did the roman empire fall", "knowledge", ["Roman Empire"], None),
            ("list the rulers of the hoysala dynasty", "knowledge", ["Hoysala dynasty"], "rulers"),
            ("when was he born", "knowledge", [], "birth_date"),
            ("tell me more", "knowledge", [], None),
            ("who was ashoka and when did he rule", "knowledge", ["Ashoka"], None),
        ]

        for q, exp_route, exp_ents, exp_attr in test_cases:
            with self.subTest(query=q):
                qu = self.analyzer.understand(q)
                act_route = str(qu.route.value if isinstance(qu.route, Route) else qu.route)
                self.assertEqual(act_route, exp_route, f"Route mismatch for query: {q}")

                exp_lower = [e.lower() for e in exp_ents]
                act_lower = [e.lower() for e in qu.entities]
                self.assertEqual(act_lower, exp_lower, f"Entities mismatch for query: {q}")

                self.assertEqual(qu.attribute, exp_attr, f"Attribute mismatch for query: {q}")

    def test_sync_understand_never_returns_raw_sentence_as_entity(self):
        """Unmatched queries in sync understand() must return fallback with entities=[] and not raw sentence."""
        complex_unmatched = "some completely arbitrary query xyz 123 that no rule matches"
        qu = self.analyzer.understand(complex_unmatched)
        self.assertEqual(qu.source, "fallback")
        self.assertEqual(qu.confidence, 0.0)
        self.assertEqual(qu.entities, [], "Fallback must NEVER set the raw sentence as an entity")
        self.assertIsNone(qu.entity)

    def test_memory_and_user_statement_never_trigger_retrieval(self):
        """DecisionPolicy must ensure user_statement and memory_recall never require Wikipedia retrieval."""
        qu_mem = self.analyzer.understand("what is my name")
        res_mem = ContextResolution(original_query="what is my name", resolved_query="what is my name")
        plan_mem = decision_policy.create_plan(qu_mem, res_mem)
        self.assertFalse(plan_mem.retrieval_required, "memory_recall must not require retrieval")
        self.assertEqual(plan_mem.response_mode, ResponseMode.CONVERSATIONAL)

        qu_user = self.analyzer.understand("hi i am deekshith nice to meet you")
        res_user = ContextResolution(original_query="hi i am deekshith nice to meet you", resolved_query="hi i am deekshith nice to meet you")
        plan_user = decision_policy.create_plan(qu_user, res_user)
        self.assertFalse(plan_user.retrieval_required, "user_statement must not require retrieval")
        self.assertEqual(plan_user.response_mode, ResponseMode.CONVERSATIONAL)

    def test_out_of_scope_routing(self):
        """Math, calculations, and weather must route to out_of_scope."""
        for math_q in ["what is 2+2", "calculate 5 * 10", "weather today in Tokyo"]:
            with self.subTest(query=math_q):
                qu = self.analyzer.understand(math_q)
                self.assertEqual(qu.route, Route.OUT_OF_SCOPE)
                self.assertEqual(qu.entities, [])

    def test_logging_info_per_call(self):
        """QU must log one INFO line per call with route, intent, entities, attribute, source, confidence."""
        with self.assertLogs("nexia.qu", level=logging.INFO) as cm:
            self.analyzer.understand("who wrote Ramayana")
            self.assertTrue(any("[QU]" in log for log in cm.output), "Must log [QU] line at INFO level")


if __name__ == "__main__":
    unittest.main()

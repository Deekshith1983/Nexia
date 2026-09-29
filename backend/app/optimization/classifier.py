"""
Deterministic Profile Classifier for Stage 13.

Maps user requests and query understanding directly to ComputationalProfiles
without making unnecessary LLM calls.
"""
import re
from typing import Any, Optional

from app.core.query_understanding import IntentCategory, QueryUnderstanding
from app.optimization.models import ComplexityLevel, ComputationalProfile

_LOW_COMPLEXITY_PATTERNS = [
    r"^(?:hi+|hello+|hey+|howdy|greetings?|yo|sup)\b",
    r"^(?:good\s*(?:morning|afternoon|evening|night))\b",
    r"^(?:thanks?|thank\s+you|thx|ty|cheers?)\b",
    r"^(?:how\s+are\s+you|how's\s+it\s+going|what'?s\s+up)\b",
    r"^(?:bye|goodbye|see\s+ya?|later|ok|okay|cool|got\s+it)\b",
]

_HIGH_COMPLEXITY_KEYWORDS = {
    "compare", "comparison", "difference", "differences", "versus", "vs", "vs.",
    "relationship", "relation", "influence", "influenced", "timeline",
    "comprehensive", "deep dive", "detailed analysis", "investigate", "multi-step",
}


class ProfileClassifier:
    """
    Classifies request complexity deterministically based on rule patterns,
    query structure, and QueryUnderstanding metadata.
    """

    def classify(
        self,
        query: str = "",
        qu: Optional[QueryUnderstanding] = None,
        resolution: Optional[Any] = None,
        plan: Optional[Any] = None,
        relevant_memories: Optional[Any] = None,
        is_multi_entity: bool = False,
    ) -> ComputationalProfile:
        """
        Produce an optimized computational profile for the given request.
        """
        raw_text = query or (qu.original_query if qu else "") or ""
        clean_q = raw_text.strip().lower()

        # Check for Conversational / Low Complexity
        is_conv = (
            getattr(plan, "response_mode", None) == "conversational"
            or any(re.search(pat, clean_q) for pat in _LOW_COMPLEXITY_PATTERNS)
            or (qu and qu.deterministic and not qu.entity)
        )
        if is_conv and len(clean_q.split()) <= 6:
            return ComputationalProfile(
                complexity=ComplexityLevel.LOW,
                llm_budget=1,
                research_budget=0,
                context_budget_chars=4000,
                max_concurrency=1,
                enable_reflection=False,
                enable_decomposition=False,
                enable_navigation=False,
                latency_metadata={"fast_path": "conversational"},
            )

        # 2. Check for High Complexity (Multi-entity / Comparison / Multi-step)
        is_multi = is_multi_entity or (qu and qu.intent == IntentCategory.COMPARISON)
        if is_multi:
            return ComputationalProfile(
                complexity=ComplexityLevel.HIGH,
                llm_budget=4,
                research_budget=6,
                context_budget_chars=16000,
                max_concurrency=2,
                enable_reflection=True,
                enable_decomposition=True,
                enable_navigation=True,
                latency_metadata={"path": "agentic_comparison"},
            )
        words = set(re.findall(r"\b\w+\b", clean_q))
        has_high_keyword = bool(words & _HIGH_COMPLEXITY_KEYWORDS)
        is_comparison = qu and qu.intent == IntentCategory.COMPARISON

        if is_comparison or is_multi_entity or has_high_keyword:
            return ComputationalProfile(
                complexity=ComplexityLevel.HIGH,
                llm_budget=4,
                research_budget=3,
                context_budget_chars=16000,
                max_concurrency=3,
                enable_reflection=True,
                enable_decomposition=True,
                enable_navigation=True,
                latency_metadata={"agent_path": "deep_research"},
            )

        # 3. Default: Medium Complexity (Standard Factual Lookup)
        return ComputationalProfile(
            complexity=ComplexityLevel.MEDIUM,
            llm_budget=2,
            research_budget=1,
            context_budget_chars=10000,
            max_concurrency=2,
            enable_reflection=True,
            enable_decomposition=False,
            enable_navigation=False,
            latency_metadata={"agent_path": "standard_factual"},
        )


profile_classifier = ProfileClassifier()

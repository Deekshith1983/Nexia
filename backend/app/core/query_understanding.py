import json
import logging
import re
from typing import Dict, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError
from app.core.llm_interface import BaseLanguageModel

logger = logging.getLogger("nexia.qu")


class IntentCategory:
    FACTUAL_LOOKUP = "factual_lookup"
    ENTITY_INFORMATION = "entity_information"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    DEFINITION = "definition"
    SUMMARIZATION = "summarization"
    GENERAL_QUESTION = "general_question"


class QueryUnderstanding(BaseModel):
    """Structured representation of a parsed user query."""
    original_query: str
    intent: str = IntentCategory.GENERAL_QUESTION
    entity: Optional[str] = None
    information_needed: Optional[str] = None
    # True if classified deterministically (no LLM call used)
    deterministic: bool = False
    metadata: Dict = Field(default_factory=dict)


# ── Greeting / small-talk fast-path ──────────────────────────────────────────
# Single-word / short greetings
_GREETING_SINGLE = re.compile(
    r"^("
    r"hi+|hello+|hey+|howdy|greetings?|yo|sup|"
    r"good\s*(?:morning|afternoon|evening|night)|"
    r"what'?s\s+up|"
    r"thanks?|thank\s+you|thx|ty|cheers?|"
    r"bye|goodbye|see\s+ya?|later|"
    r"ok(?:ay)?|sure|sounds?\s+good|alright|got\s+it|"
    r"nice|cool|great|awesome|perfect|wonderful"
    r")[\s!?.]*$",
    re.IGNORECASE,
)

# Multi-word conversational openers that start with a greeting word
_GREETING_MULTI = re.compile(
    r"^(?:hi|hello|hey|howdy)[,\s]+(?:how\s+are\s+you|how\s+r\s+u|how\s+do\s+you\s+do|what'?s\s+up|there)[\s!?.]*$",
    re.IGNORECASE,
)

# Standalone "how are you"
_HOW_ARE_YOU = re.compile(
    r"^how\s+are\s+you[\s!?.]*$",
    re.IGNORECASE,
)


def _is_greeting(query: str) -> bool:
    """Return True if query is clearly a greeting or small-talk — no LLM needed."""
    return bool(
        _GREETING_SINGLE.match(query)
        or _GREETING_MULTI.match(query)
        or _HOW_ARE_YOU.match(query)
    )

# ── Knowledge fast-path patterns ──────────────────────────────────────────────
# Each entry: (compiled_regex, intent_category, group_index_of_entity_or_None)
_KNOWLEDGE_PATTERNS: list[Tuple[re.Pattern, str, Optional[int]]] = [
    # "what is/are X", "what's X"
    (re.compile(r"^what\s+(?:is|are|was|were|'s)\s+(.+?)\??$", re.I), IntentCategory.DEFINITION, 1),
    # "who is/was X"
    (re.compile(r"^who\s+(?:is|was|were|'s)\s+(.+?)\??$", re.I), IntentCategory.ENTITY_INFORMATION, 1),
    # "explain X" / "explain how/why X"
    (re.compile(r"^explain\s+(?:(?:how|why|what)\s+)?(.+?)\??$", re.I), IntentCategory.EXPLANATION, 1),
    # "how does/do/did X work" / "how does X ..."
    (re.compile(r"^how\s+(?:does|do|did)\s+(.+?)\s+work\??$", re.I), IntentCategory.EXPLANATION, 1),
    # "how does X ..." (general)
    (re.compile(r"^how\s+(?:does|do|did|is|are|was|were)\s+(.+?)\??$", re.I), IntentCategory.EXPLANATION, 1),
    # "tell me about X"
    (re.compile(r"^(?:can\s+you\s+)?tell\s+me\s+(?:about|more\s+about)\s+(.+?)\??$", re.I), IntentCategory.ENTITY_INFORMATION, 1),
    # "describe X"
    (re.compile(r"^describe\s+(.+?)\??$", re.I), IntentCategory.ENTITY_INFORMATION, 1),
    # "define X" / "definition of X"
    (re.compile(r"^(?:define|give\s+(?:me\s+)?(?:a\s+)?definition\s+of)\s+(.+?)\??$", re.I), IntentCategory.DEFINITION, 1),
    # "when did/was X" / "when is/are X"
    (re.compile(r"^when\s+(?:did|was|were|is|are)\s+(.+?)\??$", re.I), IntentCategory.FACTUAL_LOOKUP, 1),
    # "where is/was X" / "where did X ..."
    (re.compile(r"^where\s+(?:is|was|were|did)\s+(.+?)\??$", re.I), IntentCategory.FACTUAL_LOOKUP, 1),
    # "summarize X" / "give me a summary of X"
    (re.compile(r"^(?:summarize|give\s+(?:me\s+)?(?:a\s+)?summary\s+of)\s+(.+?)\??$", re.I), IntentCategory.SUMMARIZATION, 1),
    # "compare X and Y"
    (re.compile(r"^compare\s+(.+?)\s+and\s+.+\??$", re.I), IntentCategory.COMPARISON, 1),
    # "difference between X and Y"
    (re.compile(r"^(?:what'?s?\s+(?:the\s+)?)?difference\s+between\s+(.+?)\s+and\s+.+\??$", re.I), IntentCategory.COMPARISON, 1),
]

# Stop words that would make a bad entity name (filter entity if it starts with these)
_STOP_WORDS = {"a", "an", "the", "some", "any", "my", "your", "our", "their", "its"}

# Reference pronouns — if the captured entity group starts with one of these, it is NOT
# a real entity name (e.g. "when was he born" → captures "he born" — discard as entity).
# The ContextManager will resolve the pronoun separately.
_REFERENCE_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
}


def _clean_entity(raw: str) -> Optional[str]:
    """Clean up a captured entity string.

    Returns None if the captured group:
    - Is empty or only punctuation
    - Starts with a reference pronoun (the query is a pronoun-reference question,
      not a direct entity question — coreference resolution handles these).
    - Consists only of stop-word articles.
    """
    if not raw:
        return None
    raw = raw.strip().rstrip("?.!,")
    if not raw:
        return None
    words = raw.split()
    # Discard if first word is a reference pronoun — this is a coreference query
    if words[0].lower() in _REFERENCE_PRONOUNS:
        return None
    # Skip if it's just a single stop-word
    if len(words) == 1 and raw.lower() in _STOP_WORDS:
        return None
    # Strip leading article: "the process" → "process"
    if len(words) <= 3 and words[0].lower() in _STOP_WORDS:
        raw = " ".join(words[1:])
    return raw if raw else None


ANALYSIS_PROMPT_TEMPLATE = """You are the Query Understanding module for Nexia Core.
Analyze the user query and extract its structural attributes.

Categories of intent:
- factual_lookup (asking for a specific fact, date, capital, or attribute)
- entity_information (asking who/what an entity is)
- explanation (asking how or why something works)
- comparison (asking for differences or comparisons between things)
- definition (asking for the definition of a term or concept)
- summarization (asking to summarize a topic)
- general_question (general chat or uncategorized query)

User Query: "{query}"

Respond strictly with a valid JSON object in this format (no conversational filler, no markdown):
{{
  "intent": "<category>",
  "entity": "<explicitly mentioned entity or concept, or null>",
  "information_needed": "<specific detail requested, or null>"
}}"""


class QueryAnalyzer:
    """Component responsible for converting natural language queries into structured representations."""

    def _classify_deterministically(self, query: str) -> Optional[QueryUnderstanding]:
        """
        Attempt to classify the query without calling the LLM.

        Returns a QueryUnderstanding if a deterministic rule fires, else None
        (indicating the LLM should be consulted).
        """
        stripped = query.strip()

        # ── Greeting / small-talk fast-path ──────────────────────────────────────────
        if _is_greeting(stripped):
            return QueryUnderstanding(
                original_query=stripped,
                intent=IntentCategory.GENERAL_QUESTION,
                entity=None,
                information_needed=None,
                deterministic=True,
                metadata={"fast_path": "greeting"},
            )

        # ── Knowledge fast-path ───────────────────────────────────────────
        for pattern, intent, entity_group in _KNOWLEDGE_PATTERNS:
            m = pattern.match(stripped)
            if m:
                raw_entity = m.group(entity_group) if entity_group is not None else None
                entity = _clean_entity(raw_entity) if raw_entity else None
                return QueryUnderstanding(
                    original_query=stripped,
                    intent=intent,
                    entity=entity,
                    information_needed=stripped,
                    deterministic=True,
                    metadata={"fast_path": "knowledge", "matched_pattern": pattern.pattern[:40]},
                )

        return None  # No deterministic classification possible — use LLM

    async def analyze(self, query: str, llm: Optional[BaseLanguageModel] = None) -> QueryUnderstanding:
        cleaned_query = query.strip()
        fallback = QueryUnderstanding(
            original_query=cleaned_query,
            intent=IntentCategory.GENERAL_QUESTION,
            entity=None,
            information_needed=None,
            deterministic=False,
        )

        if not cleaned_query:
            return fallback

        # ── Try deterministic classification first (zero LLM cost) ───────
        det = self._classify_deterministically(cleaned_query)
        if det is not None:
            logger.debug(
                "[QU] query=%r  intent=%s  entity=%r  information_needed=%r  deterministic=True  fast_path=%s",
                cleaned_query, det.intent, det.entity, det.information_needed,
                det.metadata.get("fast_path", "-"),
            )
            return det

        # ── Fall back to LLM-based classification ─────────────────────────
        if not llm:
            return fallback

        try:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(query=cleaned_query)
            raw_response = await llm.generate_response(prompt)

            # Extract JSON block from potential markdown fences
            json_str = raw_response.strip()
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1)
            else:
                start = json_str.find("{")
                end = json_str.rfind("}")
                if start != -1 and end != -1:
                    json_str = json_str[start : end + 1]

            parsed_data = json.loads(json_str)

            intent = str(parsed_data.get("intent", IntentCategory.GENERAL_QUESTION)).strip().lower()
            entity = parsed_data.get("entity")
            if entity and str(entity).strip().lower() in ["null", "none", "n/a", ""]:
                entity = None
            elif entity:
                entity = str(entity).strip()

            info_needed = parsed_data.get("information_needed")
            if info_needed and str(info_needed).strip().lower() in ["null", "none", "n/a", ""]:
                info_needed = None
            elif info_needed:
                info_needed = str(info_needed).strip()

            _valid_intents = {
                IntentCategory.FACTUAL_LOOKUP, IntentCategory.ENTITY_INFORMATION,
                IntentCategory.EXPLANATION, IntentCategory.COMPARISON,
                IntentCategory.DEFINITION, IntentCategory.SUMMARIZATION,
                IntentCategory.GENERAL_QUESTION,
            }
            resolved_intent = intent if intent in _valid_intents else IntentCategory.GENERAL_QUESTION

            result = QueryUnderstanding(
                original_query=cleaned_query,
                intent=resolved_intent,
                entity=entity,
                information_needed=info_needed,
                deterministic=False,
            )
            logger.debug(
                "[QU] query=%r  intent=%s  entity=%r  information_needed=%r  deterministic=False",
                cleaned_query, result.intent, result.entity, result.information_needed,
            )
            return result

        except (json.JSONDecodeError, ValidationError, Exception) as e:
            logger.debug("[QU] LLM parse failed (%s). Using fallback.", e)
            return fallback


query_analyzer = QueryAnalyzer()

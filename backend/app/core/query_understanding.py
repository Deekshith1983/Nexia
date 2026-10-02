import json
import logging
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
from app.core.llm_interface import BaseLanguageModel

logger = logging.getLogger("nexia.qu")


class Route(str, Enum):
    SMALLTALK = "smalltalk"
    USER_STATEMENT = "user_statement"
    MEMORY_RECALL = "memory_recall"
    ABOUT_BOT = "about_bot"
    KNOWLEDGE = "knowledge"
    OUT_OF_SCOPE = "out_of_scope"


class Intent(str, Enum):
    OVERVIEW = "overview"
    ATTRIBUTE = "attribute"
    EVENT_FACT = "event_fact"
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    VERIFICATION = "verification"
    LIST = "list"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    GENERAL_QUESTION = "general_question"
    SMALLTALK = "smalltalk"
    USER_STATEMENT = "user_statement"
    MEMORY_RECALL = "memory_recall"
    ABOUT_BOT = "about_bot"
    OUT_OF_SCOPE = "out_of_scope"
    # Backward compatibility aliases
    FACTUAL_LOOKUP = "factual_lookup"
    ENTITY_INFORMATION = "entity_information"


# Backward compatibility alias
IntentCategory = Intent


class AnswerType(str, Enum):
    FACT = "FACT"
    DEFINITION = "DEFINITION"
    PERSON = "PERSON"
    LIST = "LIST"
    COMPARISON = "COMPARISON"
    EXPLANATION = "EXPLANATION"
    CAUSAL = "CAUSAL"
    RELATION = "RELATION"
    BOOLEAN = "BOOLEAN"
    DATE = "DATE"
    PLACE = "PLACE"
    NUMBER = "NUMBER"


class AnswerContract(BaseModel):
    """
    Lightweight answer contract defining the expected shape and fields for the response.
    """
    answer_type: Union[AnswerType, str] = AnswerType.FACT
    target_entity: Optional[str] = None
    requested_relation: Optional[str] = None
    requested_scope: Optional[str] = None
    concept_a: Optional[str] = None
    concept_b: Optional[str] = None
    required_fields: List[str] = Field(default_factory=list)
    required_attributes: List[str] = Field(default_factory=list)
    is_multi_attribute: bool = False
    ordering: Optional[str] = None
    unverifiable_attribute: bool = False
    negative_or_exclusion: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def expected_type(self) -> Union[AnswerType, str]:
        return self.answer_type

    @property
    def negative_aspect(self) -> bool:
        return self.negative_or_exclusion


class QueryUnderstanding(BaseModel):
    """Structured representation of a parsed user query."""
    original_query: str
    route: Union[Route, str] = Route.KNOWLEDGE
    intent: Union[Intent, str] = Intent.OVERVIEW
    entities: List[str] = Field(default_factory=list)
    attribute: Optional[str] = None
    answer_type: Union[AnswerType, str] = AnswerType.FACT
    sub_queries: List[str] = Field(default_factory=list)
    needs_context: bool = False
    ambiguous: bool = False
    constraints: Dict[str, Any] = Field(default_factory=dict)
    source: Literal["rule", "llm", "fallback"] = "rule"
    confidence: float = 1.0
    information_needed: Optional[str] = None
    contract: Optional[AnswerContract] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ── Backward-compatible properties ─────────────────────────────────────────
    @property
    def entity(self) -> Optional[str]:
        return self.entities[0] if self.entities else None

    @property
    def deterministic(self) -> bool:
        return self.source == "rule"

    @property
    def answer_contract(self) -> Optional[AnswerContract]:
        if self.contract is not None:
            return self.contract
        return AnswerContract(
            answer_type=self.answer_type,
            target_entity=self.entity,
            requested_relation=self.attribute,
            concept_a=self.metadata.get("concept_a"),
            concept_b=self.metadata.get("concept_b"),
            required_fields=[self.entity] if self.entity else [],
            metadata=self.metadata,
        )


# Reference pronouns — if the captured entity group starts with or is one of these,
# it is a context reference (not a standalone entity name).
_REFERENCE_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
}

_STOP_WORDS = {"a", "an", "the", "some", "any", "my", "your", "our", "their", "its"}

# Conversational lead-in phrases to safely strip without removing honorifics/titles
_LEAD_IN_PREFIXES: List[str] = [
    r"(?:can\s+you\s+)?tell\s+me\s+(?:all\s+)?about",
    r"(?:can\s+you\s+)?tell\s+me\s+more\s+about",
    r"give\s+me\s+(?:a\s+)?(?:brief\s+)?(?:summary|overview|info(?:rmation)?)\s+(?:on|about|of)",
    r"what\s+(?:do|did)\s+you\s+know\s+about",
    r"i\s+want\s+to\s+know\s+about",
    r"explain\s+(?:(?:how|why|what)\s+)?",
    r"describe",
    r"define",
    r"give\s+(?:me\s+)?(?:a\s+)?definition\s+of",
    r"information\s+(?:on|about|regarding)",
    r"info\s+(?:on|about|regarding)",
    r"tell\s+me\s+(?:the\s+)?history\s+(?:of|about)",
    r"history\s+(?:of|about)",
    r"tell\s+me\s+something\s+about",
    r"tell\s+me",
    r"about",
]

_LEAD_IN_PREFIX_RE = re.compile(
    r"^(?:" + r"|".join(_LEAD_IN_PREFIXES) + r")\s*",
    re.IGNORECASE,
)

_TRAILING_MODIFIERS_RE = re.compile(
    r"\s+(?:in\s+detail|in\s+depth|in\s+brief|briefly|in\s+short|please|now|for\s+me|if\s+you\s+can|if\s+possible)[.!?]*$",
    re.IGNORECASE,
)


def _extract_constraints(query: str) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    lower_q = query.lower()
    if re.search(r"\b(?:in\s+detail|in\s+depth|detailed|in-depth)\b", lower_q):
        constraints["length"] = "detailed"
    elif re.search(r"\b(?:in\s+brief|briefly|in\s+short|short\s+summary|brief\s+summary)\b", lower_q):
        constraints["length"] = "brief"

    m_lines = re.search(r"\bin\s+(\d+)\s+(?:lines|sentences|bullet\s+points|words)\b", lower_q)
    if m_lines:
        constraints["format"] = m_lines.group(0)
    elif "step by step" in lower_q:
        constraints["format"] = "step_by_step"

    # Ordering constraint extraction (e.g. "in chronological order", "timeline", "in order")
    if re.search(r"\b(?:chronological(?:ly)?|in\s+chronological\s+order|timeline|time\s+order|in\s+order)\b", lower_q):
        constraints["ordering"] = "chronological"

    # Temporal constraint extraction (e.g. "after Tipu Sultan", "during Hoysala dynasty", "in Dvapara Yuga")
    m_temp = re.search(
        r"\b(after|before|during|in)\s+([A-Za-z0-9\s]+?)(?:\s+(?:kingdom|dynasty|empire|era|age|period|century|yuga|sultan))?(?:\s*\?|\s*$)",
        query,
        re.I,
    )
    if m_temp:
        prep = m_temp.group(1).lower()
        target = m_temp.group(0).strip().rstrip("?.!")
        if not re.search(r"\b(?:detail|depth|brief|short|lines|sentences|words|english|french)\b", target, re.I):
            constraints["temporal"] = target

    # Scope constraint extraction (e.g. "independent India", "ancient Greece")
    if re.search(r"\bindependent\s+india\b", lower_q):
        constraints["scope"] = "independent"
    elif re.search(r"\b(?:ancient|medieval|modern|independent)\b", lower_q):
        m_scope = re.search(r"\b(ancient|medieval|modern|independent)\b", lower_q)
        if m_scope:
            constraints["scope"] = m_scope.group(1).lower()

    return constraints


def clean_entity_for_search(entity: str) -> str:
    """
    Safely clean an entity string by stripping conversational lead-in phrases and trailing modifiers.
    DO NOT strip title words (Lord, Sri, Saint, The Great) and DO NOT strip leading 'the' blindly.
    """
    if not entity:
        return ""
    text = entity.strip().rstrip("?.!,")
    prev = None
    while prev != text:
        prev = text
        text = _LEAD_IN_PREFIX_RE.sub("", text).strip()
    text = _TRAILING_MODIFIERS_RE.sub("", text).strip()
    return text if text else entity.strip()


_LOWER_WORDS = {"of", "the", "in", "on", "at", "and", "for", "to", "a", "an", "de", "von", "van"}


def _format_entity_casing(raw: str) -> str:
    """Format entity casing properly without breaking real titles."""
    cleaned = clean_entity_for_search(raw)
    if not cleaned:
        return ""
    # Strip leading article 'the ' if present
    if cleaned.lower().startswith("the "):
        cleaned = cleaned[4:].strip()

    # Check for historical/title prefixes like "great king ashoka" or "ruler who ruled during hoysala dynasty"
    m_ruler = re.match(r"^(?:the\s+)?(?:ruler|rulers|king|kings|monarch|monarchs)\s+(?:who\s+ruled\s+(?:during|in)|of|in)\s+(.+)$", cleaned, re.I)
    if m_ruler:
        cleaned = m_ruler.group(1).strip()
    else:
        m_king = re.match(r"^(?:great\s+)?(?:king|emperor|queen|ruler|president|prime\s+minister)\s+(.+)$", cleaned, re.I)
        if m_king:
            cleaned = m_king.group(1).strip()

    # If user already entered mixed case with uppercase letters (e.g. "Sri Lanka", "Lord Byron"),
    # preserve user casing
    if any(c.isupper() for c in cleaned):
        return cleaned

    # Title-case each word respecting connecting lower words
    words = cleaned.split()
    formatted = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() in _LOWER_WORDS:
            formatted.append(w.lower())
        else:
            formatted.append(w.capitalize())
    return " ".join(formatted)


def _clean_relation_subject(raw: str) -> str:
    """Clean the target subject of a relation/attribute query (e.g. 'lord rama' -> 'Rama')."""
    # Strip leading question openers if captured: "who is rama" -> "rama"
    s = re.sub(r"^(?:who|what|which|where)\s+(?:is|was|were|are|'s)\s+", "", raw.strip(), flags=re.I)
    # Strip trailing attribute phrases: e.g. ", their reign periods, major contributions..."
    s = re.sub(r"(?:,\s*(?:their|and|with)?|\s+(?:and|with)\s+(?:their)?)\s+(?:reign|contribution|battle|war|achievement|period).*", "", s, flags=re.I).strip()
    ent = _format_entity_casing(s)
    # If starting with 'lord / saint / shri / sri ' in a relation target (e.g. 'father of lord rama'), strip for cleaner Wikipedia matching
    m_hon = re.match(r"^(?:lord|saint|shri|sri)\s+(.+)$", ent, re.I)
    if m_hon:
        return m_hon.group(1).strip()
    return ent


def extract_answer_contract(query: str, intent: Union[Intent, str], entity: Optional[str]) -> AnswerContract:
    """Extract AnswerContract for backwards compatibility."""
    lower_q = query.lower().strip()
    
    # Check for unverifiable personal preferences
    is_unverifiable = bool(re.search(r"\bfavou?rite\s+(?:colou?r|food|dish|toy|game|movie|song|pet|hobby|childhood\s+toy)\b", lower_q))
    
    if re.search(r"\b(?:why\s+was|why\s+is|why\s+were|why\s+did|how\s+was|how\s+is)\b.*?\b(?:not|never|neither|excluded|excluded\s+from|refused)\b", lower_q):
        required = []
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query):
            if match.lower() not in ("why", "was", "not", "considered", "wikipedia", "what", "who"):
                required.append(match)
        return AnswerContract(
            answer_type=AnswerType.CAUSAL,
            target_entity=entity,
            requested_relation="exclusion_or_negative_cause",
            required_fields=required,
            negative_or_exclusion=True,
            metadata={"pattern": "negative_causal"},
        )
    if (
        re.search(r"\b(?:list(?:\s+out)?|all|names\s+of)\s+(?:the\s+)?(?:major\s+)?(?:rulers|kings|monarchs|emperors|dynasties|presidents|prime\s+ministers|members|works|books)\b", lower_q)
        or re.search(r"\bwho\s+were\s+(?:the\s+)?(?:major\s+)?(?:rulers|kings|monarchs|emperors|prime\s+ministers|presidents)\b", lower_q)
        or "rulers of" in lower_q
        or "kings of" in lower_q
        or "prime ministers of" in lower_q
    ):
        relation = "prime ministers" if "prime minister" in lower_q else ("rulers" if any(w in lower_q for w in ["ruler", "rulers", "king", "kings", "monarch", "emperor"]) else "list")
        
        req_attrs = []
        if re.search(r"\b(?:reign|period|periods|dates|tenure|tenures)\b", lower_q):
            req_attrs.append("reign periods")
        if re.search(r"\b(?:contribution|contributions|achievement|achievements|reform|reforms|work|works)\b", lower_q):
            req_attrs.append("major contributions")
        if re.search(r"\b(?:battle|battles|war|wars|campaign|campaigns)\b", lower_q):
            req_attrs.append("important battles")
        if re.search(r"\b(?:event|events|incident|incidents)\b", lower_q):
            req_attrs.append("major events")

        is_multi = len(req_attrs) > 0
        ordering = "chronological" if re.search(r"\b(?:chronological|in\s+order|timeline)\b", lower_q) else None

        return AnswerContract(
            answer_type=AnswerType.LIST,
            target_entity=entity,
            requested_relation=relation,
            required_fields=[entity] if entity else [],
            required_attributes=req_attrs,
            is_multi_attribute=is_multi,
            ordering=ordering,
            metadata={"pattern": "list_extraction", "multi_attribute": is_multi, "attributes": req_attrs},
        )
    if re.search(r"\b(?:who\s+wrote|who\s+authored|author\s+of|writer\s+of|composer\s+of)\b", lower_q):
        return AnswerContract(
            answer_type=AnswerType.PERSON,
            target_entity=entity,
            requested_relation="author",
            required_fields=["author", entity] if entity else ["author"],
            metadata={"pattern": "authorship"},
        )
    if re.search(r"\b(?:who\s+(?:is|was|were)\s+(?:the\s+)?(?:founder|founders|creator|creators|inventor|inventors|discoverer|discoverers|developer|developers|builder|architect)\s+of|who\s+(?:founded|created|invented|discovered|developed|built)|founder\s+of|creator\s+of)\b", lower_q):
        return AnswerContract(
            answer_type=AnswerType.PERSON,
            target_entity=entity,
            requested_relation="founder",
            required_fields=["founder", entity] if entity else ["founder"],
            metadata={"pattern": "founder_creator"},
        )
    kin_match = re.search(r"\b(?:father|mother|son|daughter|brother|sister|parent|parents|wife|husband|spouse)\s+of\b", lower_q)
    if kin_match or re.search(r"\bwho\s+was\s+(?:the\s+)?(?:father|mother|son|daughter|wife|husband)\b", lower_q):
        kin_rel = kin_match.group(0).split()[0] if kin_match else "relation"
        return AnswerContract(
            answer_type=AnswerType.PERSON,
            target_entity=entity,
            requested_relation=kin_rel,
            required_fields=[kin_rel, entity] if entity else [kin_rel],
            metadata={"pattern": "kinship"},
        )
    if intent in (Intent.COMPARISON, "comparison") or any(w in lower_q for w in ["compare", "difference", " vs ", "versus"]):
        return AnswerContract(
            answer_type=AnswerType.COMPARISON,
            target_entity=entity,
            metadata={"pattern": "comparison"},
        )
    if intent in (Intent.DEFINITION, "definition"):
        return AnswerContract(
            answer_type=AnswerType.DEFINITION,
            target_entity=entity,
            metadata={"pattern": "definition"},
        )
    if intent in (Intent.EXPLANATION, "explanation"):
        is_why = lower_q.startswith("why ") or " why " in lower_q
        return AnswerContract(
            answer_type=AnswerType.CAUSAL if is_why else AnswerType.EXPLANATION,
            target_entity=entity,
            metadata={"pattern": "explanation"},
        )
    return AnswerContract(
        answer_type=AnswerType.FACT,
        target_entity=entity,
        unverifiable_attribute=is_unverifiable,
        metadata={"pattern": "default_fact", "unverifiable": is_unverifiable},
    )


ANALYSIS_PROMPT_TEMPLATE = """You are the Query Understanding module for Nexia (Wikipedia Conversational Assistant).
Analyze the user query and produce a strict JSON output.

Routes:
- smalltalk (greetings, farewells, casual chat)
- user_statement (user sharing personal info, e.g. "my name is...", "I live in...")
- memory_recall (user asking what the bot knows about the user)
- about_bot (asking about Nexia's identity or capabilities)
- knowledge (factual or explanatory queries)
- out_of_scope (math, weather, live stocks, non-retrievable requests)

Intents:
- overview, attribute, event_fact, definition, explanation, verification, list, comparison, general_question

User Query: "{query}"

Respond strictly with a valid JSON object matching this schema (no markdown, no other text):
{{
  "route": "<smalltalk|user_statement|memory_recall|about_bot|knowledge|out_of_scope>",
  "intent": "<overview|attribute|event_fact|definition|explanation|verification|list|comparison|general_question>",
  "entities": ["<entity1>", "<entity2>"],
  "attribute": "<attribute or relation name, or null>",
  "answer_type": "<FACT|DEFINITION|PERSON|LIST|COMPARISON|EXPLANATION|CAUSAL|RELATION|BOOLEAN|DATE|PLACE|NUMBER>",
  "needs_context": false,
  "ambiguous": false,
  "constraints": {{}}
}}"""


class QueryAnalyzer:
    """Component responsible for converting natural language queries into structured representations."""

    def _log_qu(self, qu: QueryUnderstanding) -> None:
        logger.info(
            "[QU] route=%s intent=%s entities=%s attribute=%s source=%s confidence=%.2f",
            qu.route,
            qu.intent,
            qu.entities,
            qu.attribute or "-",
            qu.source,
            qu.confidence,
        )

    def _classify_deterministically(self, query: str) -> Optional[QueryUnderstanding]:
        """
        Attempt to classify the query without calling the LLM.
        Applies strict routing before any entity extraction.
        """
        stripped = query.strip()
        lower_q = stripped.lower().rstrip("?.!")
        constraints = _extract_constraints(stripped)

        # ── 1. Out of Scope (Math, calculations, weather, time) ──────────────
        if (
            re.search(r"^(?:what\s+(?:is|'s)\s+)?\d+\s*[\+\-\*\/\^]\s*\d+", lower_q)
            or re.search(r"\b(?:calculate|what\s+time\s+is\s+it|current\s+time|weather\s+today|weather\s+in)\b", lower_q)
        ):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.OUT_OF_SCOPE,
                intent=Intent.OUT_OF_SCOPE,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                constraints=constraints,
                source="rule",
                confidence=1.0,
                metadata={"reason": "out_of_scope_capability"},
            )

        # ── 2. Smalltalk / Greeting / Farewell ────────────────────────────────
        if (
            re.match(r"^(?:hi+|hello+|hey+|howdy|yo|greetings?|good\s+(?:morning|afternoon|evening|night)|what'?s\s+up|thanks?|thank\s+you|thx|ty|bye|goodbye|see\s+ya?|how\s+are\s+you|how\s+do\s+you\s+do)[\s!?.]*$", lower_q)
        ):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.SMALLTALK,
                intent=Intent.SMALLTALK,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                constraints=constraints,
                source="rule",
                confidence=1.0,
            )

        # ── 3. User Statement (User introducing themselves / sharing facts) ──
        if (
            re.search(r"^(?:hi[,\s]+)?(?:i\s+am|i'm|my\s+name\s+is|i\s+live\s+in|i\s+work\s+at)\b", lower_q)
        ):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.USER_STATEMENT,
                intent=Intent.USER_STATEMENT,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                constraints=constraints,
                source="rule",
                confidence=1.0,
            )

        # ── 4. Memory Recall (User asking bot to recall user info) ────────────
        if (
            re.search(r"\b(?:what\s+(?:is|'s)\s+my\s+name|who\s+am\s+i|remember\s+(?:what\s+is\s+)?my\s+name|remember\s+me|what\s+do\s+you\s+know\s+about\s+me|what\s+did\s+i\s+tell\s+you|tell\s+me\s+what\s+you\s+know\s+about\s+me|have\s+you\s+(?:saved|stored)\s+my)\b", lower_q)
        ):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.MEMORY_RECALL,
                intent=Intent.MEMORY_RECALL,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                constraints=constraints,
                source="rule",
                confidence=1.0,
            )

        # ── 5. About Bot (Questions asking about Nexia) ───────────────────────
        if (
            re.search(r"^(?:who\s+are\s+you|what\s+(?:is|'s)\s+your\s+name|what\s+can\s+you\s+do|tell\s+me\s+about\s+yourself|who\s+(?:made|created)\s+you)\b", lower_q)
        ):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.ABOUT_BOT,
                intent=Intent.ABOUT_BOT,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                constraints=constraints,
                source="rule",
                confidence=1.0,
            )

        # ── 6. Needs Context (Follow-up or pronoun queries) ───────────────────
        if re.match(r"^(?:tell\s+me\s+more(?:\s+about\s+it)?|more\s+info|and\s+his\s+wife|and\s+her\s+husband|why\??|what\s+about\s+it\??)$", lower_q):
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.OVERVIEW,
                entities=[],
                attribute=None,
                answer_type=AnswerType.FACT,
                needs_context=True,
                constraints=constraints,
                source="rule",
                confidence=0.9,
            )

        # ── 7. Multi-Part Queries ("who was Ashoka and when did he rule") ─────
        multi_m = re.match(r"^who\s+(?:was|is)\s+(.+?)\s+and\s+(when\s+did\s+(?:he|she|they)\s+.+?)\??$", stripped, re.I)
        if multi_m:
            ent = _format_entity_casing(multi_m.group(1))
            sub1 = f"who was {ent}"
            sub2 = multi_m.group(2).strip()
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.OVERVIEW,
                entities=[ent] if ent else [],
                attribute=None,
                answer_type=AnswerType.PERSON,
                sub_queries=[sub1, sub2],
                constraints=constraints,
                source="rule",
                confidence=1.0,
            )

        # ── 8. Comparison (Checked BEFORE generic "what is") ──────────────────
        comp_m1 = re.match(r"^(?:(?:what(?:'s|\s+is)\s+(?:the\s+)?)?difference\s+between)\s+(.+?)\s+and\s+(.+?)\??$", stripped, re.I)
        comp_m2 = re.match(r"^compare\s+(.+?)\s+and\s+(.+?)\??$", stripped, re.I)
        comp_m3 = re.match(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)\??$", stripped, re.I)
        if comp_m1 or comp_m2 or comp_m3:
            cm = comp_m1 or comp_m2 or comp_m3
            e1 = _format_entity_casing(cm.group(1))
            e2 = _format_entity_casing(cm.group(2))
            ents = [e for e in [e1, e2] if e]
            contract = extract_answer_contract(stripped, Intent.COMPARISON, ents[0] if ents else None)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.COMPARISON,
                entities=ents,
                attribute=None,
                answer_type=AnswerType.COMPARISON,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        # ── 9. Date / Event Queries ("when did Gandhi die", "when was Taj Mahal built") ──
        # Pronoun check first: "when was he born"
        pronoun_date_m = re.match(r"^when\s+(?:did|was|were)\s+(he|she|it|they)\s+(born|die|pass\s+away|rule|reign)\??$", lower_q)
        if pronoun_date_m:
            verb = pronoun_date_m.group(2)
            attr = "birth_date" if verb == "born" else ("death_date" if verb in ("die", "pass away") else "date")
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EVENT_FACT,
                entities=[],
                attribute=attr,
                answer_type=AnswerType.DATE,
                needs_context=True,
                constraints=constraints,
                source="rule",
                confidence=0.95,
            )

        when_born_m = re.match(r"^when\s+(?:was|is)\s+(.+?)\s+born\??$", stripped, re.I)
        if when_born_m:
            raw_ent = when_born_m.group(1).strip()
            if raw_ent.lower().split()[0] not in _REFERENCE_PRONOUNS:
                ent = _format_entity_casing(raw_ent)
                contract = extract_answer_contract(stripped, Intent.EVENT_FACT, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.EVENT_FACT,
                    entities=[ent] if ent else [],
                    attribute="birth_date",
                    answer_type=AnswerType.DATE,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                )

        when_die_m = re.match(r"^when\s+did\s+(.+?)\s+(?:die|pass\s+away)\??$", stripped, re.I)
        if when_die_m:
            ent = _format_entity_casing(when_die_m.group(1))
            contract = extract_answer_contract(stripped, Intent.EVENT_FACT, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EVENT_FACT,
                entities=[ent] if ent else [],
                attribute="death_date",
                answer_type=AnswerType.DATE,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        when_built_m = re.match(r"^when\s+was\s+(?:the\s+)?(.+?)\s+(?:built|constructed|founded|created|established)\??$", stripped, re.I)
        if when_built_m:
            ent = _format_entity_casing(when_built_m.group(1))
            contract = extract_answer_contract(stripped, Intent.EVENT_FACT, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EVENT_FACT,
                entities=[ent] if ent else [],
                attribute="construction_date",
                answer_type=AnswerType.DATE,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        when_gen_m = re.match(r"^when\s+(?:did|was|were|is|are)\s+(.+?)\??$", stripped, re.I)
        if when_gen_m:
            raw_ent = when_gen_m.group(1)
            if raw_ent.lower().split()[0] in _REFERENCE_PRONOUNS:
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.EVENT_FACT,
                    entities=[],
                    attribute="date",
                    answer_type=AnswerType.DATE,
                    needs_context=True,
                    constraints=constraints,
                    source="rule",
                    confidence=0.95,
                )
            else:
                ent = _format_entity_casing(raw_ent)
                contract = extract_answer_contract(stripped, Intent.EVENT_FACT, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.EVENT_FACT,
                    entities=[ent] if ent else [],
                    attribute="date",
                    answer_type=AnswerType.DATE,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=0.9,
                )

        # ── 10. Attribute of Entity ("capital of France", "population of India", "father of Rama") ──
        # Pronoun attribute queries (e.g. "What were his major contributions?", "What did he achieve?")
        pronoun_attr_m = re.match(r"^(?:what\s+(?:were|was|is|are)\s+(?:his|her|their)\s+(.+?)|what\s+did\s+(?:he|she|they)\s+(.+?))\??$", stripped, re.I)
        if pronoun_attr_m:
            raw_attr = (pronoun_attr_m.group(1) or pronoun_attr_m.group(2)).strip().rstrip("?.!")
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.ATTRIBUTE,
                entities=[],
                attribute=raw_attr.lower(),
                answer_type=AnswerType.FACT,
                needs_context=True,
                constraints=constraints,
                source="rule",
                confidence=0.95,
            )

        # Successor / Multi-hop ("who succeeded Shivaji Maharaj and what major events occurred during his reign")
        successor_m = re.match(
            r"^who\s+(?:succeeded|took\s+over\s+after|ruled\s+after)\s+(.+?)(?:\s+and\s+(.+?))?\??$",
            stripped,
            re.I,
        )
        if successor_m:
            raw_ent = successor_m.group(1).strip()
            sec_query = successor_m.group(2).strip() if successor_m.group(2) else ""
            ent = _clean_relation_subject(raw_ent)
            sub_q1 = f"who succeeded {ent}"
            sub_q2 = sec_query if sec_query else f"major events during reign of successor of {ent}"
            contract = AnswerContract(
                answer_type=AnswerType.RELATION,
                target_entity=ent,
                requested_relation="successor",
                required_fields=[ent, "successor"],
                metadata={"pattern": "successor_multihop", "secondary_query": sec_query},
            )
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.ATTRIBUTE,
                entities=[ent] if ent else [],
                attribute="successor",
                answer_type=AnswerType.PERSON,
                sub_queries=[sub_q1, sub_q2],
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
                metadata={"multihop": True, "secondary_query": sec_query},
            )

        # Multi-attribute list: e.g. "List the major rulers of the Maratha Empire, their reign periods, major contributions, and important battles."
        multi_attr_list_m = re.match(
            r"^(?:list\s+(?:out\s+)?(?:the\s+)?(?:all\s+)?(?:major\s+)?(?:names\s+of\s+)?)?"
            r"(rulers|kings|monarchs|emperors|prime\s+ministers|presidents|dynasties|members)\s+(?:of|in)\s+(?:the\s+)?"
            r"([A-Za-z0-9\s]+?)"
            r"(?:,\s*(?:their\s+)?|(?:\s+and\s+(?:their\s+)?))"
            r"(.+?)\??$",
            stripped,
            re.I,
        )
        if multi_attr_list_m:
            raw_cat = multi_attr_list_m.group(1).lower()
            raw_ent = multi_attr_list_m.group(2).strip()
            raw_attrs_text = multi_attr_list_m.group(3).strip()
            ent = _clean_relation_subject(raw_ent)
            
            # Extract structured attribute bundle
            req_attrs = []
            if re.search(r"\b(?:reign|period|periods|dates|tenure|tenures)\b", raw_attrs_text, re.I):
                req_attrs.append("reign periods")
            if re.search(r"\b(?:contribution|contributions|achievement|achievements|reform|reforms|work|works)\b", raw_attrs_text, re.I):
                req_attrs.append("major contributions")
            if re.search(r"\b(?:battle|battles|war|wars|campaign|campaigns)\b", raw_attrs_text, re.I):
                req_attrs.append("important battles")
            if re.search(r"\b(?:event|events|incident|incidents)\b", raw_attrs_text, re.I):
                req_attrs.append("major events")
            if not req_attrs:
                req_attrs = [raw_attrs_text]
                
            contract = AnswerContract(
                answer_type=AnswerType.LIST,
                target_entity=ent,
                requested_relation=raw_cat,
                required_fields=[ent],
                required_attributes=req_attrs,
                is_multi_attribute=True,
                ordering=constraints.get("ordering"),
                metadata={"pattern": "multi_attribute_list", "attributes": req_attrs},
            )
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.LIST,
                entities=[ent] if ent else [],
                attribute=raw_cat,
                answer_type=AnswerType.LIST,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
                metadata={"multi_attribute": True, "required_attributes": req_attrs},
            )

        # General possessive: "<entity>'s <attribute>" (e.g. "what was Shivaji Maharaj's favorite color?", "who is rama's father")
        poss_gen_m = re.match(
            r"^(?:(?:(?:what|who|which|where|tell\s+me)(?:'s|\s+is|\s+was|'re|\s+were)?\s+)?(?:the\s+)?)?([A-Za-z0-9\s]+?)'s\s+(.+?)\??$",
            stripped,
            re.I,
        )
        if poss_gen_m:
            raw_ent = poss_gen_m.group(1).strip()
            raw_attr = poss_gen_m.group(2).strip()
            first_w = raw_ent.lower().split()[0] if raw_ent else ""
            _skip_poss = _REFERENCE_PRONOUNS | {"what", "who", "where", "when", "why", "how", "it", "that", "there", "here", "let", "which"}
            if first_w not in _skip_poss and len(raw_ent) >= 2:
                ent = _clean_relation_subject(raw_ent)
                attr = raw_attr.lower().rstrip("?.!")
                is_unverifiable = bool(re.search(r"\bfavou?rite\s+(?:colou?r|food|dish|toy|game|movie|song|pet|hobby|childhood\s+toy)\b", attr))
                contract = AnswerContract(
                    answer_type=AnswerType.PERSON if attr in ("father", "mother", "son", "daughter", "wife", "husband", "author", "founder", "creator", "inventor") else AnswerType.FACT,
                    target_entity=ent,
                    requested_relation=attr,
                    required_fields=[ent, attr],
                    unverifiable_attribute=is_unverifiable,
                    metadata={"pattern": "possessive_attribute", "unverifiable": is_unverifiable},
                )
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.ATTRIBUTE,
                    entities=[ent] if ent else [],
                    attribute=attr,
                    answer_type=contract.answer_type,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                    metadata={"unverifiable_attribute": is_unverifiable},
                )

        # "who was/is father of lord Rama" / "who wrote X" / "who invented X"
        who_invented_m = re.match(r"^who\s+(?:invented|founded|created|discovered|developed|built)\s+(.+?)\??$", stripped, re.I)
        if who_invented_m:
            ent = _clean_relation_subject(who_invented_m.group(1))
            contract = extract_answer_contract(stripped, Intent.ATTRIBUTE, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.ATTRIBUTE,
                entities=[ent] if ent else [],
                attribute="founder",
                answer_type=AnswerType.PERSON,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        who_wrote_m = re.match(r"^who\s+(?:wrote|authored)\s+(.+?)\??$", stripped, re.I)
        if who_wrote_m:
            ent = _clean_relation_subject(who_wrote_m.group(1))
            contract = extract_answer_contract(stripped, Intent.ATTRIBUTE, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.ATTRIBUTE,
                entities=[ent] if ent else [],
                attribute="author",
                answer_type=AnswerType.PERSON,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        # "what is the capital of France" / "what's the population of India" / "who was father of lord Rama"
        # "list out prime ministers of independent india" / "list the rulers of mysore kingdom after tippu sultan"
        attr_of_m = re.search(
            r"^(?:(?:what(?:'s|\s+is)\s+(?:the\s+)?|who\s+(?:was|were|is)\s+(?:the\s+)?|tell\s+me\s+(?:the\s+)?|list\s+(?:out\s+)?(?:the\s+)?(?:all\s+)?(?:major\s+)?(?:names\s+of\s+)?))?"
            r"(capital|population|father|mother|son|daughter|wife|husband|founder|author|history|rulers|kings|monarchs|emperors|prime\s+ministers|presidents|dynasties|members|works|books)\s+(?:of|in|about)\s+(?:the\s+)?"
            r"(.+?)\??$",
            stripped,
            re.I,
        )
        if attr_of_m:
            raw_attr = attr_of_m.group(1).lower()
            raw_remainder = attr_of_m.group(2).strip()

            # Separate trailing temporal constraint: e.g. "mysore kingdom after tippu sultan"
            temp_match = re.search(r"\s+(after|before|during|in)\s+([A-Za-z0-9\s]+?)$", raw_remainder, re.I)
            if temp_match:
                t_prep = temp_match.group(1).lower()
                t_target = temp_match.group(2).strip()
                if not re.search(r"^(?:detail|depth|brief|short|lines|sentences|words|english)\b", t_target, re.I):
                    constraints["temporal"] = f"{t_prep} {_clean_relation_subject(t_target)}"
                    raw_remainder = raw_remainder[:temp_match.start()].strip()

            # Separate leading scope qualifier: e.g. "independent india" -> entity="India", scope="independent"
            scope_match = re.match(r"^(independent|ancient|medieval|modern)\s+(.+)$", raw_remainder, re.I)
            if scope_match:
                constraints["scope"] = scope_match.group(1).lower()
                raw_remainder = scope_match.group(2).strip()

            ent = _clean_relation_subject(raw_remainder)
            attr = "prime ministers" if "prime" in raw_attr else (
                "rulers" if raw_attr in ("rulers", "kings", "monarchs", "emperors", "dynasties") else raw_attr
            )
            is_list = attr in ("rulers", "prime ministers", "presidents", "members", "works", "books") or "list" in stripped.lower()
            ans_type = AnswerType.LIST if is_list else (
                AnswerType.PLACE if attr == "capital" else (
                    AnswerType.NUMBER if attr == "population" else (
                        AnswerType.PERSON if attr in ("father", "mother", "son", "daughter", "wife", "husband", "founder", "author") else (
                            AnswerType.LIST if attr == "rulers" else AnswerType.FACT
                        )
                    )
                )
            )
            intent = Intent.LIST if is_list else Intent.ATTRIBUTE
            contract = extract_answer_contract(stripped, intent, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=intent,
                entities=[ent] if ent else [],
                attribute=attr,
                answer_type=ans_type,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        # ── 11. Verification / Yes-No ("was Ashoka a Buddhist") ───────────────
        verif_m = re.match(r"^(?:was|is|did|were|can)\s+([A-Za-z0-9\s]+?)\s+(?:a\s+|an\s+)?([A-Za-z0-9\s]+?)\??$", stripped, re.I)
        if verif_m:
            raw_sub = verif_m.group(1)
            if raw_sub.lower().split()[0] not in _REFERENCE_PRONOUNS:
                ent = _format_entity_casing(raw_sub)
                contract = extract_answer_contract(stripped, Intent.VERIFICATION, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.VERIFICATION,
                    entities=[ent] if ent else [],
                    attribute=None,
                    answer_type=AnswerType.BOOLEAN,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                )

        # ── 11.5. Conceptual Distinction ("why is it Vijayanagara Empire and not Vijayanagara dynasty") ──
        # Matches: "why is it X and not Y", "why is X called A and not B", "why X rather than Y", "difference between X and Y"
        dist_m = re.match(
            r"^(?:why\s+(?:is|was)\s+(?:it\s+)?(?:called\s+)?(.+?)\s+(?:and\s+not|rather\s+than|instead\s+of)\s+(.+?)\??$)",
            stripped,
            re.I,
        )
        if dist_m:
            term1 = dist_m.group(1).strip()
            term2 = dist_m.group(2).strip()
            # Extract common root entity if both terms share it, e.g. "Vijayanagara Empire" vs "Vijayanagara dynasty" -> "Vijayanagara Empire"
            ent = _format_entity_casing(term1)
            # Determine concepts
            concept_a = "empire" if "empire" in term1.lower() else term1
            concept_b = "dynasty" if "dynasty" in term2.lower() else term2
            contract = AnswerContract(
                answer_type=AnswerType.EXPLANATION,
                target_entity=ent,
                requested_relation="conceptual_distinction",
                concept_a=concept_a,
                concept_b=concept_b,
                required_fields=[concept_a, concept_b],
                metadata={
                    "pattern": "conceptual_distinction",
                    "concept_a": concept_a,
                    "concept_b": concept_b,
                    "term1": term1,
                    "term2": term2,
                },
            )
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EXPLANATION,
                entities=[ent],
                attribute="conceptual_distinction",
                answer_type=AnswerType.EXPLANATION,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
                metadata={"concept_a": concept_a, "concept_b": concept_b},
            )

        # ── 12. Explanation / Causal ("why did the Roman Empire fall", "how does X work") ──
        why_m = re.match(r"^why\s+(?:did|was|were|is)\s+(?:the\s+)?(.+?)\s+(?:fall|collapse|decline|end|happen|occur|fail)\??$", stripped, re.I)
        if why_m:
            ent = _format_entity_casing(why_m.group(1))
            contract = extract_answer_contract(stripped, Intent.EXPLANATION, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EXPLANATION,
                entities=[ent] if ent else [],
                attribute=None,
                answer_type=AnswerType.EXPLANATION,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        how_work_m = re.match(r"^how\s+(?:does|do|did)\s+(.+?)\s+work\??$", stripped, re.I)
        if how_work_m:
            ent = _format_entity_casing(how_work_m.group(1))
            contract = extract_answer_contract(stripped, Intent.EXPLANATION, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EXPLANATION,
                entities=[ent] if ent else [],
                attribute=None,
                answer_type=AnswerType.EXPLANATION,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
            )

        # ── 13. Negative / Exclusion questions ("Why was Kamsa not considered...") ──
        if re.search(r"\b(?:why\s+was|why\s+is|why\s+were|why\s+did)\b.*?\b(?:not|never|neither)\b", lower_q):
            ent_match = re.search(r"\b(?:why\s+(?:was|is|did|were))\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:not|never)\b", stripped, re.I)
            raw_ent = ent_match.group(1) if ent_match else None
            ent = _format_entity_casing(raw_ent) if raw_ent else None
            contract = extract_answer_contract(stripped, Intent.EXPLANATION, ent)
            return QueryUnderstanding(
                original_query=stripped,
                route=Route.KNOWLEDGE,
                intent=Intent.EXPLANATION,
                entities=[ent] if ent else [],
                attribute=None,
                answer_type=AnswerType.CAUSAL,
                constraints=constraints,
                contract=contract,
                source="rule",
                confidence=1.0,
                metadata={"fast_path": "negative_causal"},
            )

        # ── 14. Definition ("what is photosynthesis", "define X") ─────────────
        def_m = re.match(r"^(?:what(?:'s|\s+is|\s+are)\s+|define\s+|definition\s+of\s+)(?:a\s+|an\s+|the\s+)?(.+?)\??$", stripped, re.I)
        if def_m:
            raw_ent = def_m.group(1)
            if raw_ent.lower().split()[0] not in _REFERENCE_PRONOUNS and not re.search(r"^(?:my|your|his|her|their|our)\b", raw_ent.lower()):
                ent = _format_entity_casing(raw_ent)
                contract = extract_answer_contract(stripped, Intent.DEFINITION, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.DEFINITION,
                    entities=[ent] if ent else [],
                    attribute=None,
                    answer_type=AnswerType.DEFINITION,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                )

        # ── 15. Entity Overview ("tell me about X", "who is X") ───────────────
        who_is_m = re.match(r"^who\s+(?:is|was|were|'s)\s+(?:the\s+)?(.+?)\??$", stripped, re.I)
        if who_is_m:
            raw_ent = who_is_m.group(1)
            if raw_ent.lower().split()[0] not in _REFERENCE_PRONOUNS:
                ent = _format_entity_casing(raw_ent)
                contract = extract_answer_contract(stripped, Intent.OVERVIEW, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.OVERVIEW,
                    entities=[ent] if ent else [],
                    attribute=None,
                    answer_type=AnswerType.PERSON,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                )

        tell_me_m = re.match(r"^(?:(?:can\s+you\s+)?tell\s+me\s+(?:about|all\s+about|more\s+about)|describe|give\s+me\s+(?:a\s+)?summary\s+of)\s+(.+?)\??$", stripped, re.I)
        if tell_me_m:
            raw_ent = tell_me_m.group(1)
            if raw_ent.lower().split()[0] not in _REFERENCE_PRONOUNS:
                ent = _format_entity_casing(raw_ent)
                contract = extract_answer_contract(stripped, Intent.OVERVIEW, ent)
                return QueryUnderstanding(
                    original_query=stripped,
                    route=Route.KNOWLEDGE,
                    intent=Intent.OVERVIEW,
                    entities=[ent] if ent else [],
                    attribute=None,
                    answer_type=AnswerType.FACT,
                    constraints=constraints,
                    contract=contract,
                    source="rule",
                    confidence=1.0,
                )

        return None

    def understand(self, query: str) -> QueryUnderstanding:
        """Synchronous query understanding with deterministic fast-path and safe fallback."""
        cleaned = query.strip()
        det = self._classify_deterministically(cleaned)
        if det is not None:
            self._log_qu(det)
            return det
        # Safe fallback: NEVER put raw sentence as entity
        fallback = QueryUnderstanding(
            original_query=cleaned,
            route=Route.KNOWLEDGE,
            intent=Intent.GENERAL_QUESTION,
            entities=[],
            attribute=None,
            answer_type=AnswerType.FACT,
            constraints=_extract_constraints(cleaned),
            source="fallback",
            confidence=0.0,
        )
        self._log_qu(fallback)
        return fallback

    def analyze_query(self, query: str) -> QueryUnderstanding:
        """Alias for understand()."""
        return self.understand(query)

    async def analyze(self, query: str, llm: Optional[BaseLanguageModel] = None) -> QueryUnderstanding:
        """Asynchronous query understanding with LLM fallback and structured validation."""
        cleaned_query = query.strip()
        fallback = QueryUnderstanding(
            original_query=cleaned_query,
            route=Route.KNOWLEDGE,
            intent=Intent.GENERAL_QUESTION,
            entities=[],
            attribute=None,
            answer_type=AnswerType.FACT,
            constraints=_extract_constraints(cleaned_query),
            source="fallback",
            confidence=0.0,
        )

        if not cleaned_query:
            self._log_qu(fallback)
            return fallback

        # ── 1. Try deterministic classification first ───────────────────────
        det = self._classify_deterministically(cleaned_query)
        if det is not None:
            self._log_qu(det)
            return det

        # ── 2. Fall back to LLM-based classification ─────────────────────────
        if not llm:
            self._log_qu(fallback)
            return fallback

        raw_response = ""
        try:
            prompt = ANALYSIS_PROMPT_TEMPLATE.format(query=cleaned_query)
            raw_response = await llm.generate_response(prompt)

            # Extract JSON block from markdown fences or bare text
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

            route_str = str(parsed_data.get("route", Route.KNOWLEDGE)).strip().lower()
            intent_str = str(parsed_data.get("intent", Intent.GENERAL_QUESTION)).strip().lower()
            
            raw_ents = parsed_data.get("entities", [])
            if isinstance(raw_ents, str):
                raw_ents = [raw_ents]
            clean_ents = [_format_entity_casing(e) for e in raw_ents if e and str(e).lower() not in ("none", "null", "n/a", "")]
            
            attr_val = parsed_data.get("attribute")
            if attr_val and str(attr_val).strip().lower() in ("none", "null", "n/a", ""):
                attr_val = None
            elif attr_val:
                attr_val = str(attr_val).strip().lower()

            ans_type_str = str(parsed_data.get("answer_type", AnswerType.FACT)).strip().upper()
            if ans_type_str not in AnswerType.__members__:
                ans_type_str = AnswerType.FACT

            needs_ctx = bool(parsed_data.get("needs_context", False))
            is_ambig = bool(parsed_data.get("ambiguous", False))
            user_constraints = parsed_data.get("constraints", {})
            if not isinstance(user_constraints, dict):
                user_constraints = {}
            # Merge deterministic constraints
            for k, v in _extract_constraints(cleaned_query).items():
                user_constraints[k] = v

            contract = extract_answer_contract(cleaned_query, intent_str, clean_ents[0] if clean_ents else None)

            result = QueryUnderstanding(
                original_query=cleaned_query,
                route=route_str if route_str in Route.__members__.values() else Route.KNOWLEDGE,
                intent=intent_str if intent_str in Intent.__members__.values() else Intent.GENERAL_QUESTION,
                entities=clean_ents,
                attribute=attr_val,
                answer_type=ans_type_str,
                needs_context=needs_ctx,
                ambiguous=is_ambig,
                constraints=user_constraints,
                contract=contract,
                source="llm",
                confidence=0.85,
            )
            self._log_qu(result)
            return result

        except Exception as e:
            logger.warning("[QU] LLM parse failed (%s). Raw response: %.150s. Using fallback.", e, raw_response)
            self._log_qu(fallback)
            return fallback


query_analyzer = QueryAnalyzer()
query_understander = query_analyzer

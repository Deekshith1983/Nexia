"""Stage 4 Question & Multi-Entity Extraction Engine.

Identifies structured relational intents, directional roles, and named entities without using an LLM.
Strips relationship phrases so they never pollute entity names, and normalizes entities (e.g. honorifics).
"""

import re
from typing import List, Dict, Tuple, Set, Optional, NamedTuple
from app.schemas.retrieval import (
    EntityItem,
    RelationshipIntent,
    Stage4StructuredQuery,
    QueryConstraints,
)
from app.retrieval.nlp_query_parser import NLPQueryParser


class RelationPatternRule(NamedTuple):
    pattern: re.Pattern
    relation_type: str
    direction: str  # "OUTGOING", "INCOMING", "BIDIRECTIONAL"
    expected_answer_type: str
    relation_terms: List[str]
    is_multi_entity: bool


class EntityExtractor:
    """Extracts multiple named entities and structured relational intents using deterministic pattern registries."""

    HONORIFICS: Set[str] = {
        "shri", "sri", "smt", "sir", "lord", "lady", "dr", "dr.", "doctor",
        "prof", "prof.", "professor", "king", "emperor", "empress", "queen",
        "prince", "princess", "saint", "st", "st.", "swami", "pope", "maharaja",
        "raja", "sultan", "caliph", "president", "prime minister", "general"
    }

    ORG_INDICATORS: Set[str] = {
        "university", "institute", "college", "school", "academy", "corporation",
        "company", "inc", "corp", "ltd", "organization", "agency", "association",
        "bank", "foundation", "department", "ministry", "hospital", "parliament",
        "apple", "microsoft", "google", "meta", "amazon", "ibm", "openai"
    }

    KNOWN_COUNTRIES: Set[str] = {
        "india", "pakistan", "united states", "usa", "us", "france", "germany",
        "united kingdom", "uk", "britain", "japan", "china", "russia", "italy",
        "canada", "australia", "brazil", "south africa", "egypt", "turkey",
        "spain", "mexico", "bangladesh", "thailand", "vietnam", "greece", "rome",
        "israel", "iran", "iraq", "switzerland", "sweden", "norway", "poland",
        "europe", "asia", "africa", "north america", "south america", "oceania"
    }

    # Deterministic Relation Pattern Registry
    RELATION_REGISTRY: List[RelationPatternRule] = [
        # --- 1. SUCCESSION (OUTGOING: Subject was succeeded by Object) ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:was|is)\s+(?:the\s+)?)?(?:successor\s+of|next\s+(?:ruler|king|emperor|leader|person)?\s+after|who\s+came\s+after|who\s+followed)\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="SUCCESSOR_OF",
            direction="OUTGOING",
            expected_answer_type="PERSON",
            relation_terms=["successor", "succeeded", "succeeding", "succession", "heir", "after", "reign"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+)?succeeded\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="SUCCESSOR_OF",
            direction="OUTGOING",
            expected_answer_type="PERSON",
            relation_terms=["successor", "succeeded", "succeeding", "succession", "heir", "after"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(.+?)(?:'s|\s+was)?\s+successor(?:\s+was\s+who)?$",
                re.IGNORECASE,
            ),
            relation_type="SUCCESSOR_OF",
            direction="OUTGOING",
            expected_answer_type="PERSON",
            relation_terms=["successor", "succeeded", "succeeding", "succession"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(.+?)\s+was\s+succeeded\s+by\s+(?:whom|who)$",
                re.IGNORECASE,
            ),
            relation_type="SUCCESSOR_OF",
            direction="OUTGOING",
            expected_answer_type="PERSON",
            relation_terms=["successor", "succeeded", "succeeding", "succession"],
            is_multi_entity=False,
        ),

        # --- 2. PREDECESSOR (INCOMING: Object preceded Subject) ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:was|is)\s+(?:the\s+)?)?(?:predecessor\s+of|who\s+(?:was|came)\s+before)\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="PREDECESSOR_OF",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["predecessor", "preceded", "preceding", "before", "previous"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+)?preceded\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="PREDECESSOR_OF",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["predecessor", "preceded", "preceding", "before"],
            is_multi_entity=False,
        ),

        # --- 3. FOUNDER / ESTABLISHED (INCOMING: Object founded Subject) ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:founded|established|created|started))\s+(?:the\s+)?(.+?)(?:\s+and\s+who\s+was\s+(?:his|her|their)?\s*(?:business\s+)?(?:partner|co-founder|associate))?$",
                re.IGNORECASE,
            ),
            relation_type="FOUNDED_BY",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["founded", "founder", "co-founder", "established", "creator", "started", "partner"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:was|is)\s+(?:the\s+)?)?(?:founder|creator|establisher)\s+of\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="FOUNDED_BY",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["founded", "founder", "established", "creator", "started"],
            is_multi_entity=False,
        ),

        # --- 4. EMPLOYMENT / PARTNER / ASSOCIATION ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+)?(?:worked\s+with|partnered\s+with|associated\s+with|collaborated\s+with)\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="WORKED_WITH",
            direction="BIDIRECTIONAL",
            expected_answer_type="PERSON",
            relation_terms=["worked with", "partner", "colleague", "collaborator", "associated"],
            is_multi_entity=False,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:was|is)\s+(?:the\s+)?)?(?:business\s+partner|partner|colleague|collaborator)\s+of\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="BUSINESS_PARTNER_OF",
            direction="BIDIRECTIONAL",
            expected_answer_type="PERSON",
            relation_terms=["partner", "business partner", "co-founder", "collaborator", "colleague"],
            is_multi_entity=False,
        ),

        # --- 5. LOCATION / BIRTHPLACE ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:where\s+was\s+)(.+?)\s+born$",
                re.IGNORECASE,
            ),
            relation_type="BORN_IN",
            direction="OUTGOING",
            expected_answer_type="LOCATION",
            relation_terms=["born", "birthplace", "native", "place of birth"],
            is_multi_entity=False,
        ),

        # --- 6. MULTI-ENTITY: RELATIONSHIP BETWEEN X AND Y ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:what\s+is\s+(?:the\s+)?)?(?:relationship|connection|link|interaction|history)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="BILATERAL_RELATIONS",
            direction="BIDIRECTIONAL",
            expected_answer_type="RELATIONSHIP",
            relation_terms=["relationship", "relations", "connection", "history", "bilateral"],
            is_multi_entity=True,
        ),
        RelationPatternRule(
            pattern=re.compile(
                r"^how\s+(?:are|were|is|was)\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+?)\s+(?:related|connected|associated|linked)$",
                re.IGNORECASE,
            ),
            relation_type="BILATERAL_RELATIONS",
            direction="BIDIRECTIONAL",
            expected_answer_type="RELATIONSHIP",
            relation_terms=["related", "relationship", "connected", "associated"],
            is_multi_entity=True,
        ),

        # --- 7. MULTI-ENTITY: DIFFERENCE / COMPARISON BETWEEN X AND Y ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:what\s+is\s+(?:the\s+)?)?(?:difference|comparison|contrast)\s+between\s+(?:the\s+)?(.+?)\s+and\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="DIFFERENCE_COMPARISON",
            direction="BIDIRECTIONAL",
            expected_answer_type="COMPARISON",
            relation_terms=["difference", "comparison", "contrast", "versus", "vs"],
            is_multi_entity=True,
        ),

        # --- 8. FATHER OF THE NATION / NATIONAL FOUNDER ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:is|was)\s+(?:the\s+)?)?(?:father\s+of\s+(?:the\s+)?nation|national\s+founder)\s+(?:of|in)\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="FATHER_OF_NATION_OF",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["father of the nation", "father of nation", "rashtrapita", "national founder", "bapu", "mahatma gandhi"],
            is_multi_entity=False,
        ),

        # --- 9. ORDINAL OFFICE HOLDER (e.g., 3rd prime minister of india) ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:who\s+(?:is|was)\s+(?:the\s+)?)?([0-9]+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(prime\s+minister|president|monarch|king|emperor|ruler)\s+of\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="ORDINAL_OFFICE_HOLDER",
            direction="INCOMING",
            expected_answer_type="PERSON",
            relation_terms=["prime minister", "office", "term", "order", "tenure"],
            is_multi_entity=False,
        ),

        # --- 10. LIST ENUMERATION OF CATEGORIES, OFFICES & ROLES ---
        RelationPatternRule(
            pattern=re.compile(
                r"^(?:list\s+(?:out\s+)?|who\s+(?:are|were)\s+(?:the\s+)?|names?\s+(?:of\s+)?(?:all\s+)?|what\s+are\s+(?:the\s+)?)(?:the\s+)?([a-zA-Z\s]+?)\s+(?:of|in)\s+(?:the\s+)?(.+)$",
                re.IGNORECASE,
            ),
            relation_type="OFFICE_HOLDERS_OF",
            direction="INCOMING",
            expected_answer_type="ENTITY_LIST",
            relation_terms=["list", "table", "states", "territories", "countries", "office", "term", "birth", "birth place", "education", "rule period"],
            is_multi_entity=False,
        ),

        # --- 11. COMPOUND BIOGRAPHICAL QUERY (e.g., who was akbar and when did he rule...) ---
        RelationPatternRule(
            pattern=re.compile(
                r"^who\s+(?:was|is)\s+([a-zA-Z0-9\s]+?)(?:\s+(?:and|,)\s+(?:when|what|where|how|which).+)$",
                re.IGNORECASE,
            ),
            relation_type="COMPOUND_BIOGRAPHY",
            direction="OUTGOING",
            expected_answer_type="PERSON",
            relation_terms=["biography", "reign", "rule", "contribution", "administration", "achievements", "history"],
            is_multi_entity=False,
        ),
    ]

    @classmethod
    def analyze_stage4_query(cls, question: str) -> Stage4StructuredQuery:
        """Parse raw question into full Stage 4 Structured Query Representation."""
        cleaned_q = question.strip()
        # Normalize common user typos in structural question words
        cleaned_q = re.sub(r"\bthr\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bteh\b", "the", cleaned_q, flags=re.IGNORECASE)
        cleaned_q = re.sub(r"\bthier\b", "their", cleaned_q, flags=re.IGNORECASE)
        q_clean = re.sub(r"[\?\.\!]+$", "", cleaned_q).strip()

        # 1. Test against Relation Pattern Registry
        for rule in cls.RELATION_REGISTRY:
            match = rule.pattern.search(q_clean)
            if match:
                groups = match.groups()

                if rule.is_multi_entity and len(groups) >= 2:
                    raw_ent1 = cls._clean_entity_text(groups[0])
                    raw_ent2 = cls._clean_entity_text(groups[1])
                    norm1 = cls.normalize_entity_text(raw_ent1)
                    norm2 = cls.normalize_entity_text(raw_ent2)

                    entities = [
                        EntityItem(
                            text=raw_ent1,
                            normalized_text=norm1,
                            type=cls._infer_entity_type(norm1),
                            role="SUBJECT_A",
                        ),
                        EntityItem(
                            text=raw_ent2,
                            normalized_text=norm2,
                            type=cls._infer_entity_type(norm2),
                            role="SUBJECT_B",
                        ),
                    ]

                    q_type = "MULTI_ENTITY_COMPARISON" if rule.relation_type == "DIFFERENCE_COMPARISON" else "MULTI_ENTITY_RELATION"
                    rel_intent = RelationshipIntent(
                        type=rule.relation_type,
                        direction=rule.direction,
                        target_role="SUBJECT_B",
                        relation_terms=rule.relation_terms,
                    )
                    retrieval_terms = [norm1, norm2] + rule.relation_terms

                    return Stage4StructuredQuery(
                        raw_question=question,
                        question_type=q_type,
                        entities=entities,
                        relation=rel_intent,
                        expected_answer_type=rule.expected_answer_type,
                        retrieval_terms=retrieval_terms,
                    )

                elif not rule.is_multi_entity and len(groups) >= 1:
                    custom_terms = None
                    if rule.relation_type == "OFFICE_HOLDERS_OF" and len(groups) >= 2:
                        category = groups[0].strip()
                        raw_realm = groups[1].strip()
                        cleaned_realm = re.sub(
                            r"\s+(?:and|,)\s+(?:th[ie]{2}r|their|the|when|what|dates?|periods?|timeline|birth|education).*$",
                            "",
                            raw_realm,
                            flags=re.IGNORECASE,
                        ).rstrip(".,!? ").strip()
                        raw_ent = cleaned_realm
                        norm_ent = cls.normalize_entity_text(cleaned_realm)
                        custom_terms = [
                            f"{category} of {norm_ent}",
                            f"{category} in {norm_ent}",
                            f"List of {category} of {norm_ent}",
                            f"List of {category} in {norm_ent}",
                            norm_ent,
                            category,
                            "reign",
                            "period",
                            "dates",
                        ]
                    elif rule.relation_type == "ORDINAL_OFFICE_HOLDER" and len(groups) >= 3:
                        ordinal = groups[0].strip()
                        office = groups[1].strip()
                        raw_realm = groups[2].strip()
                        cleaned_realm = re.sub(
                            r"\s+(?:and|,)\s+(?:th[ie]{2}r|their|the|when|what|dates?|periods?|timeline|birth|education).*$",
                            "",
                            raw_realm,
                            flags=re.IGNORECASE,
                        ).rstrip(".,!? ").strip()
                        raw_ent = cleaned_realm
                        norm_ent = cls.normalize_entity_text(cleaned_realm)
                        custom_terms = [
                            norm_ent,
                            f"List of {office}s of {norm_ent}",
                            f"{ordinal} {office}",
                            "reign",
                            "term",
                        ]
                    elif rule.relation_type == "FATHER_OF_NATION_OF" and len(groups) >= 1:
                        raw_realm = groups[0].strip()
                        cleaned_realm = re.sub(
                            r"\s+(?:and|,)\s+(?:th[ie]{2}r|their|the|when|what|dates?|periods?|timeline|birth|education).*$",
                            "",
                            raw_realm,
                            flags=re.IGNORECASE,
                        ).rstrip(".,!? ").strip()
                        raw_ent = cleaned_realm
                        norm_ent = cls.normalize_entity_text(cleaned_realm)
                        custom_terms = [
                            norm_ent,
                            f"Father of the Nation {norm_ent}",
                            "Father of the Nation",
                            "national founder",
                            "independence",
                        ]
                    else:
                        raw_ent = cls._clean_entity_text(groups[0])
                        norm_ent = cls.normalize_entity_text(raw_ent)

                    entities = [
                        EntityItem(
                            text=raw_ent,
                            normalized_text=norm_ent,
                            type=cls._infer_entity_type(norm_ent),
                            role="SUBJECT",
                        )
                    ]

                    rel_intent = RelationshipIntent(
                        type=rule.relation_type,
                        direction=rule.direction,
                        target_role="TARGET_OBJECT",
                        relation_terms=rule.relation_terms,
                    )
                    retrieval_terms = custom_terms if custom_terms else [norm_ent] + rule.relation_terms

                    return Stage4StructuredQuery(
                        raw_question=question,
                        question_type="RELATIONSHIP",
                        entities=entities,
                        relation=rel_intent,
                        expected_answer_type=rule.expected_answer_type,
                        retrieval_terms=retrieval_terms,
                    )

        # 2. Capitalized Proper Noun Chunks fallback (Multi-entity detection)
        proper_noun_chunks = re.findall(
            r"\b[A-Z][a-z]+(?:\s+(?:of|and|the|for|in|at)\s+[A-Z][a-z]+|\s+[A-Z][a-z]+)*\b",
            cleaned_q,
        )
        EXCLUDED_PROPER_NOUNS = {
            "what", "who", "where", "when", "why", "how", "which", "is", "are",
            "list", "name", "names", "give", "tell", "show", "describe", "explain",
            "find", "enumerate", "details", "information", "provide", "mention",
            "the", "a", "an", "of", "in", "to", "for"
        }
        filtered_chunks = []
        for chunk in proper_noun_chunks:
            c_clean = chunk.strip()
            if c_clean.lower() in EXCLUDED_PROPER_NOUNS:
                continue
            if len(c_clean.split()) > 0 and len(c_clean) > 2:
                filtered_chunks.append(c_clean)

        # NLP Semantic parsing: use spaCy noun chunks & NER to identify entities and aspects
        if not filtered_chunks:
            nlp_res = NLPQueryParser.parse(cleaned_q)
            if nlp_res.candidate_entities:
                for ce in nlp_res.candidate_entities:
                    filtered_chunks.append(ce.normalized_text)
            else:
                lower_q = cleaned_q.lower()
                for country in cls.KNOWN_COUNTRIES:
                    if re.search(rf"\b{re.escape(country)}\b", lower_q):
                        filtered_chunks.append(country.title())
                for known_kw in ["akbar", "akbhar", "hoysala", "modi", "gandhi", "krishnadevaraya", "nehru", "shastri"]:
                    if re.search(rf"\b{re.escape(known_kw)}\b", lower_q):
                        filtered_chunks.append(known_kw.title())

        entities: List[EntityItem] = []
        if len(filtered_chunks) >= 2:
            for i, chunk in enumerate(filtered_chunks[:3]):
                role = f"SUBJECT_{chr(65 + i)}"
                norm = cls.normalize_entity_text(chunk)
                entities.append(
                    EntityItem(
                        text=chunk,
                        normalized_text=norm,
                        type=cls._infer_entity_type(norm),
                        role=role,
                    )
                )
            return Stage4StructuredQuery(
                raw_question=question,
                question_type="MULTI_ENTITY_RELATION",
                entities=entities,
                relation=RelationshipIntent(
                    type="GENERAL_ASSOCIATION",
                    direction="BIDIRECTIONAL",
                    relation_terms=["history", "connection", "association"],
                ),
                expected_answer_type="RELATIONSHIP",
                retrieval_terms=[e.normalized_text for e in entities],
            )
        elif len(filtered_chunks) == 1:
            raw_ent = filtered_chunks[0]
            norm_ent = cls.normalize_entity_text(raw_ent)
            entities.append(
                EntityItem(
                    text=raw_ent,
                    normalized_text=norm_ent,
                    type=cls._infer_entity_type(norm_ent),
                    role="SUBJECT",
                )
            )
            return Stage4StructuredQuery(
                raw_question=question,
                question_type="SINGLE_DOCUMENT",
                entities=entities,
                relation=None,
                expected_answer_type="ENTITY",
                retrieval_terms=[norm_ent],
            )

        # 3. Fallback: Clean question text as candidate entity
        raw_clean = cls._clean_entity_text(q_clean)
        norm_clean = cls.normalize_entity_text(raw_clean)
        entities.append(
            EntityItem(
                text=raw_clean,
                normalized_text=norm_clean,
                type=cls._infer_entity_type(norm_clean),
                role="SUBJECT",
            )
        )
        return Stage4StructuredQuery(
            raw_question=question,
            question_type="SINGLE_DOCUMENT",
            entities=entities,
            relation=None,
            expected_answer_type="ENTITY",
            retrieval_terms=[norm_clean],
        )

    @classmethod
    def extract_entities(cls, question: str) -> List[EntityItem]:
        """Convenience helper extracting entity list from query."""
        structured = cls.analyze_stage4_query(question)
        return structured.entities

    @classmethod
    def normalize_entity_text(cls, entity_text: str) -> str:
        """Strip honorifics, titles, and structural artifacts from entity name without destroying core proper noun."""
        cleaned = entity_text.strip()
        tokens = cleaned.split()
        if not tokens:
            return cleaned

        # Remove leading honorific if present (e.g. 'Shri Krishnadevaraya' -> 'Krishnadevaraya')
        first_lower = tokens[0].lower().rstrip(".,")
        if first_lower in cls.HONORIFICS and len(tokens) > 1:
            tokens = tokens[1:]

        normalized = " ".join(tokens).strip()

        # Deterministic alias mapping for common entity variations & queries
        ALIASES = {
            "akbhar": "Akbar",
            "akbar the great": "Akbar",
            "hoysala dynasty": "Hoysala Empire",
            "gandi": "Mahatma Gandhi",
            "gandhi": "Mahatma Gandhi",
            "krishnadeva raya": "Krishnadevaraya",
            "narendra damodardas modi": "Narendra Modi",
            "prime ministers of india": "List of prime ministers of India",
            "prime minister of india": "List of prime ministers of India",
        }
        lower_norm = normalized.lower()
        if lower_norm in ALIASES:
            return ALIASES[lower_norm]
        if lower_norm in cls.KNOWN_COUNTRIES:
            return lower_norm.title()
        if not any(c.isupper() for c in normalized):
            return normalized.title()

        return normalized if normalized else cleaned

    @classmethod
    def _clean_entity_text(cls, raw: str) -> str:
        """Strip filler question words, determiners, and trailing punctuation."""
        cleaned = raw.strip()
        # Remove leading determiners and question prefixes
        cleaned = re.sub(
            r"^(?:what\s+is\s+(?:the\s+)?|who\s+(?:was|is)\s+(?:the\s+)?|where\s+(?:was|is)\s+(?:the\s+)?|the\s+|a\s+|an\s+|between\s+)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"[,\.\?\!]+$", "", cleaned).strip()
        return cleaned

    @classmethod
    def _infer_entity_type(cls, entity_text: str) -> str:
        """Infer entity classification (ORGANIZATION, GPE_COUNTRY, PERSON, PRODUCT_OR_TECHNOLOGY, CONCEPT)."""
        lower = entity_text.lower()

        # Check Organization
        if any(ind in lower for ind in cls.ORG_INDICATORS):
            return "ORGANIZATION"

        # Check Country / GPE
        if lower in cls.KNOWN_COUNTRIES:
            return "GPE_COUNTRY"

        # Check Technology / Product
        if any(tech in lower for tech in ["apple", "microsoft", "google", "python", "java", "linux", "turing machine"]):
            if any(tech in lower for tech in ["apple", "microsoft", "google"]):
                return "ORGANIZATION"
            return "PRODUCT_OR_TECHNOLOGY"

        # Multi-word proper nouns default to PERSON
        words = entity_text.split()
        if len(words) >= 2 and all(w[0].isupper() for w in words if w not in ["of", "the", "and", "de", "da"]):
            return "PERSON"
        if len(words) == 1 and words[0].istitle():
            return "PERSON"

        return "CONCEPT"

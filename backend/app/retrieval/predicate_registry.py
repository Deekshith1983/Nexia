"""Stage 5 Predicate Registry: Comprehensive Canonical Predicates, Dual Inverses, and Pattern Rules."""

import re
from typing import List, Dict, Tuple, Optional, NamedTuple


class CanonicalPredicateDef(NamedTuple):
    predicate: str
    inverse_predicate: Optional[str]
    category: str
    default_direction: str  # "FORWARD", "INVERSE", "BIDIRECTIONAL"
    expected_subject_type: str
    expected_object_type: str
    extraction_patterns: List[re.Pattern]
    search_keywords: List[str]


class PredicateRegistry:
    """Registry of canonical relation predicates across diverse domains."""

    PREDICATE_DEFINITIONS: List[CanonicalPredicateDef] = [
        # --- 1. Academic & Intellectual ---
        CanonicalPredicateDef(
            predicate="STUDIED_AT",
            inverse_predicate="ALMA_MATER_OF",
            category="ACADEMIC_INSTITUTIONAL",
            default_direction="FORWARD",
            expected_subject_type="PERSON",
            expected_object_type="ORGANIZATION",
            extraction_patterns=[
                re.compile(r"(?:enrolled|studied|attended|graduated|matriculated)\s+(?:at|in|from)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+where|\s+who)", re.IGNORECASE),
                re.compile(r"alumnus\s+(?:of|at)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
            ],
            search_keywords=["studied", "enrolled", "attended", "graduated", "university", "college", "school"],
        ),
        CanonicalPredicateDef(
            predicate="EARNED_DOCTORATE_FROM",
            inverse_predicate="AWARDED_DOCTORATE_TO",
            category="ACADEMIC_INSTITUTIONAL",
            default_direction="FORWARD",
            expected_subject_type="PERSON",
            expected_object_type="ORGANIZATION",
            extraction_patterns=[
                re.compile(r"(?:awarded|earned|received|completed|submitted)\s+(?:his|her|their)?\s*(?:doctorate|phd|doctoral\s+dissertation|thesis|degree)\s+(?:at|from|by)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
            ],
            search_keywords=["doctorate", "phd", "dissertation", "thesis", "awarded", "degree"],
        ),
        CanonicalPredicateDef(
            predicate="SERVED_AS_FACULTY_AT",
            inverse_predicate="FACULTY_MEMBER",
            category="ACADEMIC_INSTITUTIONAL",
            default_direction="FORWARD",
            expected_subject_type="PERSON",
            expected_object_type="ORGANIZATION",
            extraction_patterns=[
                re.compile(r"(?:appointed|served\s+as|became)\s+(?:an?\s+)?(?:associate\s+|extraordinary\s+|full\s+|adjunct\s+)?professor\s+(?:of\s+[a-zA-Z\s]+)?(?:at|in)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
                re.compile(r"(?:taught|lectured)\s+(?:at|in)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
            ],
            search_keywords=["professor", "faculty", "taught", "lectured", "department", "chair"],
        ),

        # --- 2. Succession & Dynastic Governance ---
        CanonicalPredicateDef(
            predicate="SUCCESSOR_OF",
            inverse_predicate="PREDECESSOR_OF",
            category="SUCCESSION_GOVERNANCE",
            default_direction="FORWARD",
            expected_subject_type="PERSON",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:was\s+)?succeeded\s+by\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|son|nephew|cousin|successor)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|\s+whose|$)", re.IGNORECASE),
                re.compile(r"(?:his|her|their)?\s*successor\s+(?:was|became)\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|son|nephew|cousin)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
                re.compile(r"([A-Z][a-zA-Z\s\.\,\'\-]+?)\s+(?:succeeded\s+him|succeeded\s+to\s+the\s+throne|took\s+over\s+the\s+throne|became\s+his\s+successor)", re.IGNORECASE),
            ],
            search_keywords=["successor", "succeeded", "succeeding", "succession", "heir", "throne", "reign"],
        ),
        CanonicalPredicateDef(
            predicate="PREDECESSOR_OF",
            inverse_predicate="SUCCESSOR_OF",
            category="SUCCESSION_GOVERNANCE",
            default_direction="FORWARD",
            expected_subject_type="PERSON",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:was\s+)?preceded\s+by\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|father|predecessor)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|\s+whose|$)", re.IGNORECASE),
                re.compile(r"(?:his|her|their)?\s*predecessor\s+(?:was|had\s+been)\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
            ],
            search_keywords=["predecessor", "preceded", "preceding", "before", "prior", "previous"],
        ),

        # --- 3. Founder, Corporate & Inventions ---
        CanonicalPredicateDef(
            predicate="FOUNDED_BY",
            inverse_predicate="FOUNDER_OF",
            category="FOUNDER_CORPORATE",
            default_direction="FORWARD",
            expected_subject_type="ORGANIZATION",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:was\s+)?(?:founded|co-founded|established|started|created)\s+(?:(?:as|in|on)\s+[^,\.;\(\)]*?,?\s*)?by\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|and\s+its|\s+who|\s+to|$)", re.IGNORECASE),
                re.compile(r"([A-Z][a-zA-Z\s\.\,\'\-]+?)\s+(?:founded|co-founded|established|started)\s+(?:the\s+)?(?:company|kingdom|dynasty|empire|organization|firm|[A-Z][a-zA-Z\s]+)", re.IGNORECASE),
                re.compile(r"(?:its\s+)?(?:founder|co-founder|founders|co-founders|establisher)\s+(?:was|were|included|including)\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|\s+who|$)", re.IGNORECASE),
            ],
            search_keywords=["founded", "founder", "co-founder", "established", "creator", "started", "incorporated"],
        ),
        CanonicalPredicateDef(
            predicate="INVENTED_BY",
            inverse_predicate="INVENTOR_OF",
            category="INVENTION_CREATION",
            default_direction="FORWARD",
            expected_subject_type="PRODUCT_OR_TECHNOLOGY",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:invented|developed|designed|created|patented)\s+(?:in\s+\d{4}\s+)?by\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
            ],
            search_keywords=["invented", "inventor", "developed", "patent", "designed"],
        ),
        CanonicalPredicateDef(
            predicate="AUTHORED_BY",
            inverse_predicate="AUTHOR_OF",
            category="INVENTION_CREATION",
            default_direction="FORWARD",
            expected_subject_type="CONCEPT",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:written|authored|composed|published)\s+(?:in\s+\d{4}\s+)?by\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+who)", re.IGNORECASE),
            ],
            search_keywords=["written", "author", "composed", "published", "book", "play"],
        ),

        # --- 4. Geopolitical & Bilateral ---
        CanonicalPredicateDef(
            predicate="BILATERAL_TIES_WITH",
            inverse_predicate="BILATERAL_TIES_WITH",
            category="GEOPOLITICAL_TERRITORIAL",
            default_direction="BIDIRECTIONAL",
            expected_subject_type="GPE_COUNTRY",
            expected_object_type="GPE_COUNTRY",
            extraction_patterns=[
                re.compile(r"(?:bilateral\s+relations|diplomatic\s+relations|relations\s+between|conflict\s+between|war\s+between)\s+(?:the\s+)?", re.IGNORECASE),
            ],
            search_keywords=["bilateral", "diplomatic", "relations", "treaty", "ambassador"],
        ),
        CanonicalPredicateDef(
            predicate="SHARES_BORDER_WITH",
            inverse_predicate="SHARES_BORDER_WITH",
            category="GEOPOLITICAL_TERRITORIAL",
            default_direction="BIDIRECTIONAL",
            expected_subject_type="GPE_COUNTRY",
            expected_object_type="GPE_COUNTRY",
            extraction_patterns=[
                re.compile(r"(?:bordering|shares\s+a\s+border\s+with|bounded\s+by|adjacent\s+to)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|to\s+the|\s+and)", re.IGNORECASE),
            ],
            search_keywords=["border", "bordering", "bounded", "boundary", "neighboring"],
        ),
        CanonicalPredicateDef(
            predicate="CAPITAL_OF",
            inverse_predicate="HAS_CAPITAL",
            category="GEOPOLITICAL_TERRITORIAL",
            default_direction="FORWARD",
            expected_subject_type="LOCATION",
            expected_object_type="GPE_COUNTRY",
            extraction_patterns=[
                re.compile(r"capital\s+(?:and\s+largest\s+city\s+)?of\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|\s+and|\s+is)", re.IGNORECASE),
            ],
            search_keywords=["capital", "seat of government", "largest city"],
        ),
        CanonicalPredicateDef(
            predicate="LOCATED_IN",
            inverse_predicate="CONTAINS_LOCATION",
            category="GEOPOLITICAL_TERRITORIAL",
            default_direction="FORWARD",
            expected_subject_type="LOCATION",
            expected_object_type="GPE_COUNTRY",
            extraction_patterns=[
                re.compile(r"(?:located|situated|headquartered|based)\s+in\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|\s+and|\s+is)", re.IGNORECASE),
            ],
            search_keywords=["located", "situated", "headquartered", "based in", "city", "state"],
        ),

        # --- 5. Social, Collaborative & Association ---
        CanonicalPredicateDef(
            predicate="WORKED_WITH",
            inverse_predicate="WORKED_WITH",
            category="SOCIAL_COLLABORATION",
            default_direction="BIDIRECTIONAL",
            expected_subject_type="PERSON",
            expected_object_type="PERSON",
            extraction_patterns=[
                re.compile(r"(?:worked\s+with|partnered\s+with|collaborated\s+with|assistant\s+to|colleague\s+of)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|\s+and|\s+in)", re.IGNORECASE),
            ],
            search_keywords=["collaborated", "worked with", "partnered", "colleague", "assistant"],
        ),
        CanonicalPredicateDef(
            predicate="ASSOCIATED_WITH",
            inverse_predicate="ASSOCIATED_WITH",
            category="GENERAL_ASSOCIATION",
            default_direction="BIDIRECTIONAL",
            expected_subject_type="CONCEPT",
            expected_object_type="CONCEPT",
            extraction_patterns=[
                re.compile(r"(?:associated\s+with|connected\s+to|linked\s+with|related\s+to)\s+(?:the\s+)?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|\s+and)", re.IGNORECASE),
            ],
            search_keywords=["associated", "connected", "linked", "related", "history"],
        ),
    ]

    @classmethod
    def get_predicate_def(cls, predicate_name: str) -> Optional[CanonicalPredicateDef]:
        """Find definition by canonical name or alias."""
        p_norm = predicate_name.upper().replace(" ", "_")
        for defn in cls.PREDICATE_DEFINITIONS:
            if defn.predicate == p_norm or (defn.inverse_predicate and defn.inverse_predicate == p_norm):
                return defn
        return None

    @classmethod
    def get_inverse(cls, predicate_name: str) -> str:
        """Return canonical dual inverse predicate."""
        defn = cls.get_predicate_def(predicate_name)
        if defn and defn.inverse_predicate:
            return defn.inverse_predicate
        return predicate_name

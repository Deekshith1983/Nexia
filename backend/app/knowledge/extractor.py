"""
KnowledgeExtractor — Transforms raw Wikipedia evidence into structured knowledge records.
"""
import logging
import re
from typing import Any, List, Optional

from app.knowledge.models import Entity, EntityType, EvidenceReference, Fact, Relationship
from app.knowledge.workspace import KnowledgeWorkspace

logger = logging.getLogger("nexia.knowledge.extractor")

# Regex for dates in parentheses (e.g. "Albert Einstein (14 March 1879 – 18 April 1955)")
_DATE_PAREN_REGEX = re.compile(
    r"\(\s*(?:born\s+)?(\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4})\s*(?:[–\-—]|to)\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4}|present)\s*\)",
    re.IGNORECASE,
)
_BORN_SINGLE_REGEX = re.compile(
    r"\(\s*born\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4}|\d{4})\s*\)",
    re.IGNORECASE,
)

# Common relationship verbs and patterns
_RELATION_PATTERNS = [
    (r"(?i)\b(?:developed|formulated|created|proposed|introduced)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s–\-]+(?:\s+theory|\s+law|\s+model|\s+relativity|\s+mechanics|\s+calculus|\s+physics)?)", "developed"),
    (r"(?i)\b(?:discovered|identified|uncovered)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s–\-]+)", "discovered"),
    (r"(?i)\b(?:published|authored|wrote)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s–\-]+)", "published"),
    (r"(?i)\b(?:influenced|inspired|shaped)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s–\-]+)", "influenced"),
    (r"(?i)\b(?:known for|famous for|best known for)\s+(?:his\s+|her\s+|their\s+|the\s+)?([A-Za-z0-9\s–\-,]+)", "known for"),
    (r"(?i)\b(?:awarded|received|won)\s+(?:the\s+)?([A-Z][A-Za-z0-9\s–\-]+(?:Prize|Medal|Award)?)", "awarded"),
]


class KnowledgeExtractor:
    """
    Extracts structured entities, facts, and relationships from Wikipedia EvidenceItems.
    """

    def extract_from_evidence(
        self,
        evidence_item: Any,
        workspace: KnowledgeWorkspace,
    ) -> None:
        """
        Parse an EvidenceItem and populate the KnowledgeWorkspace with structured entities,
        facts, and relationships while retaining full evidence provenance.
        """
        page_title = evidence_item.page_title.strip()
        if not page_title:
            return

        content = evidence_item.content
        ev_ref_base = EvidenceReference(
            page_title=page_title,
            page_url=evidence_item.page_url,
            evidence_id=evidence_item.action_id,
            snippet=content[:200] if content else None,
        )

        # 1. Primary Entity Extraction
        ent_type = self._infer_entity_type(page_title, content)
        primary_entity = workspace.add_entity(
            canonical_name=page_title,
            entity_type=ent_type,
            evidence_ref=ev_ref_base,
        )

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if len(s.strip()) > 10]
        if not sentences:
            return

        # 2. Extract Birth & Death Dates (Parenthetical or Textual)
        first_para = " ".join(sentences[:3])
        m_dates = _DATE_PAREN_REGEX.search(first_para)
        if m_dates:
            birth_val = m_dates.group(1).strip()
            death_val = m_dates.group(2).strip()
            ev_snippet = EvidenceReference(
                page_title=page_title,
                page_url=evidence_item.page_url,
                evidence_id=evidence_item.action_id,
                snippet=m_dates.group(0),
            )
            workspace.add_fact(
                subject=page_title,
                predicate="born",
                object_value=birth_val,
                evidence_ref=ev_snippet,
            )
            if death_val.lower() != "present":
                workspace.add_fact(
                    subject=page_title,
                    predicate="died",
                    object_value=death_val,
                    evidence_ref=ev_snippet,
                )
        else:
            m_born = _BORN_SINGLE_REGEX.search(first_para)
            if m_born:
                workspace.add_fact(
                    subject=page_title,
                    predicate="born",
                    object_value=m_born.group(1).strip(),
                    evidence_ref=ev_ref_base,
                )

        # 3. Role / Definition Extraction ("X was a Y", "X is a Y")
        m_role = re.search(
            rf"(?i)\b{re.escape(page_title)}(?:\s*\([^)]*\))?\s+(?:is|was|were|are)\s+(?:an?|the)\s+([^,.;]+)",
            first_para,
        )
        if m_role:
            role_text = m_role.group(1).strip()
            role_text = re.sub(r"\(.*?\)", "", role_text).strip()
            role_text = re.sub(r"\(.*?\)", "", role_text).strip()
            if len(role_text) > 3:
                workspace.add_fact(
                    subject=page_title,
                    predicate="is a",
                    object_value=role_text,
                    evidence_ref=ev_ref_base,
                )

        # 4. Extract Relationships from Sentences
        for sent in sentences[:10]:
            for pattern, relation_name in _RELATION_PATTERNS:
                for match in re.finditer(pattern, sent):
                    target_candidate = match.group(1).strip().rstrip(".,;:")
                    # Filter short or non-substantive matches
                    if len(target_candidate) >= 3 and len(target_candidate.split()) <= 6:
                        # Clean up target candidate
                        clean_target = re.sub(r"^(?:the|a|an|his|her|their)\s+", "", target_candidate, flags=re.IGNORECASE).strip()
                        if clean_target and clean_target.lower() != page_title.lower():
                            ev_sent_ref = EvidenceReference(
                                page_title=page_title,
                                page_url=evidence_item.page_url,
                                evidence_id=evidence_item.action_id,
                                snippet=sent[:200],
                            )
                            # Register target as entity if capitalized
                            if clean_target[0].isupper():
                                workspace.add_entity(
                                    canonical_name=clean_target,
                                    entity_type=EntityType.OTHER,
                                    evidence_ref=ev_sent_ref,
                                )
                            workspace.add_relationship(
                                source_entity=page_title,
                                relation=relation_name,
                                target_entity=clean_target,
                                evidence_ref=ev_sent_ref,
                            )

    def _infer_entity_type(self, title: str, content: str) -> EntityType:
        """Heuristic classification of entity type based on introductory text."""
        intro = content[:300].lower() if content else ""
        if any(w in intro for w in ["physicist", "mathematician", "scientist", "philosopher", "politician", "born", "died", "who was", "who is", "author", "inventor"]):
            return EntityType.PERSON
        if any(w in intro for w in ["theory", "hypothesis", "relativity", "mechanics", "principle"]):
            return EntityType.THEORY
        if any(w in intro for w in ["process", "phenomenon", "reaction", "biological process"]):
            return EntityType.CONCEPT
        if any(w in intro for w in ["battle", "war", "revolution", "treaty", "conference"]):
            return EntityType.EVENT
        if any(w in intro for w in ["company", "organization", "university", "institute", "agency"]):
            return EntityType.ORGANIZATION
        if any(w in intro for w in ["country", "city", "river", "mountain", "island", "region"]):
            return EntityType.LOCATION
        if any(w in intro for w in ["book", "novel", "album", "painting", "treatise", "principia"]):
            return EntityType.WORK
        return EntityType.OTHER


knowledge_extractor = KnowledgeExtractor()

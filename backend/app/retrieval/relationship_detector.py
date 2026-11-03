"""Stage 4 Relationship Detector & Evidence Linker.

Discovers, extracts target entities (objects), validates directional predicates,
and preserves complete source provenance for relationship queries.
"""

import re
from typing import List, Dict, Optional, Tuple
from app.schemas.retrieval import (
    EntityItem,
    EvidencePassage,
    ExtractedRelationship,
    RelationshipEvidenceItem,
    SourceDocumentReference,
    Stage4StructuredQuery,
)


class RelationshipDetector:
    """Discovers, extracts object entities, and validates directional relationships from retrieved passages."""

    GENERIC_NOUNS = {
        "brother", "son", "father", "mother", "daughter", "nephew", "cousin", "uncle",
        "successor", "predecessor", "heir", "ruler", "king", "emperor", "queen", "general",
        "minister", "partner", "colleague", "associate", "him", "her", "them", "someone", "who"
    }

    # In-passage detection patterns for single-entity relationship extraction
    SUCCESSION_PATTERNS = [
        # "succeeded by his brother Achyuta Deva Raya", "succeeded by Achyuta Deva Raya", "succeeded by his younger brother, Achyuta Deva Raya"
        re.compile(r"(?:was\s+)?succeeded\s+by\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|son|nephew|cousin|successor)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|\s+whose|$)", re.IGNORECASE),
        # "his successor was Achyuta Deva Raya"
        re.compile(r"(?:his|her|their)?\s*successor\s+(?:was|became)\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|son|nephew|cousin)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
        # "Achyuta Deva Raya succeeded him"
        re.compile(r"([A-Z][a-zA-Z\s\.\,\'\-]+?)\s+(?:succeeded\s+him|succeeded\s+to\s+the\s+throne|took\s+over\s+the\s+throne|became\s+his\s+successor)", re.IGNORECASE),
        # "succession of Achyuta Deva Raya"
        re.compile(r"succession\s+of\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
    ]

    PREDECESSOR_PATTERNS = [
        # "preceded by his brother Vira Narasimha Raya"
        re.compile(r"(?:was\s+)?preceded\s+by\s+(?:(?:his|her|their|the)\s+(?:younger\s+|elder\s+|older\s+|half-)?(?:brother|father|predecessor)\s*(?:,\s*|\s+))?([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|\s+whose|$)", re.IGNORECASE),
        # "his predecessor was Vira Narasimha Raya"
        re.compile(r"(?:his|her|their)?\s*predecessor\s+(?:was|had\s+been)\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
        # "succeeded Vira Narasimha Raya"
        re.compile(r"(?:succeeded|took\s+over\s+from)\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|after|\s+as|\s+who|$)", re.IGNORECASE),
    ]

    FOUNDER_PATTERNS = [
        # "founded as Apple Computer Company on April 1, 1976, by Steve Jobs, Steve Wozniak"
        re.compile(r"(?:was\s+)?(?:founded|co-founded|established|started|created)\s+(?:(?:as|in|on)\s+[^,\.;\(\)]*?,?\s*)?by\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|and\s+its|\s+who|\s+to|$)", re.IGNORECASE),
        # "Steve Jobs and Steve Wozniak founded"
        re.compile(r"([A-Z][a-zA-Z\s\.\,\'\-]+?)\s+(?:founded|co-founded|established|started)\s+(?:the\s+)?(?:company|kingdom|dynasty|empire|organization|firm|[A-Z][a-zA-Z\s]+)", re.IGNORECASE),
        # "founders were Steve Jobs and Steve Wozniak" / "co-founders Steve Jobs"
        re.compile(r"(?:its\s+)?(?:founder|co-founder|founders|co-founders|establisher)\s+(?:was|were|included|including)\s+([A-Z][a-zA-Z\s\.\,\'\-]+?)(?:[\.,;\(\)]|in\s+\d{4}|\s+who|$)", re.IGNORECASE),
    ]

    ACADEMIC_PATTERNS = [
        (re.compile(r"(?:awarded|earned|received|completed|submitted)\s+(?:his|her|their)?\s*(?:doctorate|phd|doctoral\s+dissertation|thesis|degree)\s+(?:at|from|to)\s+(?:the\s+)?", re.IGNORECASE), "earned_doctorate_from"),
        (re.compile(r"(?:appointed|served\s+as|became)\s+(?:an?\s+)?(?:associate\s+|extraordinary\s+|full\s+)?professor\s+(?:of\s+[a-zA-Z\s]+)?(?:at|in)\s+(?:the\s+)?", re.IGNORECASE), "served_as_professor_at"),
        (re.compile(r"(?:enrolled|studied|attended|graduated|matriculated)\s+(?:at|in|from)\s+(?:the\s+)?", re.IGNORECASE), "studied_at"),
        (re.compile(r"(?:faculty|alumnus|alumni|academic\s+staff|fellow)\s+(?:of|at)\s+(?:the\s+)?", re.IGNORECASE), "affiliated_with"),
    ]

    GEOPOLITICAL_PATTERNS = [
        (re.compile(r"(?:partition\s+of|divided\s+into|independence\s+from|gained\s+independence\s+from)\s+(?:the\s+)?", re.IGNORECASE), "partition_and_independence"),
        (re.compile(r"(?:bilateral\s+relations|diplomatic\s+relations|relations\s+between|conflict\s+between|war\s+between)\s+(?:the\s+)?", re.IGNORECASE), "bilateral_and_historical_relations"),
        (re.compile(r"(?:bordering|shares\s+a\s+border\s+with|adjacent\s+to)\s+(?:the\s+)?", re.IGNORECASE), "shares_border_with"),
        (re.compile(r"(?:signed\s+the|signatory\s+to|treaty\s+of|accord\s+with)\s+(?:the\s+)?", re.IGNORECASE), "diplomatic_treaty_with"),
    ]

    @classmethod
    def detect_single_entity_relationship(
        cls,
        query: Stage4StructuredQuery,
        subject_entity: EntityItem,
        passages: List[EvidencePassage],
        source_doc_ref: SourceDocumentReference,
    ) -> Tuple[List[ExtractedRelationship], List[RelationshipEvidenceItem]]:
        """Extract relationship object, validate directional evidence, and build provenance items for single-entity queries."""
        if not query.relation or not passages:
            return [], []

        rel_type = query.relation.type
        rel_items: List[RelationshipEvidenceItem] = []
        relationships: List[ExtractedRelationship] = []

        patterns_to_check = []
        category = "GENERAL"

        if rel_type == "SUCCESSOR_OF":
            patterns_to_check = cls.SUCCESSION_PATTERNS
            category = "SUCCESSION"
        elif rel_type == "PREDECESSOR_OF":
            patterns_to_check = cls.PREDECESSOR_PATTERNS
            category = "SUCCESSION"
        elif rel_type == "FOUNDED_BY":
            patterns_to_check = cls.FOUNDER_PATTERNS
            category = "FOUNDER_BUSINESS"

        for p in passages:
            for pat in patterns_to_check:
                match = pat.search(p.text)
                if match:
                    raw_object = match.group(1).strip()
                    # Clean object candidate
                    obj_clean = cls._clean_extracted_object(raw_object)
                    if obj_clean and len(obj_clean) > 2 and obj_clean.lower() != subject_entity.normalized_text.lower():
                        summary = cls._synthesize_sentence(p.text, match.start(), match.end())

                        evidence_item = RelationshipEvidenceItem(
                            subject=subject_entity.normalized_text,
                            relation=rel_type,
                            object=obj_clean,
                            source_document=source_doc_ref,
                            source_section=p.section_title,
                            source_paragraph=p.paragraph_id,
                            retrieval_score=p.score,
                            relationship_score=0.92,
                            validated=True,
                            passage_text=p.text,
                            summary_statement=summary,
                        )
                        rel_items.append(evidence_item)

                        relationships.append(
                            ExtractedRelationship(
                                subject=subject_entity.normalized_text,
                                relation=rel_type.lower(),
                                object=obj_clean,
                                relation_category=category,
                                confidence_score=0.92,
                                supporting_evidence_ids=[p.passage_id],
                                summary_statement=summary,
                            )
                        )
                        break

        # Deduplicate relationships by (subject, relation, object)
        unique_rels: List[ExtractedRelationship] = []
        seen = set()
        for r in relationships:
            key = (r.subject.lower(), r.relation.lower(), r.object.lower())
            if key not in seen:
                seen.add(key)
                unique_rels.append(r)

        return unique_rels, rel_items

    @classmethod
    def detect_cross_document_relationships(
        cls,
        entities: List[EntityItem],
        passages: List[EvidencePassage],
        doc_refs: Dict[str, SourceDocumentReference],
    ) -> Tuple[List[ExtractedRelationship], List[RelationshipEvidenceItem]]:
        """Detect structured relationships between multiple entities across candidate document passages."""
        if len(entities) < 2 or not passages:
            return [], []

        relationships: List[ExtractedRelationship] = []
        rel_items: List[RelationshipEvidenceItem] = []

        ent_a = entities[0].normalized_text
        ent_b = entities[1].normalized_text
        ent_a_lower = ent_a.lower()
        ent_b_lower = ent_b.lower()

        # Group supporting passages by domain
        supporting_academic: List[EvidencePassage] = []
        supporting_founder: List[EvidencePassage] = []
        supporting_geopolitical: List[EvidencePassage] = []
        supporting_general: List[EvidencePassage] = []

        for p in passages:
            text_lower = p.text.lower()
            has_ent_a = ent_a_lower in text_lower or any(tok in text_lower for tok in ent_a_lower.split() if len(tok) > 3)
            has_ent_b = ent_b_lower in text_lower or any(tok in text_lower for tok in ent_b_lower.split() if len(tok) > 3)

            if has_ent_a and has_ent_b:
                if any(pat[0].search(p.text) for pat in cls.ACADEMIC_PATTERNS):
                    supporting_academic.append(p)
                elif any(pat.search(p.text) for pat in cls.FOUNDER_PATTERNS):
                    supporting_founder.append(p)
                elif any(pat[0].search(p.text) for pat in cls.GEOPOLITICAL_PATTERNS):
                    supporting_geopolitical.append(p)
                else:
                    supporting_general.append(p)

        if supporting_academic:
            top_p = supporting_academic[0]
            pred = "affiliated_with"
            for pat, name in cls.ACADEMIC_PATTERNS:
                if pat.search(top_p.text):
                    pred = name
                    break

            summary = cls._synthesize_cross_summary(ent_a, pred, ent_b, top_p.text)
            doc_ref = doc_refs.get(top_p.document_title, SourceDocumentReference(title=top_p.document_title, page_id=0, url=""))

            rel_obj = ExtractedRelationship(
                subject=ent_a,
                relation=pred,
                object=ent_b,
                relation_category="ACADEMIC_INSTITUTIONAL",
                confidence_score=0.92,
                supporting_evidence_ids=[p.passage_id for p in supporting_academic[:3]],
                summary_statement=summary,
            )
            relationships.append(rel_obj)

            rel_items.append(
                RelationshipEvidenceItem(
                    subject=ent_a,
                    relation=pred,
                    object=ent_b,
                    source_document=doc_ref,
                    source_section=top_p.section_title,
                    source_paragraph=top_p.paragraph_id,
                    retrieval_score=top_p.score,
                    relationship_score=0.92,
                    validated=True,
                    passage_text=top_p.text,
                    summary_statement=summary,
                )
            )

        elif supporting_founder:
            top_p = supporting_founder[0]
            pred = "co_founded_or_partnered"
            summary = cls._synthesize_cross_summary(ent_a, pred, ent_b, top_p.text)
            doc_ref = doc_refs.get(top_p.document_title, SourceDocumentReference(title=top_p.document_title, page_id=0, url=""))

            rel_obj = ExtractedRelationship(
                subject=ent_a,
                relation=pred,
                object=ent_b,
                relation_category="FOUNDER_BUSINESS",
                confidence_score=0.90,
                supporting_evidence_ids=[p.passage_id for p in supporting_founder[:3]],
                summary_statement=summary,
            )
            relationships.append(rel_obj)

            rel_items.append(
                RelationshipEvidenceItem(
                    subject=ent_a,
                    relation=pred,
                    object=ent_b,
                    source_document=doc_ref,
                    source_section=top_p.section_title,
                    source_paragraph=top_p.paragraph_id,
                    retrieval_score=top_p.score,
                    relationship_score=0.90,
                    validated=True,
                    passage_text=top_p.text,
                    summary_statement=summary,
                )
            )

        elif supporting_geopolitical:
            top_p = supporting_geopolitical[0]
            pred = "bilateral_relations"
            for pat, name in cls.GEOPOLITICAL_PATTERNS:
                if pat.search(top_p.text):
                    pred = name
                    break

            summary = cls._synthesize_cross_summary(ent_a, pred, ent_b, top_p.text)
            doc_ref = doc_refs.get(top_p.document_title, SourceDocumentReference(title=top_p.document_title, page_id=0, url=""))

            rel_obj = ExtractedRelationship(
                subject=ent_a,
                relation=pred,
                object=ent_b,
                relation_category="GEOPOLITICAL",
                confidence_score=0.88,
                supporting_evidence_ids=[p.passage_id for p in supporting_geopolitical[:3]],
                summary_statement=summary,
            )
            relationships.append(rel_obj)

            rel_items.append(
                RelationshipEvidenceItem(
                    subject=ent_a,
                    relation=pred,
                    object=ent_b,
                    source_document=doc_ref,
                    source_section=top_p.section_title,
                    source_paragraph=top_p.paragraph_id,
                    retrieval_score=top_p.score,
                    relationship_score=0.88,
                    validated=True,
                    passage_text=top_p.text,
                    summary_statement=summary,
                )
            )

        elif supporting_general:
            top_p = supporting_general[0]
            summary = f"{ent_a} and {ent_b} are directly connected in historical context: {top_p.text[:200]}..."
            doc_ref = doc_refs.get(top_p.document_title, SourceDocumentReference(title=top_p.document_title, page_id=0, url=""))

            rel_obj = ExtractedRelationship(
                subject=ent_a,
                relation="associated_with",
                object=ent_b,
                relation_category="ASSOCIATION",
                confidence_score=0.72,
                supporting_evidence_ids=[p.passage_id for p in supporting_general[:2]],
                summary_statement=summary,
            )
            relationships.append(rel_obj)

            rel_items.append(
                RelationshipEvidenceItem(
                    subject=ent_a,
                    relation="associated_with",
                    object=ent_b,
                    source_document=doc_ref,
                    source_section=top_p.section_title,
                    source_paragraph=top_p.paragraph_id,
                    retrieval_score=top_p.score,
                    relationship_score=0.72,
                    validated=True,
                    passage_text=top_p.text,
                    summary_statement=summary,
                )
            )

        return relationships, rel_items

    @classmethod
    def _clean_extracted_object(cls, raw: str) -> str:
        """Strip trailing commas, conjunctions, relationship descriptors from extracted object entity."""
        cleaned = raw.strip()
        # Strip descriptor prefixes like 'his younger brother', 'his brother,', 'the king'
        cleaned = re.sub(
            r"^(?:(?:was\s+)?(?:succeeded|preceded|founded|created)\s+by\s+)?(?:(?:his|her|their|the|a|an)\b\s*)?(?:(?:younger|elder|older|half|adopted)\b\s*)?(?:(?:brother|son|father|mother|daughter|nephew|cousin|uncle|successor|predecessor|heir|ruler|king|emperor|queen|general|minister|associate|partner|colleague)\b\s*)?(?:(?:named|called|known\s+as)\b\s*)?[:,\s]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"[\.,;\(\)]+$", "", cleaned).strip()

        # Take first proper noun phrase if multiple separated by and/or
        if " and " in cleaned:
            parts = cleaned.split(" and ")
            if len(parts) > 0 and len(parts[0].strip().split()) >= 2:
                cleaned = parts[0].strip()

        # Reject generic nouns
        if cleaned.lower() in cls.GENERIC_NOUNS:
            return ""

        return cleaned


    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        """Split text into sentences cleanly using spaCy sentence segmentation to preserve abbreviations and titles."""
        try:
            from app.retrieval.nlp_query_parser import NLPQueryParser
            nlp = NLPQueryParser.get_nlp()
            if len(text) < 2500:
                sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
                if sents:
                    return sents
        except Exception:
            pass
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    @classmethod
    def _synthesize_sentence(cls, passage_text: str, start_pos: int, end_pos: int) -> str:
        """Extract the exact containing sentence around the matched match span."""
        sentences = cls._split_sentences(passage_text)
        for sent in sentences:
            if passage_text[start_pos:end_pos].strip() in sent or any(
                tok in sent for tok in passage_text[start_pos:end_pos].split() if len(tok) > 3
            ):
                return sent.strip()
        return passage_text[:250].strip()

    @classmethod
    def _synthesize_cross_summary(cls, subject: str, relation: str, object_entity: str, passage_text: str) -> str:
        """Extract or construct a clean summary sentence containing both entities."""
        sentences = cls._split_sentences(passage_text)
        sub_low = subject.lower()
        obj_low = object_entity.lower()

        for sent in sentences:
            sent_low = sent.lower()
            if any(tok in sent_low for tok in sub_low.split() if len(tok) > 3) and any(
                tok in sent_low for tok in obj_low.split() if len(tok) > 3
            ):
                return sent.strip()

        clean_rel = relation.replace("_", " ")
        return f"{subject} is documented as having {clean_rel} {object_entity}."

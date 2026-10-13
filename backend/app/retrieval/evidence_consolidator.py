"""Stage 5 Evidence Consolidator: Cross-Document Provenance Aggregator and Relationship Linker."""

import re
from typing import List, Dict, Tuple, Optional, Set
from app.schemas.retrieval import (
    EntityItem,
    EvidencePassage,
    SourceDocumentReference,
)
from app.schemas.relationships import (
    EvidenceProvenanceItem,
    DiscoveredRelationshipEdge,
    Stage5EvidenceGraph,
)
from app.retrieval.predicate_registry import PredicateRegistry, CanonicalPredicateDef
from app.retrieval.relationship_detector import RelationshipDetector
from app.retrieval.entity_extractor import EntityExtractor


class EvidenceConsolidator:
    """Consolidates cross-document evidence into verified relationship edges with provenance."""

    @classmethod
    def consolidate_evidence(
        cls,
        entities: List[EntityItem],
        passages: List[EvidencePassage],
        doc_refs: Dict[str, SourceDocumentReference],
        query_predicate: Optional[str] = None,
        min_confidence: float = 0.60,
    ) -> List[DiscoveredRelationshipEdge]:
        """Aggregate evidence passages across documents into structured, proven relationship edges."""
        if not passages:
            return []

        # Map to hold accumulated evidence chains keyed by (canonical_sub, canonical_pred, canonical_obj)
        edge_chains: Dict[Tuple[str, str, str], List[EvidenceProvenanceItem]] = {}
        edge_metadata: Dict[Tuple[str, str, str], Dict[str, any]] = {}

        # 1. Evaluate Every Passage Against Predicate Registry
        for p in passages:
            doc_ref = doc_refs.get(
                p.document_title,
                SourceDocumentReference(title=p.document_title, page_id=0, url=""),
            )

            # A. Check against registered canonical predicates
            for defn in PredicateRegistry.PREDICATE_DEFINITIONS:
                # If a specific predicate is requested, prioritize it
                if query_predicate and query_predicate.upper() not in [defn.predicate, defn.inverse_predicate]:
                    continue

                for pat in defn.extraction_patterns:
                    match = pat.search(p.text)
                    if match:
                        raw_extracted = match.group(1).strip() if match.groups() else ""
                        obj_clean = RelationshipDetector._clean_extracted_object(raw_extracted)

                        if obj_clean and len(obj_clean) > 2:
                            # Identify subject (from document title or entities)
                            sub_text = cls._identify_subject(p.document_title, entities, obj_clean)
                            if not sub_text or sub_text.lower() == obj_clean.lower():
                                continue

                            sentence = cls._extract_exact_sentence(p.text, match.start(), match.end())
                            lex_cues = [kw for kw in defn.search_keywords if kw in p.text.lower()]

                            provenance = EvidenceProvenanceItem(
                                document_title=p.document_title,
                                page_id=doc_ref.page_id,
                                url=doc_ref.url,
                                section_title=p.section_title,
                                section_level=p.section_level,
                                paragraph_id=p.paragraph_id,
                                passage_text=p.text,
                                exact_sentence=sentence,
                                support_score=min(1.0, p.score + 0.1),
                                lexical_cues_matched=lex_cues,
                            )

                            key = (sub_text.title(), defn.predicate, obj_clean.title())
                            if key not in edge_chains:
                                edge_chains[key] = []
                                edge_metadata[key] = {
                                    "category": defn.category,
                                    "inverse": defn.inverse_predicate,
                                    "sub_type": defn.expected_subject_type,
                                    "obj_type": defn.expected_object_type,
                                }
                            edge_chains[key].append(provenance)

            # B. Cross-Entity Mutual Mentions (for multi-entity relationship discovery)
            if len(entities) >= 2:
                for i in range(len(entities)):
                    for j in range(len(entities)):
                        if i == j:
                            continue
                        e_a = entities[i].normalized_text
                        e_b = entities[j].normalized_text
                        if e_a.lower() in p.text.lower() and e_b.lower() in p.text.lower():
                            # Check academic relation cues
                            for pat, pred_name in RelationshipDetector.ACADEMIC_PATTERNS:
                                if pat.search(p.text):
                                    sentence = cls._extract_cross_sentence(p.text, e_a, e_b)
                                    provenance = EvidenceProvenanceItem(
                                        document_title=p.document_title,
                                        page_id=doc_ref.page_id,
                                        url=doc_ref.url,
                                        section_title=p.section_title,
                                        section_level=p.section_level,
                                        paragraph_id=p.paragraph_id,
                                        passage_text=p.text,
                                        exact_sentence=sentence,
                                        support_score=min(1.0, p.score + 0.15),
                                        lexical_cues_matched=[pred_name],
                                    )
                                    key = (e_a.title(), pred_name.upper(), e_b.title())
                                    if key not in edge_chains:
                                        edge_chains[key] = []
                                        edge_metadata[key] = {
                                            "category": "ACADEMIC_INSTITUTIONAL",
                                            "inverse": "AFFILIATED_INSTITUTION_OF",
                                            "sub_type": "PERSON",
                                            "obj_type": "ORGANIZATION",
                                        }
                                    edge_chains[key].append(provenance)

                            # Check geopolitical relation cues
                            for pat, pred_name in RelationshipDetector.GEOPOLITICAL_PATTERNS:
                                if pat.search(p.text):
                                    sentence = cls._extract_cross_sentence(p.text, e_a, e_b)
                                    provenance = EvidenceProvenanceItem(
                                        document_title=p.document_title,
                                        page_id=doc_ref.page_id,
                                        url=doc_ref.url,
                                        section_title=p.section_title,
                                        section_level=p.section_level,
                                        paragraph_id=p.paragraph_id,
                                        passage_text=p.text,
                                        exact_sentence=sentence,
                                        support_score=min(1.0, p.score + 0.12),
                                        lexical_cues_matched=[pred_name],
                                    )
                                    key = (e_a.title(), pred_name.upper(), e_b.title())
                                    if key not in edge_chains:
                                        edge_chains[key] = []
                                        edge_metadata[key] = {
                                            "category": "GEOPOLITICAL_TERRITORIAL",
                                            "inverse": pred_name.upper(),
                                            "sub_type": "GPE_COUNTRY",
                                            "obj_type": "GPE_COUNTRY",
                                        }
                                    edge_chains[key].append(provenance)

        # 2. Synthesize & Score Consolidated Relationship Edges
        discovered_edges: List[DiscoveredRelationshipEdge] = []

        for (sub_name, predicate, obj_name), chain in edge_chains.items():
            meta = edge_metadata.get((sub_name, predicate, obj_name), {})
            category = meta.get("category", "GENERAL")
            inverse_pred = meta.get("inverse")

            # Count distinct source documents providing evidence
            distinct_docs: Set[str] = {item.document_title for item in chain}
            doc_count = len(distinct_docs)

            # Base confidence from top supporting passage
            top_support = max(item.support_score for item in chain) if chain else 0.70
            
            # Cross-document reinforcement bonus (+0.08 if confirmed in multiple Wikipedia articles)
            doc_bonus = 0.08 if doc_count >= 2 else 0.0
            evidence_count_bonus = min(0.06, len(chain) * 0.02)
            final_confidence = min(0.99, top_support + doc_bonus + evidence_count_bonus)

            if final_confidence < min_confidence:
                continue

            # Determine Verification Status
            if final_confidence >= 0.85 or (doc_count >= 2 and final_confidence >= 0.80):
                status = "PROVEN"
            elif final_confidence >= 0.60:
                status = "PARTIALLY_SUPPORTED"
            else:
                status = "UNSUPPORTED"

            # Create Entities
            sub_ent = EntityItem(
                text=sub_name,
                normalized_text=sub_name,
                type=meta.get("sub_type", EntityExtractor._infer_entity_type(sub_name)),
                role="SUBJECT",
            )
            obj_ent = EntityItem(
                text=obj_name,
                normalized_text=obj_name,
                type=meta.get("obj_type", EntityExtractor._infer_entity_type(obj_name)),
                role="OBJECT",
            )

            # Synthesize grounded claim statement
            top_item = max(chain, key=lambda x: x.support_score)
            synthesis = top_item.exact_sentence if top_item.exact_sentence else f"{sub_name} is documented as {predicate.lower().replace('_', ' ')} {obj_name}."

            edge_id = f"rel_{re.sub(r'[^a-zA-Z0-9]', '_', sub_name.lower())}_{predicate.lower()}_{re.sub(r'[^a-zA-Z0-9]', '_', obj_name.lower())}"

            edge = DiscoveredRelationshipEdge(
                edge_id=edge_id,
                subject=sub_ent,
                predicate=predicate,
                inverse_predicate=inverse_pred,
                object=obj_ent,
                category=category,
                status=status,
                confidence_score=round(final_confidence, 3),
                evidence_chain=chain[:5],  # Top 5 supporting provenance items
                distinct_documents_count=doc_count,
                synthesis_statement=synthesis,
            )
            discovered_edges.append(edge)

        # Sort edges by confidence score descending
        discovered_edges.sort(key=lambda e: e.confidence_score, reverse=True)
        return discovered_edges

    @classmethod
    def _identify_subject(cls, doc_title: str, entities: List[EntityItem], object_name: str) -> str:
        """Resolve primary subject from document title or extracted entities."""
        # If document title is different from object, it's usually the subject
        clean_doc = re.sub(r"\s+\([^\)]+\)", "", doc_title).strip()
        if clean_doc.lower() != object_name.lower():
            return clean_doc

        # Otherwise pick first non-matching entity
        for ent in entities:
            if ent.normalized_text.lower() != object_name.lower():
                return ent.normalized_text

        return clean_doc

    @classmethod
    def _extract_exact_sentence(cls, text: str, start: int, end: int) -> str:
        """Extract the exact containing sentence around a match span."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if text[start:end].strip() in sent or any(
                tok in sent for tok in text[start:end].split() if len(tok) > 3
            ):
                return sent.strip()
        return text[:200].strip()

    @classmethod
    def _extract_cross_sentence(cls, text: str, entity_a: str, entity_b: str) -> str:
        """Extract containing sentence mentioning both entities."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        ea_low = entity_a.lower()
        eb_low = entity_b.lower()
        for sent in sentences:
            s_low = sent.lower()
            if (ea_low in s_low or any(t in s_low for t in ea_low.split() if len(t) > 3)) and (
                eb_low in s_low or any(t in s_low for t in eb_low.split() if len(t) > 3)
            ):
                return sent.strip()
        return text[:250].strip()

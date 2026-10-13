"""Stage 4 Multi-Document & Relationship-Aware Retrieval Orchestrator.

Orchestrates Stage 4 query analysis, entity normalization, multi-article acquisition,
cross-document BM25 scoring, directional relationship validation, and diagnostic tracking.
"""

import asyncio
from typing import List, Optional, Dict
from app.schemas.retrieval import (
    MultiDocRetrievalRequest,
    MultiDocRetrievalResponse,
    EntityItem,
    DocumentEvidenceBundle,
    ExtractedRelationship,
    RelationshipEvidenceItem,
    SourceDocumentReference,
    Stage4Diagnostics,
    Stage4StructuredQuery,
    EvidencePassage,
    Document,
    Passage,
    SearchCandidate,
)
from app.retrieval.document_retriever import DocumentRetriever
from app.retrieval.passage_builder import PassageBuilder
from app.retrieval.entity_extractor import EntityExtractor
from app.retrieval.bm25_scorer import BM25RelevanceScorer
from app.retrieval.relationship_detector import RelationshipDetector


class MultiDocRetriever:
    """Orchestrator for Stage 4 multi-entity identification, multi-document harvesting, and relationship validation."""

    def __init__(self, document_retriever: Optional[DocumentRetriever] = None):
        self.doc_retriever = document_retriever or DocumentRetriever()
        self.passage_builder = PassageBuilder()

    async def retrieve_multi_document(
        self, request: MultiDocRetrievalRequest
    ) -> MultiDocRetrievalResponse:
        """Execute Stage 4 relationship-aware cross-document retrieval and validation."""
        question = request.question.strip()
        if not question:
            return MultiDocRetrievalResponse(
                question="",
                structured_query=None,
                entities_identified=[],
                documents=[],
                relationships_discovered=[],
                relationship_evidence=[],
                diagnostics=Stage4Diagnostics(
                    question_type_detected="UNKNOWN",
                    entities_identified=[],
                    failure_stage="ENTITY_EXTRACTION",
                    details="Question cannot be empty.",
                ),
                status="ERROR",
                is_relationship_proven=False,
                validation_reason="Question cannot be empty.",
                total_passages_evaluated=0,
                error_message="Question cannot be empty.",
            )

        # 1. Stage 4 Structured Query Analysis & Entity Normalization
        stage4_query: Stage4StructuredQuery = EntityExtractor.analyze_stage4_query(question)

        # Allow explicit manual entity overrides if provided
        if request.entities and len(request.entities) > 0:
            override_entities = []
            for i, ent_str in enumerate(request.entities):
                raw_e = ent_str.strip()
                norm_e = EntityExtractor.normalize_entity_text(raw_e)
                role = f"SUBJECT_{chr(65 + i)}" if i < 3 else "RELATED_ENTITY"
                override_entities.append(
                    EntityItem(
                        text=raw_e,
                        normalized_text=norm_e,
                        type=EntityExtractor._infer_entity_type(norm_e),
                        role=role,
                    )
                )
            stage4_query.entities = override_entities
            stage4_query.retrieval_terms = [e.normalized_text for e in override_entities]

        entities = stage4_query.entities
        entity_names = [e.normalized_text for e in entities]

        # 2. Execution Path A: Single-Entity Relationship Query (e.g. "successor of Shri Krishnadevaraya")
        if stage4_query.question_type == "RELATIONSHIP" and len(entities) == 1 and stage4_query.relation:
            primary_entity = entities[0]
            
            # Fetch document for primary entity with organizational disambiguation if needed
            search_query = primary_entity.normalized_text
            if primary_entity.type == "ORGANIZATION" and search_query.lower() in ["apple", "amazon", "meta", "alphabet"]:
                search_query = f"{primary_entity.normalized_text} Inc."
            elif stage4_query.relation.type == "FATHER_OF_NATION_OF":
                search_query = f"Father of the Nation {primary_entity.normalized_text}"
            elif stage4_query.relation.type in ("OFFICE_HOLDERS_OF", "ORDINAL_OFFICE_HOLDER"):
                if stage4_query.retrieval_terms and len(stage4_query.retrieval_terms) > 0:
                    search_query = stage4_query.retrieval_terms[0]
                else:
                    q_low = question.lower()
                    office = "rulers" if "ruler" in q_low else "prime ministers" if "prime minister" in q_low else "presidents" if "president" in q_low else "monarchs" if "monarch" in q_low else "kings" if "king" in q_low else "emperors" if "emperor" in q_low else "leaders"
                    search_query = f"{office} of {primary_entity.normalized_text}"
            elif stage4_query.relation.type == "FOUNDED_BY" and "dynasty" in primary_entity.normalized_text.lower():
                search_query = re.sub(r"\bdynasty\b", "Empire", primary_entity.normalized_text, flags=re.IGNORECASE).strip()

            doc_resp = await self.doc_retriever.retrieve_document(
                query=search_query,
                explicit_title=None,
            )
            if (doc_resp.status != "success" or not doc_resp.document) and search_query != primary_entity.normalized_text:
                logger.info("Search with '%s' yielded no doc, falling back to primary entity: '%s'", search_query, primary_entity.normalized_text)
                doc_resp = await self.doc_retriever.retrieve_document(
                    query=primary_entity.normalized_text,
                    explicit_title=None,
                )


            if doc_resp.status != "success" or not doc_resp.document:
                diagnostics = Stage4Diagnostics(
                    question_type_detected=stage4_query.question_type,
                    entities_identified=entity_names,
                    relationship_detected=stage4_query.relation.type if stage4_query.relation else None,
                    documents_retrieved=0,
                    passages_evaluated=0,
                    relationship_evidence_found=False,
                    failure_stage="DOCUMENT_RETRIEVAL",
                    details=f"Could not retrieve Wikipedia document for entity: '{primary_entity.normalized_text}'.",
                )
                return MultiDocRetrievalResponse(
                    question=question,
                    structured_query=stage4_query,
                    entities_identified=entities,
                    documents=[],
                    relationships_discovered=[],
                    relationship_evidence=[],
                    diagnostics=diagnostics,
                    status="DOCUMENT_NOT_FOUND",
                    is_relationship_proven=False,
                    validation_reason=f"Target Wikipedia article for '{primary_entity.normalized_text}' could not be found.",
                    total_passages_evaluated=0,
                )

            doc: Document = doc_resp.document
            doc_passages = self.passage_builder.build_passages(doc)
            total_passages_evaluated = len(doc_passages)

            # Score passages with BM25 using relation-expanded retrieval terms
            bm25_scorer = BM25RelevanceScorer(doc_passages)
            scored_items = bm25_scorer.score_query(stage4_query.retrieval_terms)

            doc_evidence_passages: List[EvidencePassage] = []
            for rank, (p, score, matched, unmatched) in enumerate(
                scored_items[: request.top_k_per_document * 2], start=1
            ):
                if score >= request.min_score:
                    ev_item = EvidencePassage(
                        rank=rank,
                        passage_id=p.passage_id,
                        document_title=p.document_title,
                        section_title=p.section_title,
                        section_level=p.section_level,
                        paragraph_id=p.paragraph_id,
                        text=p.text,
                        score=score,
                        is_above_threshold=True,
                        matched_terms=matched,
                        unmatched_terms=unmatched,
                    )
                    doc_evidence_passages.append(ev_item)

            source_doc_ref = SourceDocumentReference(
                title=doc.title,
                page_id=doc.page_id,
                url=doc.url,
            )

            # Detect directional relationship & extract object entity from passages
            relationships, rel_items = RelationshipDetector.detect_single_entity_relationship(
                query=stage4_query,
                subject_entity=primary_entity,
                passages=doc_evidence_passages,
                source_doc_ref=source_doc_ref,
            )

            bundle = DocumentEvidenceBundle(
                entity_name=primary_entity.normalized_text,
                document_title=doc.title,
                page_id=doc.page_id,
                url=doc.url,
                evidence_passages=doc_evidence_passages[: request.top_k_per_document],
            )

            if relationships and len(relationships) > 0 and relationships[0].confidence_score >= 0.75:
                top_rel = relationships[0]
                status = "RELATIONSHIP_SUPPORTED"
                is_proven = True
                reason = (
                    f"Verified {top_rel.relation.upper()} for '{top_rel.subject}' -> '{top_rel.object}' "
                    f"supported by document '{doc.title}'."
                )
                diagnostics = Stage4Diagnostics(
                    question_type_detected=stage4_query.question_type,
                    entities_identified=entity_names,
                    relationship_detected=stage4_query.relation.type,
                    documents_retrieved=1,
                    passages_evaluated=total_passages_evaluated,
                    relationship_evidence_found=True,
                    failure_stage="NONE",
                    details=f"Discovered relation '{top_rel.relation}' with object '{top_rel.object}'.",
                )
            elif len(doc_evidence_passages) > 0:
                status = "PARTIALLY_SUPPORTED"
                is_proven = False
                reason = (
                    f"Retrieved {len(doc_evidence_passages)} relevant passages for '{primary_entity.normalized_text}', "
                    f"but exact '{stage4_query.relation.type}' target was not definitively extracted."
                )
                diagnostics = Stage4Diagnostics(
                    question_type_detected=stage4_query.question_type,
                    entities_identified=entity_names,
                    relationship_detected=stage4_query.relation.type,
                    documents_retrieved=1,
                    passages_evaluated=total_passages_evaluated,
                    relationship_evidence_found=False,
                    failure_stage="RELATIONSHIP_VALIDATION",
                    details="Passages matched lexically, but did not satisfy relationship pattern constraints.",
                )
            else:
                status = "INSUFFICIENT_EVIDENCE"
                is_proven = False
                reason = f"No passages in '{doc.title}' met the relevance score threshold ({request.min_score})."
                diagnostics = Stage4Diagnostics(
                    question_type_detected=stage4_query.question_type,
                    entities_identified=entity_names,
                    relationship_detected=stage4_query.relation.type,
                    documents_retrieved=1,
                    passages_evaluated=total_passages_evaluated,
                    relationship_evidence_found=False,
                    failure_stage="PASSAGE_RETRIEVAL",
                    details="No passages reached the minimum score threshold.",
                )

            return MultiDocRetrievalResponse(
                question=question,
                structured_query=stage4_query,
                entities_identified=entities,
                documents=[bundle],
                relationships_discovered=relationships,
                relationship_evidence=rel_items,
                diagnostics=diagnostics,
                status=status,
                is_relationship_proven=is_proven,
                validation_reason=reason,
                total_passages_evaluated=total_passages_evaluated,
            )

        # 3. Execution Path B: Multi-Entity Queries (e.g. Einstein & University of Zurich, India & Pakistan)
        document_bundles: List[DocumentEvidenceBundle] = []
        all_evaluated_passages: List[EvidencePassage] = []
        doc_refs: Dict[str, SourceDocumentReference] = {}
        total_passages_evaluated = 0

        for i, entity_item in enumerate(entities):
            other_entities = [e.normalized_text for j, e in enumerate(entities) if j != i]

            doc_resp = await self.doc_retriever.retrieve_document(
                query=entity_item.normalized_text,
                explicit_title=None,
            )

            if doc_resp.status != "success" or not doc_resp.document:
                continue

            doc: Document = doc_resp.document
            doc_refs[doc.title] = SourceDocumentReference(
                title=doc.title,
                page_id=doc.page_id,
                url=doc.url,
            )
            doc_passages = self.passage_builder.build_passages(doc)
            total_passages_evaluated += len(doc_passages)

            if not doc_passages:
                continue

            cross_search_terms = list(other_entities) + [
                tok for tok in question.lower().split()
                if tok not in ["what", "is", "the", "relationship", "between", "and", "how", "are", "difference", "compare"]
            ]

            bm25_scorer = BM25RelevanceScorer(doc_passages)
            scored_items = bm25_scorer.score_query(cross_search_terms)

            doc_evidence_passages: List[EvidencePassage] = []
            for rank, (p, score, matched, unmatched) in enumerate(
                scored_items[: request.top_k_per_document * 2], start=1
            ):
                if score >= request.min_score:
                    ev_item = EvidencePassage(
                        rank=rank,
                        passage_id=p.passage_id,
                        document_title=p.document_title,
                        section_title=p.section_title,
                        section_level=p.section_level,
                        paragraph_id=p.paragraph_id,
                        text=p.text,
                        score=score,
                        is_above_threshold=True,
                        matched_terms=matched,
                        unmatched_terms=unmatched,
                    )
                    doc_evidence_passages.append(ev_item)
                    all_evaluated_passages.append(ev_item)

            document_bundles.append(
                DocumentEvidenceBundle(
                    entity_name=entity_item.normalized_text,
                    document_title=doc.title,
                    page_id=doc.page_id,
                    url=doc.url,
                    evidence_passages=doc_evidence_passages[: request.top_k_per_document],
                )
            )

        # Detect cross-document relationships
        relationships, rel_items = RelationshipDetector.detect_cross_document_relationships(
            entities=entities,
            passages=all_evaluated_passages,
            doc_refs=doc_refs,
        )

        if len(relationships) > 0 and relationships[0].confidence_score >= 0.75:
            top_rel = relationships[0]
            status = "RELATIONSHIP_SUPPORTED"
            is_proven = True
            reason = (
                f"Discovered verified relationship '{top_rel.relation}' between "
                f"'{top_rel.subject}' and '{top_rel.object}' with {len(top_rel.supporting_evidence_ids)} supporting cross-document passages."
            )
            diagnostics = Stage4Diagnostics(
                question_type_detected=stage4_query.question_type,
                entities_identified=entity_names,
                relationship_detected=top_rel.relation,
                documents_retrieved=len(document_bundles),
                passages_evaluated=total_passages_evaluated,
                relationship_evidence_found=True,
                failure_stage="NONE",
                details=f"Discovered cross-document relation '{top_rel.relation}'.",
            )
        elif len(all_evaluated_passages) > 0:
            status = "PARTIALLY_SUPPORTED"
            is_proven = False
            reason = (
                f"Retrieved {len(document_bundles)} documents with {len(all_evaluated_passages)} relevant passages, "
                "providing contextual evidence across entities."
            )
            diagnostics = Stage4Diagnostics(
                question_type_detected=stage4_query.question_type,
                entities_identified=entity_names,
                relationship_detected=stage4_query.relation.type if stage4_query.relation else None,
                documents_retrieved=len(document_bundles),
                passages_evaluated=total_passages_evaluated,
                relationship_evidence_found=False,
                failure_stage="RELATIONSHIP_VALIDATION",
                details="Passages retrieved across documents, but formal relationship predicate confidence below threshold.",
            )
        else:
            status = "INSUFFICIENT_EVIDENCE"
            is_proven = False
            reason = "No cross-document evidence passages reached the relevance threshold."
            diagnostics = Stage4Diagnostics(
                question_type_detected=stage4_query.question_type,
                entities_identified=entity_names,
                relationship_detected=stage4_query.relation.type if stage4_query.relation else None,
                documents_retrieved=len(document_bundles),
                passages_evaluated=total_passages_evaluated,
                relationship_evidence_found=False,
                failure_stage="PASSAGE_RETRIEVAL",
                details="No cross-document passages reached the minimum score threshold.",
            )

        return MultiDocRetrievalResponse(
            question=question,
            structured_query=stage4_query,
            entities_identified=entities,
            documents=document_bundles,
            relationships_discovered=relationships,
            relationship_evidence=rel_items,
            diagnostics=diagnostics,
            status=status,
            is_relationship_proven=is_proven,
            validation_reason=reason,
            total_passages_evaluated=total_passages_evaluated,
        )

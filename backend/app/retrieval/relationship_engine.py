"""Stage 5 Relationship Discovery & Evidence Linking Engine.

Orchestrates cross-document article acquisition, BM25 candidate harvesting,
predicate matching, evidence chain consolidation, and in-memory evidence graph synthesis.
"""

from typing import List, Optional, Dict
from app.schemas.retrieval import (
    EntityItem,
    EvidencePassage,
    SourceDocumentReference,
    MultiDocRetrievalRequest,
    MultiDocRetrievalResponse,
)
from app.schemas.relationships import (
    RelationshipDiscoveryRequest,
    RelationshipDiscoveryResponse,
    Stage5EvidenceGraph,
    DiscoveredRelationshipEdge,
    EvidenceProvenanceItem,
)
from app.retrieval.multi_doc_retriever import MultiDocRetriever
from app.retrieval.evidence_consolidator import EvidenceConsolidator
from app.retrieval.entity_extractor import EntityExtractor
from app.retrieval.predicate_registry import PredicateRegistry


class RelationshipEngine:
    """Stage 5 Core Engine: Discovers, validates, and links grounded relationship edges across documents."""

    def __init__(self, multi_doc_retriever: Optional[MultiDocRetriever] = None):
        self.multi_doc_retriever = multi_doc_retriever or MultiDocRetriever()

    async def discover_relationships(
        self, request: RelationshipDiscoveryRequest
    ) -> RelationshipDiscoveryResponse:
        """Execute Stage 5 relationship discovery and evidence linking."""
        question = request.question.strip()
        if not question:
            return RelationshipDiscoveryResponse(
                question="",
                structured_query=None,
                evidence_graph=Stage5EvidenceGraph(),
                status="ERROR",
                primary_relationship=None,
                all_relationships=[],
                validation_summary="Question cannot be empty.",
                error_message="Question cannot be empty.",
            )

        # 1. Analyze Question Intent & Entities
        stage4_query = EntityExtractor.analyze_stage4_query(question)

        # Only pass entities override if explicitly supplied by caller
        manual_entities = None
        if request.subject_entity or request.object_entity:
            manual_entities = []
            if request.subject_entity:
                manual_entities.append(request.subject_entity.strip())
            if request.object_entity:
                manual_entities.append(request.object_entity.strip())

        # 2. Retrieve Candidate Documents and Evidence Passages via MultiDocRetriever
        multi_doc_req = MultiDocRetrievalRequest(
            question=question,
            entities=manual_entities,
            top_k_per_document=request.top_k_per_document,
            min_score=0.10,  # Broader recall window for Stage 5 relational harvesting
        )

        stage4_resp: MultiDocRetrievalResponse = await self.multi_doc_retriever.retrieve_multi_document(multi_doc_req)

        if stage4_resp.status == "DOCUMENT_NOT_FOUND":
            return RelationshipDiscoveryResponse(
                question=question,
                structured_query=stage4_query,
                evidence_graph=Stage5EvidenceGraph(),
                status="DOCUMENT_NOT_FOUND",
                primary_relationship=None,
                all_relationships=[],
                validation_summary=stage4_resp.validation_reason,
            )

        # 3. Harvest All Passages and Build Document Provenance Map
        all_passages: List[EvidencePassage] = []
        doc_refs: Dict[str, SourceDocumentReference] = {}

        for bundle in stage4_resp.documents:
            doc_refs[bundle.document_title] = SourceDocumentReference(
                title=bundle.document_title,
                page_id=bundle.page_id,
                url=bundle.url,
            )
            all_passages.extend(bundle.evidence_passages)

        # 4. Consolidate Evidence into Grounded Relationship Edges
        target_predicate = request.predicate_filter
        if not target_predicate and stage4_query.relation:
            target_predicate = stage4_query.relation.type

        discovered_edges = EvidenceConsolidator.consolidate_evidence(
            entities=stage4_resp.entities_identified,
            passages=all_passages,
            doc_refs=doc_refs,
            query_predicate=target_predicate,
            min_confidence=request.min_confidence,
        )

        # Also incorporate direct Stage 4 relationship evidence items
        if stage4_resp.relationship_evidence:
            for rel_item in stage4_resp.relationship_evidence:
                edge_id = f"rel_{rel_item.subject.lower()}_{rel_item.relation.lower()}_{rel_item.object.lower()}".replace(" ", "_")
                sub_ent = EntityItem(
                    text=rel_item.subject,
                    normalized_text=rel_item.subject,
                    type=EntityExtractor._infer_entity_type(rel_item.subject),
                    role="SUBJECT",
                )
                obj_ent = EntityItem(
                    text=rel_item.object,
                    normalized_text=rel_item.object,
                    type=EntityExtractor._infer_entity_type(rel_item.object),
                    role="OBJECT",
                )
                prov = EvidenceProvenanceItem(
                    document_title=rel_item.source_document.title,
                    page_id=rel_item.source_document.page_id,
                    url=rel_item.source_document.url,
                    section_title=rel_item.source_section,
                    section_level=2,
                    paragraph_id=rel_item.source_paragraph,
                    passage_text=rel_item.passage_text,
                    exact_sentence=rel_item.summary_statement,
                    support_score=rel_item.relationship_score,
                    lexical_cues_matched=[rel_item.relation],
                )
                direct_edge = DiscoveredRelationshipEdge(
                    edge_id=edge_id,
                    subject=sub_ent,
                    predicate=rel_item.relation.upper(),
                    inverse_predicate=PredicateRegistry.get_inverse(rel_item.relation.upper()),
                    object=obj_ent,
                    category=stage4_query.relation.type if stage4_query.relation else "GENERAL",
                    status="PROVEN" if rel_item.validated else "PARTIALLY_SUPPORTED",
                    confidence_score=rel_item.relationship_score,
                    evidence_chain=[prov],
                    distinct_documents_count=1,
                    synthesis_statement=rel_item.summary_statement,
                )
                # Prepend if not already present
                if not any(e.subject.normalized_text.lower() == sub_ent.normalized_text.lower() and e.object.normalized_text.lower() == obj_ent.normalized_text.lower() for e in discovered_edges):
                    discovered_edges.insert(0, direct_edge)

        # 5. Assemble In-Memory Evidence Graph
        graph_entities: List[EntityItem] = list(stage4_resp.entities_identified)
        # Add discovered object entities into graph if not already present
        known_ent_names = {e.normalized_text.lower() for e in graph_entities}
        for edge in discovered_edges:
            if edge.object.normalized_text.lower() not in known_ent_names:
                graph_entities.append(edge.object)
                known_ent_names.add(edge.object.normalized_text.lower())


        provenance_summary = (
            f"Evidence graph constructed from {len(stage4_resp.documents)} Wikipedia article(s) "
            f"({', '.join(d.document_title for d in stage4_resp.documents)}) with "
            f"{stage4_resp.total_passages_evaluated} total passages evaluated."
        )

        cand_provenance: List[EvidenceProvenanceItem] = []
        for p in all_passages:
            doc_ref = doc_refs.get(p.document_title)
            cand_provenance.append(
                EvidenceProvenanceItem(
                    document_title=p.document_title,
                    page_id=doc_ref.page_id if doc_ref else 0,
                    url=doc_ref.url if doc_ref else f"https://en.wikipedia.org/wiki/{p.document_title.replace(' ', '_')}",
                    section_title=p.section_title or "General",
                    section_level=p.section_level,
                    paragraph_id=p.paragraph_id,
                    passage_text=p.text,
                    exact_sentence=p.text[:200],
                    support_score=p.score,
                    lexical_cues_matched=p.matched_terms,
                )
            )

        evidence_graph = Stage5EvidenceGraph(
            entities=graph_entities,
            relationships=discovered_edges,
            total_documents_consulted=len(stage4_resp.documents),
            total_passages_evaluated=stage4_resp.total_passages_evaluated,
            graph_provenance_summary=provenance_summary,
            candidate_passages=cand_provenance,
        )

        # 6. Formulate Overall Response Status
        primary_edge = discovered_edges[0] if discovered_edges else None

        if primary_edge and primary_edge.status == "PROVEN":
            status = "PROVEN_RELATIONSHIP"
            summary = (
                f"Verified PROVEN relationship: {primary_edge.subject.normalized_text} —[{primary_edge.predicate}]—> "
                f"{primary_edge.object.normalized_text} (Confidence: {primary_edge.confidence_score}, "
                f"Corroborated across {primary_edge.distinct_documents_count} document(s))."
            )
        elif len(discovered_edges) > 1:
            status = "MULTI_RELATIONSHIP_DISCOVERED"
            summary = f"Discovered {len(discovered_edges)} candidate relationship edges linking {len(graph_entities)} entities."
        elif primary_edge and primary_edge.status == "PARTIALLY_SUPPORTED":
            status = "PARTIALLY_SUPPORTED"
            summary = (
                f"Partially supported relationship: {primary_edge.subject.normalized_text} —[{primary_edge.predicate}]—> "
                f"{primary_edge.object.normalized_text} (Confidence: {primary_edge.confidence_score})."
            )
        else:
            status = "NO_RELATIONSHIP_FOUND"
            summary = "No verified relationship could be proven from the retrieved document evidence."

        return RelationshipDiscoveryResponse(
            question=question,
            structured_query=stage4_query,
            evidence_graph=evidence_graph,
            status=status,
            primary_relationship=primary_edge,
            all_relationships=discovered_edges,
            validation_summary=summary,
        )

"""Stage 5 Schemas: Relationship Discovery, Evidence Linking, and In-Memory Evidence Graphs."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.schemas.retrieval import (
    EntityItem,
    Stage4StructuredQuery,
    SourceDocumentReference,
)


class EvidenceProvenanceItem(BaseModel):
    """Detailed provenance for an individual piece of supporting evidence."""
    document_title: str
    page_id: int
    url: str
    section_title: str
    section_level: int
    paragraph_id: str
    passage_text: str
    exact_sentence: str
    support_score: float = Field(..., description="Relevance & extraction confidence score (0.0 to 1.0)")
    lexical_cues_matched: List[str] = Field(default_factory=list)


class DiscoveredRelationshipEdge(BaseModel):
    """A discovered, evidence-backed relationship edge between two entities."""
    edge_id: str = Field(..., description="Unique deterministic edge ID")
    subject: EntityItem
    predicate: str = Field(..., description="Canonical predicate: e.g. 'STUDIED_AT', 'SUCCESSOR_OF', 'FOUNDED_BY'")
    inverse_predicate: Optional[str] = Field(default=None, description="Dual inverse predicate: e.g. 'PREDECESSOR_OF', 'ALMA_MATER_OF'")
    object: EntityItem
    category: str = Field(
        default="GENERAL",
        description="'ACADEMIC_INSTITUTIONAL', 'SUCCESSION_GOVERNANCE', 'FOUNDER_CORPORATE', 'GEOPOLITICAL_TERRITORIAL', 'INVENTION_CREATION', 'SOCIAL_COLLABORATION', 'GENERAL_ASSOCIATION'"
    )
    status: str = Field(..., description="'PROVEN', 'PARTIALLY_SUPPORTED', or 'UNSUPPORTED'")
    confidence_score: float = Field(..., description="Consolidated confidence score (0.0 to 1.0)")
    evidence_chain: List[EvidenceProvenanceItem] = Field(default_factory=list)
    distinct_documents_count: int = Field(default=1, description="Number of distinct source Wikipedia articles supporting this edge")
    synthesis_statement: str = Field(..., description="Synthesized grounded statement explaining the relation")


class Stage5EvidenceGraph(BaseModel):
    """In-memory evidence graph linking discovered entities and verified relationship edges."""
    entities: List[EntityItem] = Field(default_factory=list)
    relationships: List[DiscoveredRelationshipEdge] = Field(default_factory=list)
    total_documents_consulted: int = 0
    total_passages_evaluated: int = 0
    graph_provenance_summary: str = ""
    candidate_passages: List[EvidenceProvenanceItem] = Field(default_factory=list)


class RelationshipDiscoveryRequest(BaseModel):
    """Request schema for Stage 5 relationship discovery & evidence linking."""
    question: str = Field(..., description="Natural language question to discover and link relationships for")
    subject_entity: Optional[str] = Field(default=None, description="Optional explicit subject entity override")
    object_entity: Optional[str] = Field(default=None, description="Optional explicit object entity override")
    predicate_filter: Optional[str] = Field(default=None, description="Optional target predicate filter")
    top_k_per_document: int = Field(default=4, ge=1, le=10, description="Passages to evaluate per document")
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold for relationship inclusion")


class RelationshipDiscoveryResponse(BaseModel):
    """Response schema for Stage 5 relationship discovery & evidence linking."""
    question: str
    structured_query: Optional[Stage4StructuredQuery] = None
    evidence_graph: Stage5EvidenceGraph
    status: str = Field(
        ...,
        description="'PROVEN_RELATIONSHIP', 'MULTI_RELATIONSHIP_DISCOVERED', 'PARTIALLY_SUPPORTED', 'NO_RELATIONSHIP_FOUND', 'DOCUMENT_NOT_FOUND', or 'ERROR'"
    )
    primary_relationship: Optional[DiscoveredRelationshipEdge] = None
    all_relationships: List[DiscoveredRelationshipEdge] = Field(default_factory=list)
    validation_summary: str
    error_message: Optional[str] = None

"""Stage 6 Schemas: Advanced Evidence Reasoning, Clustering, Conflict Detection, and LLM-Ready Evidence Package."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from app.schemas.retrieval import Stage4StructuredQuery
from app.schemas.relationships import EvidenceProvenanceItem


class EvidenceItem(BaseModel):
    """A granular, deduplicated, and re-scored unit of textual evidence."""
    item_id: str = Field(..., description="Unique deterministic identifier for the evidence item")
    text: str = Field(..., description="Canonical passage or sentence text")
    source_document: str = Field(..., description="Source article/document title")
    source_section: str = Field(..., description="Source section title")
    relevance_score: float = Field(..., description="Re-scored relevance to user question (0.0 to 1.0)")
    support_class: Literal["DIRECT_ANSWER", "SUPPORTING", "CONTEXTUAL", "IRRELEVANT"] = Field(
        ..., description="Classification of evidence utility"
    )
    is_duplicate: bool = Field(default=False, description="Whether this item is marked as a near-duplicate")
    provenance: Optional[EvidenceProvenanceItem] = Field(
        default=None, description="Original Stage 5 provenance metadata if available"
    )


class EvidenceCluster(BaseModel):
    """A thematic cluster of related evidence items grouped by document and section/topic."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    cluster_topic: str = Field(..., description="Topic label or source document/section heading")
    items: List[EvidenceItem] = Field(default_factory=list, description="Evidence items belonging to this cluster")
    cluster_relevance: float = Field(default=0.0, description="Aggregate relevance score of the cluster")


class EvidenceConflict(BaseModel):
    """Contradiction or factual discrepancy detected across evidence items."""
    conflict_id: str = Field(..., description="Unique conflict identifier")
    description: str = Field(..., description="Description of the factual discrepancy")
    conflicting_items: List[str] = Field(default_factory=list, description="IDs of items involved in the conflict")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(..., description="Conflict severity level")


class SufficiencyAssessment(BaseModel):
    """Deterministic evaluation of whether the evidence package is adequate to answer the user question."""
    status: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] = Field(
        ..., description="Overall evidence sufficiency rating"
    )
    confidence: float = Field(..., description="Confidence in the sufficiency rating (0.0 to 1.0)")
    missing_aspects: List[str] = Field(default_factory=list, description="Identified missing entities, dates, or relations")
    reasoning: str = Field(..., description="Deterministic justification for the sufficiency rating")


class AnswerCandidate(BaseModel):
    """An extracted entity or factual assertion grounded directly in evidence."""
    candidate_text: str = Field(..., description="Candidate answer string or statement")
    support_count: int = Field(default=1, description="Number of distinct evidence items supporting this candidate")
    supporting_items: List[str] = Field(default_factory=list, description="IDs of supporting evidence items")
    confidence: float = Field(default=0.5, description="Confidence score for this answer candidate")


class EvidencePackage(BaseModel):
    """Complete, structured, LLM-ready evidence package produced by Stage 6."""
    question: str = Field(..., description="Original user query")
    question_intent: Optional[Stage4StructuredQuery] = Field(
        default=None, description="Stage 4 structured intent representation"
    )
    evidence_clusters: List[EvidenceCluster] = Field(
        default_factory=list, description="Grouped thematic evidence clusters"
    )
    conflicts: List[EvidenceConflict] = Field(
        default_factory=list, description="Detected discrepancies among evidence items"
    )
    sufficiency_assessment: SufficiencyAssessment = Field(
        ..., description="Deterministic sufficiency evaluation"
    )
    answer_candidates: List[AnswerCandidate] = Field(
        default_factory=list, description="Grounded candidate answers surfaced from evidence"
    )
    total_evidence_items: int = Field(default=0, description="Total evidence items collected (including duplicates)")
    total_clusters: int = Field(default=0, description="Total evidence clusters formed")
    total_conflicts: int = Field(default=0, description="Total conflicts detected")
    deduplication_summary: str = Field(default="", description="Summary of deduplication operations performed")
    stage6_status: str = Field(default="COMPLETED", description="Stage 6 execution status")


class EvidenceReasoningRequest(BaseModel):
    """Request schema for Stage 6 evidence reasoning and synthesis."""
    question: str = Field(..., description="Natural language question")
    subject_entity: Optional[str] = Field(default=None, description="Optional explicit subject entity override")
    object_entity: Optional[str] = Field(default=None, description="Optional explicit object entity override")
    predicate_filter: Optional[str] = Field(default=None, description="Optional target predicate filter")
    top_k_per_document: int = Field(default=4, ge=1, le=10, description="Passages to evaluate per document")
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold")
    dedup_similarity_threshold: float = Field(
        default=0.75, ge=0.1, le=1.0, description="Jaccard similarity threshold for near-duplicate detection"
    )


class EvidenceReasoningResponse(BaseModel):
    """Response schema for Stage 6 evidence reasoning."""
    evidence_package: EvidencePackage
    status: str = Field(..., description="'COMPLETED', 'PARTIALLY_SUPPORTED', 'NO_EVIDENCE', or 'ERROR'")
    validation_summary: str
    error_message: Optional[str] = None

"""Pydantic schemas for Stage 7 Grounded Answer API."""

from typing import List, Literal, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from app.schemas.evidence_package import EvidencePackage


class AnswerRequest(BaseModel):
    """Request schema for grounded answer generation."""

    question: str = Field(..., min_length=2, description="User question to answer")
    subject_entity: Optional[str] = Field(default=None, description="Optional subject entity hint")
    object_entity: Optional[str] = Field(default=None, description="Optional object entity hint")
    predicate_filter: Optional[str] = Field(default=None, description="Optional relationship predicate filter")
    top_k_per_document: int = Field(default=4, ge=1, le=10, description="Passages per document for initial stages")
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_evidence_passages: int = Field(default=3, ge=1, le=12, description="Maximum evidence passages sent to LLM")
    force_refresh: bool = Field(default=False, description="Bypass document cache and re-fetch from Wikipedia")
    conversation_id: Optional[UUID] = Field(default=None, description="Optional active conversation session UUID")


class SourceReference(BaseModel):
    """Individual source passage referenced in the generated answer."""

    chunk_id: int
    document_title: str
    source_url: str
    section_title: str
    passage_text: str
    rrf_score: float
    retrieval_method: str  # "vector" | "keyword" | "hybrid"


class TimingBreakdown(BaseModel):
    """Fine-grained monotonic high-resolution timings in milliseconds."""

    total_request_ms: float = 0.0
    stage6_reasoning_ms: float = 0.0
    db_cache_lookup_ms: float = 0.0
    wikipedia_network_ms: float = 0.0
    chunking_ms: float = 0.0
    embedding_generation_ms: float = 0.0
    embedding_batch_count: int = 0
    db_write_ms: float = 0.0
    vector_retrieval_ms: float = 0.0
    keyword_retrieval_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    hybrid_retrieval_total_ms: float = 0.0
    evidence_selection_prompt_ms: float = 0.0
    llm_request_ms: float = 0.0
    llm_load_duration_ms: Optional[float] = None
    llm_prompt_eval_duration_ms: Optional[float] = None
    llm_eval_duration_ms: Optional[float] = None
    llm_tokens_per_second: Optional[float] = None
    llm_prompt_tokens: Optional[int] = None
    llm_response_tokens: Optional[int] = None
    llm_call_count: int = 0
    response_serialization_ms: float = 0.0


class RetrievalDiagnostics(BaseModel):
    """Diagnostic metrics and metadata for retrieval execution."""

    documents_checked: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    documents_fetched: int = 0
    chunks_in_db: int = 0
    vector_candidates: int = 0
    keyword_candidates: int = 0
    passages_sent_to_llm: int = 0
    embedding_model: str
    llm_model: str
    timings: Optional[TimingBreakdown] = None
    query_complexity: Optional[str] = None
    subqueries_generated: Optional[List[str]] = None
    evidence_coverage_ratio: Optional[float] = None
    missing_aspects: Optional[List[str]] = None
    claims_verified: int = 0
    claims_supported: int = 0
    unsupported_claims_removed: int = 0
    citations_stripped: int = 0



class AnswerResponse(BaseModel):
    """Response schema returned by the POST /retrieval/answer endpoint."""

    question: str
    answer_text: str
    answer_status: Literal[
        "ANSWERED",
        "INSUFFICIENT_EVIDENCE",
        "CONFLICTING_EVIDENCE",
        "RETRIEVAL_FAILED",
        "LLM_FAILED",
    ]
    source_references: List[SourceReference] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    sufficiency_rating: str = "UNKNOWN"
    retrieval_diagnostics: RetrievalDiagnostics
    stage6_evidence_package: Optional[EvidencePackage] = None
    error_message: Optional[str] = None
    conversation_id: Optional[UUID] = None
    message_id: Optional[UUID] = None

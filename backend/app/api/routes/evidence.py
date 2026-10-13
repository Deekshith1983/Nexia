"""API routes for Stage 6: Advanced Evidence Reasoning & Synthesis."""

from typing import Optional
from fastapi import APIRouter, Query, Body
from app.schemas.evidence_package import (
    EvidenceReasoningRequest,
    EvidenceReasoningResponse,
)
from app.retrieval.evidence_reasoner import EvidenceReasoner

router = APIRouter(tags=["Stage 6 - Evidence Reasoning"])
evidence_reasoner = EvidenceReasoner()


@router.post("/retrieval/evidence", response_model=EvidenceReasoningResponse)
@router.post("/retrieve/evidence", response_model=EvidenceReasoningResponse, include_in_schema=False)
async def reason_evidence_post(
    request: EvidenceReasoningRequest = Body(...),
) -> EvidenceReasoningResponse:
    """Stage 6: Advanced Evidence Reasoning, Clustering, Deduplication, and Synthesis (POST)."""
    return await evidence_reasoner.reason(request)


@router.get("/retrieval/evidence", response_model=EvidenceReasoningResponse)
@router.get("/retrieve/evidence", response_model=EvidenceReasoningResponse, include_in_schema=False)
async def reason_evidence_get(
    question: str = Query(..., description="Natural language question to reason evidence for"),
    subject_entity: Optional[str] = Query(default=None, description="Optional subject entity override"),
    object_entity: Optional[str] = Query(default=None, description="Optional object entity override"),
    predicate_filter: Optional[str] = Query(default=None, description="Optional predicate filter"),
    top_k_per_document: int = Query(default=4, ge=1, le=10, description="Passages to evaluate per document"),
    min_confidence: float = Query(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    dedup_similarity_threshold: float = Query(default=0.75, ge=0.1, le=1.0, description="Deduplication threshold"),
) -> EvidenceReasoningResponse:
    """Stage 6: Advanced Evidence Reasoning, Clustering, Deduplication, and Synthesis (GET)."""
    request = EvidenceReasoningRequest(
        question=question,
        subject_entity=subject_entity,
        object_entity=object_entity,
        predicate_filter=predicate_filter,
        top_k_per_document=top_k_per_document,
        min_confidence=min_confidence,
        dedup_similarity_threshold=dedup_similarity_threshold,
    )
    return await evidence_reasoner.reason(request)

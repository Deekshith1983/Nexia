"""API routes for Stage 5: Relationship Discovery & Evidence Linking."""

from typing import Optional
from fastapi import APIRouter, Query, Body, HTTPException
from app.schemas.relationships import (
    RelationshipDiscoveryRequest,
    RelationshipDiscoveryResponse,
)
from app.retrieval.relationship_engine import RelationshipEngine

router = APIRouter(tags=["Stage 5 - Relationships"])
relationship_engine = RelationshipEngine()


@router.post("/retrieval/relationships", response_model=RelationshipDiscoveryResponse)
@router.post("/retrieve/relationships", response_model=RelationshipDiscoveryResponse, include_in_schema=False)
async def discover_relationships_post(
    request: RelationshipDiscoveryRequest = Body(...),
) -> RelationshipDiscoveryResponse:
    """Stage 5: Discover and link grounded relationship edges across documents with provenance (POST)."""
    return await relationship_engine.discover_relationships(request)


@router.get("/retrieval/relationships", response_model=RelationshipDiscoveryResponse)
@router.get("/retrieve/relationships", response_model=RelationshipDiscoveryResponse, include_in_schema=False)
async def discover_relationships_get(
    question: str = Query(..., description="Question to discover relationships for"),
    subject_entity: Optional[str] = Query(default=None, description="Optional subject entity override"),
    object_entity: Optional[str] = Query(default=None, description="Optional object entity override"),
    predicate_filter: Optional[str] = Query(default=None, description="Optional predicate filter: e.g. 'STUDIED_AT', 'SUCCESSOR_OF', 'FOUNDED_BY'"),
    top_k_per_document: int = Query(default=4, ge=1, le=10, description="Passages to evaluate per document"),
    min_confidence: float = Query(default=0.60, ge=0.0, le=1.0, description="Minimum confidence threshold"),
) -> RelationshipDiscoveryResponse:
    """Stage 5: Discover and link grounded relationship edges across documents with provenance (GET)."""
    request = RelationshipDiscoveryRequest(
        question=question,
        subject_entity=subject_entity,
        object_entity=object_entity,
        predicate_filter=predicate_filter,
        top_k_per_document=top_k_per_document,
        min_confidence=min_confidence,
    )
    return await relationship_engine.discover_relationships(request)

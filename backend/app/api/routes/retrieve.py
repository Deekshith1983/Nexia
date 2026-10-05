"""API routes for Document Retrieval (Stage 1)."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from app.schemas.retrieval import SearchResponse, DocumentRetrievalResponse
from app.retrieval.document_retriever import DocumentRetriever

router = APIRouter(prefix="/retrieve", tags=["Retrieval"])
retriever = DocumentRetriever()


@router.get("/search", response_model=SearchResponse)
async def search_candidates(
    query: str = Query(..., description="Topic or entity to search on Wikipedia"),
    limit: int = Query(default=10, ge=1, le=20, description="Max candidate results"),
) -> SearchResponse:
    """Stage 1A: Search Wikipedia and inspect raw candidate articles."""
    return await retriever.search_candidates(query=query, limit=limit)


@router.get("/document", response_model=DocumentRetrievalResponse)
async def retrieve_document(
    query: str = Query(default="", description="Search query or topic"),
    title: Optional[str] = Query(default=None, description="Explicit Wikipedia article title to fetch directly"),
) -> DocumentRetrievalResponse:
    """Stage 1B/1C/1D: Retrieve and inspect full structured Wikipedia document."""
    if not query and not title:
        raise HTTPException(status_code=400, detail="Must provide either 'query' or 'title' parameter.")
    
    return await retriever.retrieve_document(query=query, explicit_title=title)


@router.get("", response_model=DocumentRetrievalResponse)
async def retrieve_default(
    query: str = Query(..., description="Search query or topic"),
) -> DocumentRetrievalResponse:
    """Convenience endpoint for single document retrieval."""
    return await retriever.retrieve_document(query=query)

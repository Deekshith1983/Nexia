"""API routes for Document & Passage Retrieval (Stage 1, 2, 3 & 3.2)."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException, Body
from app.schemas.retrieval import (
    SearchResponse,
    DocumentRetrievalResponse,
    PassageRetrievalRequest,
    PassageRetrievalResponse,
    MultiDocRetrievalRequest,
    MultiDocRetrievalResponse,
)
from app.retrieval.document_retriever import DocumentRetriever
from app.retrieval.passage_retriever import PassageRetriever
from app.retrieval.multi_doc_retriever import MultiDocRetriever

router = APIRouter(tags=["Retrieval"])
document_retriever = DocumentRetriever()
passage_retriever = PassageRetriever(document_retriever=document_retriever)
multi_doc_retriever = MultiDocRetriever(document_retriever=document_retriever)


# --- Stage 1: Search Candidates ---

@router.get("/retrieval/search", response_model=SearchResponse)
@router.get("/retrieve/search", response_model=SearchResponse, include_in_schema=False)
async def search_candidates(
    query: str = Query(..., description="Topic or entity to search on Wikipedia"),
    limit: int = Query(default=10, ge=1, le=20, description="Max candidate results to retrieve"),
) -> SearchResponse:
    """Stage 1: Search Wikipedia and inspect raw candidate articles."""
    return await document_retriever.search_candidates(query=query, limit=limit)


# --- Stage 2: Full Document Retrieval & Structural Parsing ---

@router.get("/retrieval/document", response_model=DocumentRetrievalResponse)
@router.get("/retrieve/document", response_model=DocumentRetrievalResponse, include_in_schema=False)
@router.get("/retrieve", response_model=DocumentRetrievalResponse, include_in_schema=False)
async def retrieve_document(
    query: str = Query(default="", description="Search query or topic"),
    title: Optional[str] = Query(default=None, description="Explicit Wikipedia article title to fetch directly"),
) -> DocumentRetrievalResponse:
    """Stage 2: Retrieve and inspect full structured Wikipedia document with section & paragraph hierarchy."""
    if not query and not title:
        raise HTTPException(status_code=400, detail="Must provide either 'query' or 'title' parameter.")
    
    return await document_retriever.retrieve_document(query=query, explicit_title=title)


# --- Stage 3 & 3.2: Evolved Question-Aware Passage Retrieval & Validation ---

@router.post("/retrieval/passages", response_model=PassageRetrievalResponse)
@router.post("/retrieve/passages", response_model=PassageRetrievalResponse, include_in_schema=False)
async def retrieve_passages_post(
    request: PassageRetrievalRequest = Body(...),
) -> PassageRetrievalResponse:
    """Stage 3.2: Retrieve, score (BM25 / TF-IDF), rank, and validate evidence passages (POST)."""
    return await passage_retriever.retrieve_passages(request)


@router.get("/retrieval/passages", response_model=PassageRetrievalResponse)
@router.get("/retrieve/passages", response_model=PassageRetrievalResponse, include_in_schema=False)
async def retrieve_passages_get(
    question: str = Query(..., description="User question to answer"),
    document: Optional[str] = Query(default=None, description="Target Wikipedia article title"),
    algorithm: str = Query(default="bm25", description="Scoring algorithm: 'bm25' (recommended) or 'tfidf'"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top passages to return"),
    min_score: float = Query(default=0.15, ge=0.0, le=1.0, description="Minimum relevance threshold"),
) -> PassageRetrievalResponse:
    """Stage 3.2: Retrieve, score (BM25 / TF-IDF), rank, and validate evidence passages (GET)."""
    request = PassageRetrievalRequest(
        question=question,
        document=document,
        algorithm=algorithm,
        top_k=top_k,
        min_score=min_score,
    )
    return await passage_retriever.retrieve_passages(request)


# --- Stage 4: Multi-Entity & Multi-Document Retrieval & Relationship Discovery ---

@router.post("/retrieval/multi-document", response_model=MultiDocRetrievalResponse)
@router.post("/retrieve/multi-document", response_model=MultiDocRetrievalResponse, include_in_schema=False)
async def retrieve_multi_document_post(
    request: MultiDocRetrievalRequest = Body(...),
) -> MultiDocRetrievalResponse:
    """Stage 4: Extract entities, fetch multiple documents, search cross-document evidence, and detect relationships (POST)."""
    return await multi_doc_retriever.retrieve_multi_document(request)


@router.get("/retrieval/multi-document", response_model=MultiDocRetrievalResponse)
@router.get("/retrieve/multi-document", response_model=MultiDocRetrievalResponse, include_in_schema=False)
async def retrieve_multi_document_get(
    question: str = Query(..., description="Multi-entity question to analyze"),
    top_k_per_document: int = Query(default=3, ge=1, le=10, description="Max passages to return per document"),
    min_score: float = Query(default=0.15, ge=0.0, le=1.0, description="Minimum relevance threshold"),
) -> MultiDocRetrievalResponse:
    """Stage 4: Extract entities, fetch multiple documents, search cross-document evidence, and detect relationships (GET)."""
    request = MultiDocRetrievalRequest(
        question=question,
        top_k_per_document=top_k_per_document,
        min_score=min_score,
    )
    return await multi_doc_retriever.retrieve_multi_document(request)


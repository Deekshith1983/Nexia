"""Hybrid vector and full-text keyword retrieval with Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict
import logging
from typing import Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.connection import get_session_factory
from app.db.models import NexiaChunk, NexiaDocument
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RetrievedChunk(BaseModel):
    """Unified candidate chunk scored and ranked by hybrid retrieval."""

    chunk_id: int
    document_title: str
    source_url: str
    section_title: str
    passage_text: str
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    rrf_score: float = 0.0
    retrieval_method: str = "hybrid"  # "vector" | "keyword" | "hybrid"


class HybridRetrievalService:
    """Combines pgvector dense semantic search and PostgreSQL full-text search."""

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        self.embedding_service = embedding_service or EmbeddingService()

    async def retrieve(
        self,
        question: str,
        page_ids: Optional[List[int]] = None,
        top_k: Optional[int] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[RetrievedChunk]:
        """Perform vector + keyword retrieval and fuse results using Reciprocal Rank Fusion."""
        k = top_k or settings.HYBRID_TOP_K

        if session is not None:
            return await self._execute_retrieval(session, question, page_ids, k)

        session_factory = get_session_factory()
        async with session_factory() as sess:
            return await self._execute_retrieval(sess, question, page_ids, k)

    async def _execute_retrieval(
        self,
        session: AsyncSession,
        question: str,
        page_ids: Optional[List[int]],
        top_k: int,
    ) -> List[RetrievedChunk]:
        # 1. Compute query vector
        query_vector = await self.embedding_service.embed_text(question)

        # 2. Dense Vector Search (Cosine Distance)
        # Using pgvector <=> operator via SQLAlchemy
        vector_query = (
            select(
                NexiaChunk.id,
                NexiaChunk.section_title,
                NexiaChunk.chunk_text,
                NexiaChunk.source_url,
                NexiaDocument.title.label("document_title"),
                NexiaChunk.embedding.cosine_distance(query_vector).label("distance"),
            )
            .join(NexiaDocument, NexiaChunk.document_id == NexiaDocument.id)
            .where(NexiaChunk.embedding.is_not(None))
        )
        if page_ids:
            vector_query = vector_query.where(NexiaChunk.page_id.in_(page_ids))

        vector_query = vector_query.order_by("distance").limit(
            settings.VECTOR_SEARCH_CANDIDATES
        )

        vector_results = (await session.execute(vector_query)).fetchall()

        # 3. Full-Text Keyword Search
        keyword_query = (
            select(
                NexiaChunk.id,
                NexiaChunk.section_title,
                NexiaChunk.chunk_text,
                NexiaChunk.source_url,
                NexiaDocument.title.label("document_title"),
                func.ts_rank_cd(
                    func.to_tsvector("english", NexiaChunk.chunk_text),
                    func.plainto_tsquery("english", question),
                ).label("rank_score"),
            )
            .join(NexiaDocument, NexiaChunk.document_id == NexiaDocument.id)
            .where(
                func.to_tsvector("english", NexiaChunk.chunk_text).op("@@")(
                    func.plainto_tsquery("english", question)
                )
            )
        )
        if page_ids:
            keyword_query = keyword_query.where(NexiaChunk.page_id.in_(page_ids))

        keyword_query = keyword_query.order_by(text("rank_score DESC")).limit(
            settings.KEYWORD_SEARCH_CANDIDATES
        )

        keyword_results = (await session.execute(keyword_query)).fetchall()

        # 4. Reciprocal Rank Fusion (RRF)
        # RRF formula: sum(1 / (k_const + rank))
        k_const = 60
        fused_items: Dict[int, Dict] = {}
        rrf_scores: Dict[int, float] = defaultdict(float)

        # Process vector ranks (1-indexed)
        for rank_idx, row in enumerate(vector_results, start=1):
            chunk_id = row.id
            rrf_scores[chunk_id] += 1.0 / (k_const + rank_idx)
            if chunk_id not in fused_items:
                fused_items[chunk_id] = {
                    "id": chunk_id,
                    "section_title": row.section_title,
                    "chunk_text": row.chunk_text,
                    "source_url": row.source_url,
                    "document_title": row.document_title,
                    "vector_rank": rank_idx,
                    "keyword_rank": None,
                }
            else:
                fused_items[chunk_id]["vector_rank"] = rank_idx

        # Process keyword ranks (1-indexed)
        for rank_idx, row in enumerate(keyword_results, start=1):
            chunk_id = row.id
            rrf_scores[chunk_id] += 1.0 / (k_const + rank_idx)
            if chunk_id not in fused_items:
                fused_items[chunk_id] = {
                    "id": chunk_id,
                    "section_title": row.section_title,
                    "chunk_text": row.chunk_text,
                    "source_url": row.source_url,
                    "document_title": row.document_title,
                    "vector_rank": None,
                    "keyword_rank": rank_idx,
                }
            else:
                fused_items[chunk_id]["keyword_rank"] = rank_idx

        # Sort by final RRF score descending
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )

        final_chunks: List[RetrievedChunk] = []
        for cid in sorted_chunk_ids[:top_k]:
            data = fused_items[cid]
            v_rank = data["vector_rank"]
            k_rank = data["keyword_rank"]

            if v_rank is not None and k_rank is not None:
                method = "hybrid"
            elif v_rank is not None:
                method = "vector"
            else:
                method = "keyword"

            final_chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    document_title=data["document_title"],
                    source_url=data["source_url"],
                    section_title=data["section_title"],
                    passage_text=data["chunk_text"],
                    vector_rank=v_rank,
                    keyword_rank=k_rank,
                    rrf_score=round(rrf_scores[cid], 5),
                    retrieval_method=method,
                )
            )

        return final_chunks

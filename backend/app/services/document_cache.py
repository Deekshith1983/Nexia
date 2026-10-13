"""Document cache service providing persistent PostgreSQL storage and idempotent ingestion."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from typing import Optional, Tuple
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.connection import get_session_factory
from app.db.models import NexiaChunk, NexiaDocument
from app.retrieval.document_retriever import DocumentRetriever
from app.schemas.retrieval import Document
from app.services.chunker import ChunkerService
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class DocumentCacheService:
    """Manages Wikipedia document caching, freshness checks, and vector ingestion in PostgreSQL."""

    def __init__(
        self,
        document_retriever: Optional[DocumentRetriever] = None,
        chunker: Optional[ChunkerService] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.doc_retriever = document_retriever or DocumentRetriever()
        self.chunker = chunker or ChunkerService()
        self.embedding_service = embedding_service or EmbeddingService()

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize article title for consistent lookup."""
        return title.strip().lower()

    @staticmethod
    def compute_content_hash(doc: Document) -> str:
        """Compute stable hash of document contents."""
        serialized = json.dumps(doc.model_dump(), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def get_or_fetch_document(
        self,
        title: str,
        page_id: Optional[int] = None,
        force_refresh: bool = False,
        session: Optional[AsyncSession] = None,
    ) -> Tuple[Document, str]:
        """Lookup document in cache. If missing, stale, or forced, fetch and ingest.

        Returns:
            Tuple[Document, str]: (Document, status) where status is "HIT", "MISS", or "REFRESHED".
        """
        normalized = self.normalize_title(title)

        if session is not None:
            return await self._get_or_fetch_with_session(
                session, normalized, title, page_id, force_refresh
            )

        session_factory = get_session_factory()
        async with session_factory() as sess:
            return await self._get_or_fetch_with_session(
                sess, normalized, title, page_id, force_refresh
            )

    async def _get_or_fetch_with_session(
        self,
        session: AsyncSession,
        normalized_title: str,
        raw_title: str,
        page_id: Optional[int],
        force_refresh: bool,
    ) -> Tuple[Document, str]:
        # 1. Check existing record
        query = select(NexiaDocument).where(
            (NexiaDocument.normalized_title == normalized_title)
            | (NexiaDocument.page_id == page_id if page_id else False)
        )
        result = await session.execute(query)
        db_doc = result.scalars().first()

        now = datetime.now(timezone.utc)
        freshness_limit = now - timedelta(days=settings.CACHE_FRESHNESS_DAYS)

        # 2. Check if fresh hit
        if db_doc and not force_refresh:
            fetched_at = db_doc.fetched_at
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.replace(tzinfo=timezone.utc)

            if fetched_at >= freshness_limit:
                # Fresh cache hit
                try:
                    cached_model = Document.model_validate(db_doc.content_json)
                    return cached_model, "HIT"
                except Exception as e:
                    logger.warning("Failed to deserialize cached document JSON: %s. Re-fetching.", e)

        # 3. Cache Miss or Refresh required -> Fetch from Wikipedia
        fetch_res = await self.doc_retriever.retrieve_document(
            query=raw_title, explicit_title=raw_title
        )
        fresh_doc = fetch_res.document
        if not fresh_doc:
            fetch_res = await self.doc_retriever.retrieve_document(query=raw_title)
            fresh_doc = fetch_res.document

        if not fresh_doc:
            raise ValueError(f"Wikipedia document not found for '{raw_title}'.")

        # Check if canonical page_id is already in DB under its canonical title
        if not db_doc and fresh_doc.page_id:
            existing_by_page = (
                await session.execute(
                    select(NexiaDocument).where(NexiaDocument.page_id == fresh_doc.page_id)
                )
            ).scalars().first()
            if existing_by_page and not force_refresh:
                try:
                    cached_model = Document.model_validate(existing_by_page.content_json)
                    return cached_model, "HIT"
                except Exception:
                    pass

        new_hash = self.compute_content_hash(fresh_doc)

        # If existing record has unchanged hash, update timestamp only
        if db_doc and db_doc.content_hash == new_hash and not force_refresh:
            db_doc.last_checked_at = now
            await session.commit()
            return fresh_doc, "HIT"

        # 4. Prepare chunks & embeddings
        chunks = self.chunker.chunk_document(fresh_doc)
        chunk_texts = [c.chunk_text for c in chunks]
        embeddings = await self.embedding_service.embed_batch(chunk_texts)

        # 5. Persist transactionally
        status = "REFRESHED" if db_doc else "MISS"

        if db_doc:
            # Delete old chunks
            await session.execute(
                delete(NexiaChunk).where(NexiaChunk.document_id == db_doc.id)
            )
            # Update document record
            db_doc.page_id = fresh_doc.page_id
            db_doc.title = fresh_doc.title
            db_doc.normalized_title = normalized_title
            db_doc.url = fresh_doc.url
            db_doc.introduction = fresh_doc.introduction
            db_doc.content_json = fresh_doc.model_dump()
            db_doc.content_hash = new_hash
            db_doc.total_sections = fresh_doc.metadata.total_sections
            db_doc.total_words = fresh_doc.metadata.total_words
            db_doc.fetched_at = now
            db_doc.last_checked_at = now
            doc_id = db_doc.id
        else:
            new_db_doc = NexiaDocument(
                page_id=fresh_doc.page_id,
                source="wikipedia_en",
                title=fresh_doc.title,
                normalized_title=normalized_title,
                url=fresh_doc.url,
                introduction=fresh_doc.introduction,
                content_json=fresh_doc.model_dump(),
                content_hash=new_hash,
                total_sections=fresh_doc.metadata.total_sections,
                total_words=fresh_doc.metadata.total_words,
                status="indexed",
                schema_version=1,
                fetched_at=now,
                last_checked_at=now,
            )
            session.add(new_db_doc)
            await session.flush()
            doc_id = new_db_doc.id

        # Insert chunks
        chunk_models = []
        for chunk_payload, emb in zip(chunks, embeddings):
            chunk_models.append(
                NexiaChunk(
                    document_id=doc_id,
                    page_id=fresh_doc.page_id,
                    chunk_index=chunk_payload.chunk_index,
                    section_title=chunk_payload.section_title,
                    section_level=chunk_payload.section_level,
                    parent_section=chunk_payload.parent_section,
                    paragraph_id=chunk_payload.paragraph_id,
                    chunk_text=chunk_payload.chunk_text,
                    source_url=chunk_payload.source_url,
                    content_hash=chunk_payload.content_hash,
                    embedding=emb,
                    embedding_model=self.embedding_service.model,
                    embedding_at=now,
                )
            )
        session.add_all(chunk_models)
        await session.commit()

        return fresh_doc, status

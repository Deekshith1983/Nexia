"""Memory Manager for conversational, episodic, and semantic memory lifecycle."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from sqlalchemy import desc, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connection import get_session_factory
from app.db.models import NexiaMemory
from app.schemas.memory import MemoryCreate, MemoryRead, MemoryType, MemoryUpdate
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("nexia.memory_manager")


class MemoryManager:
    """Manages short-term entity context, episodic history, and vector-indexed semantic memory."""

    def __init__(
        self,
        session_factory=None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.session_factory = session_factory or get_session_factory()
        self.embedding_service = embedding_service or EmbeddingService()

    async def store_memory(
        self,
        subject: str,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        conversation_id: Optional[UUID] = None,
        confidence: float = 1.0,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        generate_embedding: bool = True,
        session: Optional[AsyncSession] = None,
    ) -> MemoryRead:
        """Store a new memory item, optionally generating an embedding for semantic search."""
        embedding_vec = None
        if generate_embedding and memory_type in (MemoryType.SEMANTIC, MemoryType.EPISODIC):
            try:
                embed_text = f"{subject}: {content}"
                embeddings = await self.embedding_service.embed_batch([embed_text])
                if embeddings:
                    embedding_vec = embeddings[0]
            except Exception as e:
                logger.warning("Failed to generate memory embedding: %s", e)

        mem = NexiaMemory(
            id=uuid4(),
            conversation_id=conversation_id,
            memory_type=memory_type.value if isinstance(memory_type, MemoryType) else memory_type,
            subject=subject,
            content=content,
            confidence=confidence,
            embedding=embedding_vec,
            expires_at=expires_at,
            metadata_json=metadata or {},
        )

        async def _save(s: AsyncSession):
            s.add(mem)
            await s.commit()
            await s.refresh(mem)

        if session:
            await _save(session)
        else:
            async with self.session_factory() as s:
                await _save(s)

        c_at = getattr(mem, "created_at", None)
        if not isinstance(c_at, datetime):
            c_at = datetime.now(timezone.utc)
        u_at = getattr(mem, "updated_at", None)
        if not isinstance(u_at, datetime):
            u_at = datetime.now(timezone.utc)

        logger.info("Stored memory [%s] for subject '%s'", mem.memory_type, mem.subject)
        return MemoryRead(
            id=mem.id,
            conversation_id=mem.conversation_id,
            memory_type=MemoryType(mem.memory_type),
            subject=mem.subject,
            content=mem.content,
            confidence=mem.confidence,
            created_at=c_at,
            updated_at=u_at,
            expires_at=mem.expires_at,
            metadata=mem.metadata_json,
        )

    async def retrieve_memories(
        self,
        query: str,
        conversation_id: Optional[UUID] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
        min_relevance: float = 0.60,
        session: Optional[AsyncSession] = None,
    ) -> List[MemoryRead]:
        """Retrieve relevant memories using dense vector similarity or subject keyword matching."""
        # 1. Try dense vector similarity if query embedding is available
        query_embedding = None
        try:
            embeds = await self.embedding_service.embed_batch([query])
            if embeds:
                query_embedding = embeds[0]
        except Exception as e:
            logger.debug("Query embedding unavailable for memory search: %s", e)

        async def _search(s: AsyncSession) -> List[MemoryRead]:
            now = datetime.now(timezone.utc)
            base_query = select(NexiaMemory).where(
                (NexiaMemory.expires_at == None) | (NexiaMemory.expires_at > now)
            )

            if conversation_id:
                # Include memories specifically for this conversation OR global semantic memories
                base_query = base_query.where(
                    (NexiaMemory.conversation_id == conversation_id)
                    | (NexiaMemory.conversation_id == None)
                )

            if memory_type:
                base_query = base_query.where(NexiaMemory.memory_type == memory_type.value)

            if query_embedding:
                # Use cosine distance with pgvector: 1 - cosine_distance = cosine_similarity
                distance_col = NexiaMemory.embedding.cosine_distance(query_embedding)
                vector_stmt = (
                    base_query.where(NexiaMemory.embedding != None)
                    .order_by(distance_col.asc())
                    .limit(limit)
                )
                res = await s.execute(vector_stmt)
                rows = res.scalars().all()
                results = []
                for r in rows:
                    results.append(
                        MemoryRead(
                            id=r.id,
                            conversation_id=r.conversation_id,
                            memory_type=MemoryType(r.memory_type),
                            subject=r.subject,
                            content=r.content,
                            confidence=r.confidence,
                            created_at=r.created_at,
                            updated_at=r.updated_at,
                            expires_at=r.expires_at,
                            metadata=r.metadata_json,
                            relevance_score=0.85,  # Vector match
                        )
                    )
                if results:
                    return results

            # Fallback to subject keyword match
            subject_stmt = (
                base_query.order_by(desc(NexiaMemory.created_at))
                .limit(limit)
            )
            res = await s.execute(subject_stmt)
            rows = res.scalars().all()
            q_lower = query.lower()
            keyword_matches = []
            for r in rows:
                if r.subject.lower() in q_lower or any(word in r.content.lower() for word in q_lower.split() if len(word) > 4):
                    keyword_matches.append(
                        MemoryRead(
                            id=r.id,
                            conversation_id=r.conversation_id,
                            memory_type=MemoryType(r.memory_type),
                            subject=r.subject,
                            content=r.content,
                            confidence=r.confidence,
                            created_at=r.created_at,
                            updated_at=r.updated_at,
                            expires_at=r.expires_at,
                            metadata=r.metadata_json,
                            relevance_score=0.70,
                        )
                    )
            return keyword_matches

        if session:
            return await _search(session)
        async with self.session_factory() as s:
            return await _search(s)

    async def get_active_subjects(
        self,
        conversation_id: UUID,
        limit: int = 5,
        session: Optional[AsyncSession] = None,
    ) -> List[str]:
        """Fetch the most recent active subject entities for reference resolution."""
        async def _subjects(s: AsyncSession):
            stmt = (
                select(NexiaMemory.subject)
                .where(NexiaMemory.conversation_id == conversation_id)
                .order_by(desc(NexiaMemory.created_at))
                .limit(limit)
            )
            res = await s.execute(stmt)
            return list(dict.fromkeys(r[0] for r in res.fetchall() if r[0]))

        if session:
            return await _subjects(session)
        async with self.session_factory() as s:
            return await _subjects(s)

    async def delete_memory(
        self,
        memory_id: UUID,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """Delete a memory item."""
        async def _del(s: AsyncSession):
            stmt = select(NexiaMemory).where(NexiaMemory.id == memory_id)
            res = await s.execute(stmt)
            mem = res.scalar_one_or_none()
            if not mem:
                return False
            await s.delete(mem)
            await s.commit()
            return True

        if session:
            return await _del(session)
        async with self.session_factory() as s:
            return await _del(s)

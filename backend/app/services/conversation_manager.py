"""Conversation Manager for multi-turn conversational session and message lifecycle."""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.connection import get_session_factory
from app.db.models import NexiaConversation, NexiaConversationMessage
from app.services.cache_service import CacheService
from app.schemas.conversation import (
    ChatMessage,
    ConversationCreate,
    ConversationRead,
    ConversationStatus,
    ConversationWithHistory,
    MessageRole,
    MessageStatus,
)

logger = logging.getLogger("nexia.conversation_manager")


class ConversationManager:
    """Manages conversational session persistence, message histories, and reference retrieval."""

    def __init__(self, session_factory=None):
        self.session_factory = session_factory or get_session_factory()

    async def create_conversation(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> ConversationRead:
        """Create a new conversation session."""
        conv_title = title or "New Chat"
        conv = NexiaConversation(
            id=uuid4(),
            title=conv_title,
            status=ConversationStatus.ACTIVE.value,
            metadata_json=metadata or {},
        )

        async def _persist(s: AsyncSession):
            s.add(conv)
            await s.commit()
            await s.refresh(conv)

        if session:
            await _persist(session)
        else:
            async with self.session_factory() as s:
                await _persist(s)

        c_at = getattr(conv, "created_at", None)
        if not isinstance(c_at, datetime):
            c_at = datetime.now(timezone.utc)
        u_at = getattr(conv, "updated_at", None)
        if not isinstance(u_at, datetime):
            u_at = datetime.now(timezone.utc)

        logger.info("Created conversation %s: '%s'", conv.id, conv.title)
        await CacheService.get_instance().invalidate_prefix("conv_list:")
        return ConversationRead(
            id=conv.id,
            title=conv.title,
            status=ConversationStatus(conv.status),
            metadata=conv.metadata_json,
            created_at=c_at,
            updated_at=u_at,
            message_count=0,
        )

    async def get_conversation(
        self,
        conversation_id: UUID,
        session: Optional[AsyncSession] = None,
    ) -> Optional[ConversationRead]:
        """Fetch conversation by ID."""
        async def _query(s: AsyncSession):
            stmt = select(NexiaConversation).where(NexiaConversation.id == conversation_id)
            res = await s.execute(stmt)
            conv = res.scalar_one_or_none()
            if not conv:
                return None
            count_stmt = select(func.count()).select_from(NexiaConversationMessage).where(
                NexiaConversationMessage.conversation_id == conversation_id
            )
            count_res = await s.execute(count_stmt)
            msg_count = count_res.scalar() or 0
            return ConversationRead(
                id=conv.id,
                title=conv.title,
                status=ConversationStatus(conv.status),
                metadata=conv.metadata_json,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
                message_count=msg_count,
            )

        if session:
            return await _query(session)
        async with self.session_factory() as s:
            return await _query(s)

    async def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[ConversationStatus] = None,
        session: Optional[AsyncSession] = None,
    ) -> List[ConversationRead]:
        """List conversations ordered by updated_at desc with batch count query and caching."""
        cache = CacheService.get_instance()
        cache_key = f"conv_list:{limit}:{offset}:{status.value if status else 'all'}"
        cached = await cache.get(cache_key)
        if cached is not None and isinstance(cached, list):
            return [ConversationRead.model_validate(c) for c in cached]

        async def _query(s: AsyncSession):
            stmt = select(NexiaConversation)
            if status:
                stmt = stmt.where(NexiaConversation.status == status.value)
            stmt = stmt.order_by(desc(NexiaConversation.updated_at)).limit(limit).offset(offset)
            res = await s.execute(stmt)
            conversations = res.scalars().all()
            if not conversations:
                return []

            conv_ids = [c.id for c in conversations]
            counts_stmt = (
                select(
                    NexiaConversationMessage.conversation_id,
                    func.count().label("cnt"),
                )
                .where(NexiaConversationMessage.conversation_id.in_(conv_ids))
                .group_by(NexiaConversationMessage.conversation_id)
            )
            counts_res = await s.execute(counts_stmt)
            counts_map = {row[0]: row[1] for row in counts_res.all()}

            results = []
            for c in conversations:
                results.append(
                    ConversationRead(
                        id=c.id,
                        title=c.title,
                        status=ConversationStatus(c.status),
                        metadata=c.metadata_json,
                        created_at=c.created_at,
                        updated_at=c.updated_at,
                        message_count=counts_map.get(c.id, 0),
                    )
                )
            return results

        if session:
            res_list = await _query(session)
        else:
            async with self.session_factory() as s:
                res_list = await _query(s)

        await cache.set(
            cache_key,
            [r.model_dump() for r in res_list],
            ttl_seconds=float(settings.CACHE_CONVERSATIONS_TTL_SECONDS),
        )
        return res_list

    async def update_conversation(
        self,
        conversation_id: UUID,
        title: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        metadata: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> Optional[ConversationRead]:
        """Update conversation title, status, or metadata."""
        async def _update(s: AsyncSession):
            stmt = select(NexiaConversation).where(NexiaConversation.id == conversation_id)
            res = await s.execute(stmt)
            conv = res.scalar_one_or_none()
            if not conv:
                return None
            if title is not None:
                conv.title = title
            if status is not None:
                conv.status = status.value
            if metadata is not None:
                conv.metadata_json = metadata
            conv.updated_at = datetime.now(timezone.utc)
            s.add(conv)
            await s.commit()
            await s.refresh(conv)
            return await self.get_conversation(conversation_id, session=s)

        if session:
            res = await _update(session)
        else:
            async with self.session_factory() as s:
                res = await _update(s)
        if res:
            await CacheService.get_instance().invalidate_prefix("conv_list:")
        return res

    async def delete_conversation(
        self,
        conversation_id: UUID,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """Permanently delete a conversation and its messages."""
        async def _del(s: AsyncSession):
            stmt = select(NexiaConversation).where(NexiaConversation.id == conversation_id)
            res = await s.execute(stmt)
            conv = res.scalar_one_or_none()
            if not conv:
                return False
            await s.delete(conv)
            await s.commit()
            return True

        if session:
            ok = await _del(session)
        else:
            async with self.session_factory() as s:
                ok = await _del(s)
        if ok:
            await CacheService.get_instance().invalidate_prefix("conv_list:")
        return ok

    async def add_message(
        self,
        conversation_id: UUID,
        role: MessageRole,
        content: str,
        token_count: Optional[int] = None,
        message_status: MessageStatus = MessageStatus.COMPLETED,
        source_references: Optional[List[Dict[str, Any]]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
    ) -> ChatMessage:
        """Append a message turn to the conversation history."""
        msg = NexiaConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            role=role.value if isinstance(role, MessageRole) else role,
            content=content,
            token_count=token_count,
            message_status=message_status.value if isinstance(message_status, MessageStatus) else message_status,
            source_references=source_references or [],
            diagnostics=diagnostics or {},
        )

        async def _add(s: AsyncSession):
            now = datetime.now(timezone.utc)
            # Ensure conversation exists
            conv_stmt = select(NexiaConversation).where(NexiaConversation.id == conversation_id)
            conv_res = await s.execute(conv_stmt)
            conv = conv_res.scalar_one_or_none()
            if not conv:
                # Auto-create if not present with default temporary title
                conv = NexiaConversation(
                    id=conversation_id,
                    title="New Chat",
                    status=ConversationStatus.ACTIVE.value,
                    metadata_json={},
                    created_at=now,
                    updated_at=now,
                )
                s.add(conv)
            else:
                conv.updated_at = now
                s.add(conv)

            s.add(msg)
            await s.commit()
            await s.refresh(msg)

        if session:
            await _add(session)
        else:
            async with self.session_factory() as s:
                await _add(s)

        await CacheService.get_instance().invalidate_prefix("conv_list:")

        m_at = getattr(msg, "created_at", None)
        if not isinstance(m_at, datetime):
            m_at = datetime.now(timezone.utc)

        return ChatMessage(
            id=msg.id,
            conversation_id=msg.conversation_id,
            role=MessageRole(msg.role),
            content=msg.content,
            token_count=msg.token_count,
            message_status=MessageStatus(msg.message_status),
            source_references=msg.source_references or [],
            diagnostics=msg.diagnostics or {},
            created_at=m_at,
        )

    async def get_messages(
        self,
        conversation_id: UUID,
        limit: int = 50,
        session: Optional[AsyncSession] = None,
    ) -> List[ChatMessage]:
        """Fetch conversation messages ordered chronologically."""
        async def _query(s: AsyncSession):
            stmt = (
                select(NexiaConversationMessage)
                .where(NexiaConversationMessage.conversation_id == conversation_id)
                .order_by(NexiaConversationMessage.created_at.asc())
                .limit(limit)
            )
            res = await s.execute(stmt)
            msgs = res.scalars().all()
            return [
                ChatMessage(
                    id=m.id,
                    conversation_id=m.conversation_id,
                    role=MessageRole(m.role),
                    content=m.content,
                    token_count=m.token_count,
                    message_status=MessageStatus(m.message_status),
                    source_references=m.source_references or [],
                    diagnostics=m.diagnostics or {},
                    created_at=m.created_at,
                )
                for m in msgs
            ]

        if session:
            return await _query(session)
        async with self.session_factory() as s:
            return await _query(s)

    async def get_conversation_with_history(
        self,
        conversation_id: UUID,
        limit: int = 50,
        session: Optional[AsyncSession] = None,
    ) -> Optional[ConversationWithHistory]:
        """Fetch conversation with its chronological messages."""
        conv = await self.get_conversation(conversation_id, session=session)
        if not conv:
            return None
        messages = await self.get_messages(conversation_id, limit=limit, session=session)
        return ConversationWithHistory(
            id=conv.id,
            title=conv.title,
            status=conv.status,
            metadata=conv.metadata,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=len(messages),
            messages=messages,
        )

    async def is_eligible_for_title_generation(
        self,
        conversation_id: UUID,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """Check if conversation is eligible for automatic title generation.

        Eligible iff:
        1. Conversation exists
        2. Title is currently 'New Chat' or 'New Conversation' (case-insensitive check)
        3. Message count is exactly 2 (first user turn + first assistant answer)
        """
        async def _check(s: AsyncSession) -> bool:
            stmt = select(NexiaConversation).where(NexiaConversation.id == conversation_id)
            res = await s.execute(stmt)
            conv = res.scalar_one_or_none()
            if not conv:
                return False

            normalized_title = conv.title.strip().lower()
            if normalized_title not in ("new chat", "new conversation"):
                return False

            count_stmt = select(func.count()).select_from(NexiaConversationMessage).where(
                NexiaConversationMessage.conversation_id == conversation_id
            )
            count_res = await s.execute(count_stmt)
            msg_count = count_res.scalar() or 0
            return msg_count == 2

        if session:
            return await _check(session)
        async with self.session_factory() as s:
            return await _check(s)

    async def update_title_atomic(
        self,
        conversation_id: UUID,
        new_title: str,
        session: Optional[AsyncSession] = None,
    ) -> bool:
        """Atomically update title only if it is still 'New Chat' or 'New Conversation'.

        Returns True if row was updated, False if conversation was missing or already renamed.
        """
        async def _update(s: AsyncSession) -> bool:
            stmt = (
                update(NexiaConversation)
                .where(
                    NexiaConversation.id == conversation_id,
                    NexiaConversation.title.in_(["New Chat", "New Conversation"]),
                )
                .values(
                    title=new_title,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            res = await s.execute(stmt)
            await s.commit()
            return res.rowcount > 0

        if session:
            ok = await _update(session)
        else:
            async with self.session_factory() as s:
                ok = await _update(s)
        if ok:
            await CacheService.get_instance().invalidate_prefix("conv_list:")
        return ok

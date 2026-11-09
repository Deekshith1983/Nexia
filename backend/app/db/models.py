"""SQLAlchemy ORM models for Nexia document and chunk persistence."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy declarative models."""
    pass


class NexiaDocument(Base):
    """Stores full ingested articles and metadata."""

    __tablename__ = "nexia_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="wikipedia_en")
    title: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_title: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    introduction: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_json: Mapped[Any] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    total_sections: Mapped[int] = mapped_column(Integer, default=0)
    total_words: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="indexed")
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    chunks: Mapped[List["NexiaChunk"]] = relationship(
        "NexiaChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="NexiaChunk.chunk_index",
    )

    __table_args__ = (
        UniqueConstraint("source", "page_id", name="uq_doc_source_pageid"),
        Index("idx_documents_normalized_title", "normalized_title"),
        Index("idx_documents_fetched_at", "fetched_at"),
    )


class NexiaChunk(Base):
    """Stores paragraph-level chunks with dense vector embeddings."""

    __tablename__ = "nexia_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("nexia_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_title: Mapped[str] = mapped_column(Text, nullable=False)
    section_level: Mapped[int] = mapped_column(Integer, default=2)
    parent_section: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    paragraph_id: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    
    # 768-dimensional embedding from nomic-embed-text
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    document: Mapped["NexiaDocument"] = relationship("NexiaDocument", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "paragraph_id", name="uq_chunk_doc_paragraph"),
        Index("idx_chunks_document_id", "document_id"),
        Index("idx_chunks_page_id", "page_id"),
    )


class NexiaConversation(Base):
    """Stores multi-turn conversational sessions."""

    __tablename__ = "nexia_conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="New Chat")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    messages: Mapped[List["NexiaConversationMessage"]] = relationship(
        "NexiaConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="NexiaConversationMessage.created_at",
    )
    memories: Mapped[List["NexiaMemory"]] = relationship(
        "NexiaMemory",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_conversations_created_at", "created_at"),
        Index("idx_conversations_updated_at", "updated_at"),
        Index("idx_conversations_status", "status"),
        Index("idx_conversations_status_updated_at", "status", "updated_at"),
    )


class NexiaConversationMessage(Base):
    """Stores individual turns/messages in a conversation."""

    __tablename__ = "nexia_conversation_messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("nexia_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # "user" | "assistant" | "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message_status: Mapped[str] = mapped_column(String(32), nullable=False, default="COMPLETED")
    source_references: Mapped[Any] = mapped_column(JSONB, nullable=False, default=list)
    diagnostics: Mapped[Any] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    # Relationships
    conversation: Mapped["NexiaConversation"] = relationship(
        "NexiaConversation",
        back_populates="messages",
    )

    __table_args__ = (
        Index("idx_conv_messages_conv_id", "conversation_id"),
        Index("idx_conv_messages_created_at", "created_at"),
        Index("idx_conv_messages_conv_created", "conversation_id", "created_at"),
    )


class NexiaMemory(Base):
    """Stores short-term, episodic, and semantic memories."""

    __tablename__ = "nexia_memories"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("nexia_conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    memory_type: Mapped[str] = mapped_column(String(32), nullable=False)  # "SHORT_TERM", "EPISODIC", "SEMANTIC"
    subject: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    
    # 768-dim nomic-embed-text for semantic memory similarity
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSIONS),
        nullable=True,
    )
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    conversation: Mapped[Optional["NexiaConversation"]] = relationship(
        "NexiaConversation",
        back_populates="memories",
    )

    __table_args__ = (
        Index("idx_memories_subject", "subject"),
        Index("idx_memories_type", "memory_type"),
        Index("idx_memories_created_at", "created_at"),
    )

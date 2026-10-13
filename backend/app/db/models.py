"""SQLAlchemy ORM models for Nexia document and chunk persistence."""

from datetime import datetime
from typing import Any, List, Optional
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
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

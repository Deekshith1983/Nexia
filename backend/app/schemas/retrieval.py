"""Schemas for Stage 1 Document Retrieval."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchCandidate(BaseModel):
    """A search result candidate from Wikipedia."""
    title: str
    page_id: int
    snippet: str = ""
    url: str
    word_count: Optional[int] = None


class SearchResponse(BaseModel):
    """Response schema for Wikipedia search candidates."""
    query: str
    total_hits: int
    candidates: List[SearchCandidate]


class Paragraph(BaseModel):
    """Individually addressable paragraph within an article section."""
    id: str = Field(..., description="Unique addressable ID: section_idx_p_idx")
    section_title: str
    index: int = Field(..., description="Paragraph index within the section")
    text: str
    word_count: int


class Section(BaseModel):
    """Structured section of an article."""
    index: int
    title: str
    level: int = Field(default=2, description="Heading level (2 for ==, 3 for ===, etc.)")
    paragraphs: List[Paragraph] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    """Metadata describing the retrieved article."""
    page_id: int
    url: str
    language: str = "en"
    total_sections: int
    total_paragraphs: int
    total_words: int
    categories: List[str] = Field(default_factory=list)


class Document(BaseModel):
    """Structured internal representation of a full Wikipedia article."""
    title: str
    page_id: int
    url: str
    introduction: str = Field(..., description="Lead section before the first heading")
    sections: List[Section] = Field(default_factory=list)
    metadata: DocumentMetadata


class DocumentRetrievalResponse(BaseModel):
    """Response schema for full document retrieval inspection."""
    query: str
    status: str = Field(..., description="'success', 'not_found', or 'error'")
    selected_candidate: Optional[SearchCandidate] = None
    document: Optional[Document] = None
    other_candidates: List[SearchCandidate] = Field(default_factory=list)
    error_message: Optional[str] = None

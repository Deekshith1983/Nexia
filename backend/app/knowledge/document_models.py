"""
Wikipedia Document, Section, and Passage Models for Retrieval Engine V2.

Provides rich hierarchical representations of Wikipedia articles:
- WikipediaDocument (root article with metadata, lead, sections, links)
- WikipediaDocumentSection (individual sections with level, heading, text, and sub-passages)
- WikipediaPassage (bounded chunk of 150-800 characters/tokens with passage_id and provenance)
"""
from datetime import datetime
import hashlib
import re
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def _hash_id(text: str) -> str:
    """Generate deterministic 16-character hex hash."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class WikipediaPassage(BaseModel):
    """
    Fine-grained passage/chunk within a Wikipedia section.
    Designed for bounded context injection into Gemma (300-800 tokens).
    """
    passage_id: str
    canonical_title: str
    section_title: str
    section_index: int = 0
    passage_index: int = 0
    text: str
    token_count_estimate: int = 0
    score: float = 0.0

    @classmethod
    def create(
        cls,
        canonical_title: str,
        section_title: str,
        section_index: int,
        passage_index: int,
        text: str,
    ) -> "WikipediaPassage":
        clean_text = text.strip()
        pid = _hash_id(f"{canonical_title}::{section_title}::{passage_index}::{clean_text[:64]}")
        # Rough token estimate: ~4 chars per token in English
        tokens = max(1, len(clean_text) // 4)
        return cls(
            passage_id=pid,
            canonical_title=canonical_title,
            section_title=section_title,
            section_index=section_index,
            passage_index=passage_index,
            text=clean_text,
            token_count_estimate=tokens,
        )


class WikipediaDocumentSection(BaseModel):
    """
    A single section or subsection in a WikipediaDocument.
    """
    section_id: str
    title: str
    level: int = 2
    index: int = 0
    parent_section_title: Optional[str] = None
    content: str = ""
    passages: List[WikipediaPassage] = Field(default_factory=list)

    @classmethod
    def create(
        cls,
        canonical_title: str,
        title: str,
        level: int,
        index: int,
        content: str,
        parent_section_title: Optional[str] = None,
    ) -> "WikipediaDocumentSection":
        sec_id = _hash_id(f"{canonical_title}::{index}::{title}")
        clean_content = content.strip()
        
        # Chunk section content into passages
        passages: List[WikipediaPassage] = []
        if clean_content:
            # Paragraph split
            paras = [p.strip() for p in re.split(r"\n\s*\n+", clean_content) if p.strip()]
            for p_idx, para in enumerate(paras):
                # If paragraph is very long (> 1500 chars), split into sentence chunks
                if len(para) > 1500:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    chunk = []
                    chunk_len = 0
                    c_idx = p_idx
                    for s in sentences:
                        if chunk_len + len(s) > 1200 and chunk:
                            passages.append(
                                WikipediaPassage.create(
                                    canonical_title=canonical_title,
                                    section_title=title,
                                    section_index=index,
                                    passage_index=c_idx,
                                    text=" ".join(chunk),
                                )
                            )
                            c_idx += 1
                            chunk = [s]
                            chunk_len = len(s)
                        else:
                            chunk.append(s)
                            chunk_len += len(s) + 1
                    if chunk:
                        passages.append(
                            WikipediaPassage.create(
                                canonical_title=canonical_title,
                                section_title=title,
                                section_index=index,
                                passage_index=c_idx,
                                text=" ".join(chunk),
                            )
                        )
                else:
                    passages.append(
                        WikipediaPassage.create(
                            canonical_title=canonical_title,
                            section_title=title,
                            section_index=index,
                            passage_index=p_idx,
                            text=para,
                        )
                    )

        return cls(
            section_id=sec_id,
            title=title,
            level=level,
            index=index,
            parent_section_title=parent_section_title,
            content=clean_content,
            passages=passages,
        )


class WikipediaDocument(BaseModel):
    """
    Comprehensive, normalized Wikipedia document model.
    Contains full article metadata, lead summary, hierarchical sections, and discovered links.
    """
    document_id: str
    canonical_title: str
    page_id: Optional[int] = None
    revision_id: Optional[str] = None
    url: str
    language: str = "en"
    lead_summary: str = ""
    sections: List[WikipediaDocumentSection] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    retrieved_at: float = Field(default_factory=time.time)
    content_hash: str = ""
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        canonical_title: str,
        url: str,
        lead_summary: str = "",
        page_id: Optional[int] = None,
        revision_id: Optional[str] = None,
        sections: Optional[List[WikipediaDocumentSection]] = None,
        links: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "WikipediaDocument":
        doc_id = _hash_id(canonical_title.strip().lower())
        full_text = lead_summary + "".join(s.content for s in (sections or []))
        chash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()[:32]
        return cls(
            document_id=doc_id,
            canonical_title=canonical_title.strip(),
            page_id=page_id,
            revision_id=revision_id,
            url=url,
            lead_summary=lead_summary.strip(),
            sections=sections or [],
            links=links or [],
            content_hash=chash,
            aliases=aliases or [],
            metadata=metadata or {},
        )

    def get_section(self, title: str) -> Optional[WikipediaDocumentSection]:
        """Case-insensitive section lookup."""
        norm_t = title.strip().lower()
        for s in self.sections:
            if s.title.strip().lower() == norm_t:
                return s
        for s in self.sections:
            if norm_t in s.title.strip().lower():
                return s
        return None

    def get_passages(self) -> List[WikipediaPassage]:
        """Aggregate all passages across all sections."""
        all_p = []
        for s in self.sections:
            all_p.extend(s.passages)
        return all_p

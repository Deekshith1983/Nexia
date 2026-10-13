"""Chunk builder for Nexia documents, respecting section hierarchy and paragraph boundaries."""

import hashlib
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.retrieval import Document


class ChunkPayload(BaseModel):
    """Payload representing a single processed chunk ready for embedding and storage."""

    chunk_index: int
    page_id: int
    document_title: str
    section_title: str
    section_level: int = 2
    parent_section: Optional[str] = None
    paragraph_id: str
    chunk_text: str
    source_url: str
    content_hash: str
    word_count: int


class ChunkerService:
    """Splits structured Document objects into paragraph-aligned chunks."""

    MIN_CHUNK_WORDS: int = 15
    MAX_CHUNK_WORDS: int = 500

    @classmethod
    def compute_hash(cls, text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    @classmethod
    def split_large_paragraph(cls, text: str, max_words: int) -> List[str]:
        """Split a long paragraph into sentence-aligned segments under max_words."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        segments: List[str] = []
        current_segment: List[str] = []
        current_word_count = 0

        for sent in sentences:
            sent_words = len(sent.split())
            if current_word_count + sent_words > max_words and current_segment:
                segments.append(" ".join(current_segment))
                current_segment = [sent]
                current_word_count = sent_words
            else:
                current_segment.append(sent)
                current_word_count += sent_words

        if current_segment:
            segments.append(" ".join(current_segment))

        return segments

    def chunk_document(self, document: Document) -> List[ChunkPayload]:
        """Convert a structured Document into a list of ChunkPayload objects."""
        chunks: List[ChunkPayload] = []
        page_id = document.metadata.page_id
        source_url = document.metadata.url
        chunk_counter = 0

        # Process all sections and paragraphs
        for section in document.sections:
            for p_idx, paragraph in enumerate(section.paragraphs):
                text = paragraph.text.strip()
                words = text.split()
                word_count = len(words)

                # Skip paragraphs that are too short to hold meaningful factual evidence
                if word_count < self.MIN_CHUNK_WORDS:
                    continue

                # If paragraph exceeds max words, break it at sentence boundaries
                if word_count > self.MAX_CHUNK_WORDS:
                    sub_texts = self.split_large_paragraph(text, self.MAX_CHUNK_WORDS)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        sub_words = len(sub_text.split())
                        if sub_words < self.MIN_CHUNK_WORDS:
                            continue
                        sub_p_id = f"{paragraph.id}_sub_{sub_idx}"
                        chunk_counter += 1
                        chunks.append(
                            ChunkPayload(
                                chunk_index=chunk_counter,
                                page_id=page_id,
                                document_title=document.title,
                                section_title=section.title,
                                section_level=section.level,
                                parent_section=section.parent_section,
                                paragraph_id=sub_p_id,
                                chunk_text=sub_text,
                                source_url=source_url,
                                content_hash=self.compute_hash(sub_text),
                                word_count=sub_words,
                            )
                        )
                else:
                    chunk_counter += 1
                    chunks.append(
                        ChunkPayload(
                            chunk_index=chunk_counter,
                            page_id=page_id,
                            document_title=document.title,
                            section_title=section.title,
                            section_level=section.level,
                            parent_section=section.parent_section,
                            paragraph_id=paragraph.id,
                            chunk_text=text,
                            source_url=source_url,
                            content_hash=self.compute_hash(text),
                            word_count=word_count,
                        )
                    )

        return chunks

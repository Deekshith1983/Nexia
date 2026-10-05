"""Article Parser for converting Wikipedia raw extracts into structured Documents."""

import re
from typing import List, Dict, Any, Tuple
from app.schemas.retrieval import Document, Section, Paragraph, DocumentMetadata


class ArticleParser:
    """Parses raw Wikipedia article extract text into structured sections and addressable paragraphs."""

    # Heading regex for wiki format (e.g., == Section ==, === Subsection ===)
    HEADING_REGEX = re.compile(r"(?m)^(={2,5})\s*(.+?)\s*\1\s*$")

    # Boilerplate sections to filter out from analytical passage retrieval
    IGNORED_SECTIONS = {
        "see also",
        "references",
        "further reading",
        "external links",
        "notes",
        "sources",
        "bibliography",
    }

    @classmethod
    def parse_raw_page(cls, page_data: Dict[str, Any]) -> Document:
        """Parse Wikipedia API page dict into a structured Document."""
        title = page_data.get("title", "")
        page_id = page_data.get("pageid", 0)
        url = page_data.get("fullurl") or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        raw_extract = page_data.get("extract", "")

        # Extract categories if present
        categories_raw = page_data.get("categories", [])
        categories = [
            cat.get("title", "").replace("Category:", "")
            for cat in categories_raw
            if "title" in cat
        ]

        # Parse text into introduction and sections
        introduction, sections = cls._parse_text_into_sections(raw_extract)

        # Compute document totals
        total_sections = len(sections)
        total_paragraphs = sum(len(sec.paragraphs) for sec in sections)
        
        # Word counts
        intro_words = len(introduction.split()) if introduction else 0
        section_words = sum(
            sum(p.word_count for p in sec.paragraphs) for sec in sections
        )
        total_words = intro_words + section_words

        metadata = DocumentMetadata(
            page_id=page_id,
            url=url,
            language="en",
            total_sections=total_sections,
            total_paragraphs=total_paragraphs,
            total_words=total_words,
            categories=categories,
        )

        return Document(
            title=title,
            page_id=page_id,
            url=url,
            introduction=introduction,
            sections=sections,
            metadata=metadata,
        )

    @classmethod
    def _parse_text_into_sections(cls, raw_text: str) -> Tuple[str, List[Section]]:
        """Split raw wiki-formatted text into introduction and structured sections."""
        if not raw_text.strip():
            return "", []

        # Find all headings with their positions
        heading_matches = list(cls.HEADING_REGEX.finditer(raw_text))

        if not heading_matches:
            # Article without headings - entire content is introduction / single block
            introduction = raw_text.strip()
            intro_paragraphs = cls._split_paragraphs(introduction, "Introduction", 0)
            if intro_paragraphs:
                return introduction, [
                    Section(
                        index=0,
                        title="Main Content",
                        level=2,
                        paragraphs=intro_paragraphs,
                    )
                ]
            return introduction, []

        # Lead text is everything before the first heading
        first_heading_start = heading_matches[0].start()
        introduction = raw_text[:first_heading_start].strip()

        sections: List[Section] = []
        section_idx = 0

        # Include lead paragraph as section 0 if it has content
        if introduction:
            intro_paragraphs = cls._split_paragraphs(introduction, "Introduction", section_idx)
            if intro_paragraphs:
                sections.append(
                    Section(
                        index=section_idx,
                        title="Introduction",
                        level=2,
                        paragraphs=intro_paragraphs,
                    )
                )
                section_idx += 1

        for i, match in enumerate(heading_matches):
            level = len(match.group(1))
            heading_title = match.group(2).strip()

            # Skip boilerplate reference/external links sections
            if heading_title.lower() in cls.IGNORED_SECTIONS:
                continue

            # Determine content span between this heading and the next
            content_start = match.end()
            content_end = (
                heading_matches[i + 1].start()
                if i + 1 < len(heading_matches)
                else len(raw_text)
            )

            section_body = raw_text[content_start:content_end].strip()
            paragraphs = cls._split_paragraphs(section_body, heading_title, section_idx)

            if paragraphs:
                sections.append(
                    Section(
                        index=section_idx,
                        title=heading_title,
                        level=level,
                        paragraphs=paragraphs,
                    )
                )
                section_idx += 1

        return introduction, sections

    @classmethod
    def _split_paragraphs(
        cls, body_text: str, section_title: str, section_idx: int
    ) -> List[Paragraph]:
        """Split a section body into individually addressable paragraphs."""
        if not body_text:
            return []

        # Split on two or more newlines
        raw_blocks = re.split(r"\n\s*\n+", body_text)
        paragraphs: List[Paragraph] = []
        p_idx = 0

        for block in raw_blocks:
            cleaned = block.strip()
            # Ignore empty blocks or trivial single-character markers
            if not cleaned or len(cleaned) < 5:
                continue

            # Clean internal excessive whitespaces while preserving paragraph flow
            normalized_text = re.sub(r"\s+", " ", cleaned)
            word_count = len(normalized_text.split())

            paragraphs.append(
                Paragraph(
                    id=f"sec_{section_idx}_p_{p_idx}",
                    section_title=section_title,
                    index=p_idx,
                    text=normalized_text,
                    word_count=word_count,
                )
            )
            p_idx += 1

        return paragraphs

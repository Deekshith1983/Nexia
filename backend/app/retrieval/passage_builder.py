"""Passage Builder: Converts structured Documents into independently retrievable Passages."""

from typing import List
from app.schemas.retrieval import Document, Passage


class PassageBuilder:
    """Extracts and standardizes all paragraphs from a structured Document into a flat, addressable list of Passages."""

    @staticmethod
    def build_passages(document: Document) -> List[Passage]:
        """Convert a structured hierarchical Document into a list of traceable Passage units."""
        passages: List[Passage] = []

        for sec in document.sections:
            for p in sec.paragraphs:
                # Omit trivially short non-informative fragments (< 4 words)
                if p.word_count < 4:
                    continue

                passage_id = f"pass_{document.page_id}_{sec.index}_{p.index}"
                passages.append(
                    Passage(
                        passage_id=passage_id,
                        document_title=document.title,
                        document_page_id=document.page_id,
                        section_title=sec.title,
                        section_level=sec.level,
                        paragraph_id=p.id,
                        text=p.text,
                        word_count=p.word_count,
                    )
                )

        return passages

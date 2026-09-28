"""
Finding — a derived piece of knowledge from accumulated research evidence.

Findings represent meaningful informational conclusions extracted during
the research phase. They record which evidence supports each statement
and which Wikipedia pages are the authoritative sources.

Findings are temporary — scoped to the current research run only.
They do not become persistent memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Finding:
    """
    A meaningful piece of information derived from retrieved Wikipedia evidence.

    Fields:
        statement            — a factual claim supported by the evidence
        supporting_evidence  — action_ids of EvidenceItems that support this finding
        source_references    — Wikipedia page titles that are the authoritative sources
        metadata             — additional context (confidence, derivation method, etc.)
    """
    statement: str
    supporting_evidence: List[str]   # action_ids
    source_references: List[str]     # page_titles
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "supporting_evidence": self.supporting_evidence,
            "source_references": self.source_references,
        }

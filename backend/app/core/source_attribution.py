from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.services.wikipedia_retriever import RetrievalResult, RetrievalStatus
from app.core.answer_validator import ValidationResult, ValidationStatus


class SourceMetadata(BaseModel):
    """Structured representation of a single verified external source."""
    source: str = "Wikipedia"
    title: str
    url: str
    metadata: Dict = Field(default_factory=dict)


class FinalResponse(BaseModel):
    """Structured representation of the final canonical Nexia response."""
    content: str
    sources: List[SourceMetadata] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)


class SourceAttributor:
    """Component responsible for attaching verified retrieval source metadata to accepted responses."""

    def build_sources(
        self,
        evidence: Optional[RetrievalResult],
        validation: Optional[ValidationResult],
    ) -> List[SourceMetadata]:
        """
        Extract verified source metadata ONLY if:
        1. Valid evidence exists (status == SUCCESS, with valid title and URL).
        2. Validation was not explicitly UNSUPPORTED or INSUFFICIENT_EVIDENCE.

        VALIDATION_ERROR is treated as a tooling failure (not a factual rejection),
        so sources are still attributed — the answer was grounded even if the
        validator's own LLM output failed to parse correctly.
        """
        if not evidence or evidence.status != RetrievalStatus.SUCCESS:
            return []

        # Suppress sources only for definitive factual rejection statuses
        if validation and validation.status in (
            ValidationStatus.UNSUPPORTED,
            ValidationStatus.INSUFFICIENT_EVIDENCE,
        ):
            return []

        if not evidence.page_title or not evidence.page_url:
            return []

        return [
            SourceMetadata(
                source=evidence.source or "Wikipedia",
                title=evidence.page_title,
                url=evidence.page_url,
                metadata=evidence.metadata,
            )
        ]


source_attributor = SourceAttributor()

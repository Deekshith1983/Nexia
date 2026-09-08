import re
from typing import Dict, Optional
from pydantic import BaseModel, Field
from app.core.llm_interface import BaseLanguageModel
from app.core.conversation_manager import ConversationState
from app.core.query_understanding import QueryUnderstanding

class ResolutionStatus:
    RESOLVED = "resolved"
    NOT_REQUIRED = "not_required"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"

class ContextResolution(BaseModel):
    """Structured representation of conversational context resolution."""
    original_query: str
    resolved_query: str
    resolved_entity: Optional[str] = None
    resolved_references: Dict[str, str] = Field(default_factory=dict)
    resolution_status: str = ResolutionStatus.NOT_REQUIRED
    confidence: float = 1.0
    metadata: Dict = Field(default_factory=dict)

class ContextManager:
    """Component responsible for coreference resolution and context tracking."""

    PRONOUN_PATTERN = re.compile(
        r"\b(he|him|his|she|her|hers|it|its|they|them|their|theirs)\b", re.IGNORECASE
    )

    POSSESSIVE_PRONOUNS = {"his", "her", "hers", "its", "their", "theirs"}

    def _possessive_form(self, entity: str) -> str:
        if entity.endswith("s") or entity.endswith("S"):
            return f"{entity}'"
        return f"{entity}'s"

    async def resolve(
        self,
        qu: QueryUnderstanding,
        state: ConversationState,
        llm: Optional[BaseLanguageModel] = None,
    ) -> ContextResolution:
        original_q = qu.original_query.strip()

        # Check if query introduces an explicit entity
        if qu.entity:
            # Update state with newly introduced explicit entity (Topic Switching)
            state.current_entity = qu.entity
            state.current_topic = qu.entity

        # Check for explicit ambiguity flag or multiple competing entities without antecedent
        if qu.metadata.get("is_ambiguous", False):
            return ContextResolution(
                original_query=original_q,
                resolved_query=original_q,
                resolved_entity=state.current_entity,
                resolution_status=ResolutionStatus.AMBIGUOUS,
                confidence=0.3,
            )

        # Detect pronouns or references in query
        matches = self.PRONOUN_PATTERN.findall(original_q)
        
        if not matches:
            # No reference pronoun present
            return ContextResolution(
                original_query=original_q,
                resolved_query=original_q,
                resolved_entity=state.current_entity,
                resolution_status=ResolutionStatus.NOT_REQUIRED,
                confidence=1.0,
            )

        # Reference pronouns detected but no established current entity
        if not state.current_entity:
            return ContextResolution(
                original_query=original_q,
                resolved_query=original_q,
                resolved_entity=None,
                resolution_status=ResolutionStatus.UNRESOLVED,
                confidence=0.0,
            )

        # Perform coreference resolution using established state entity
        current_ent = state.current_entity
        resolved_q = original_q
        resolved_refs: Dict[str, str] = {}

        for match in matches:
            ref_lower = match.lower()
            if ref_lower in self.POSSESSIVE_PRONOUNS:
                replacement = self._possessive_form(current_ent)
            else:
                replacement = current_ent

            # Replace exact match word boundary
            pattern = re.compile(r"\b" + re.escape(match) + r"\b")
            resolved_q = pattern.sub(replacement, resolved_q, count=1)
            resolved_refs[match] = current_ent

        return ContextResolution(
            original_query=original_q,
            resolved_query=resolved_q,
            resolved_entity=current_ent,
            resolved_references=resolved_refs,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=0.95,
        )

context_manager = ContextManager()

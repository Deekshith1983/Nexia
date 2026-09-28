import logging
import re
from typing import Dict, Optional
from pydantic import BaseModel, Field
from app.core.llm_interface import BaseLanguageModel
from app.core.conversation_manager import ConversationState
from app.core.query_understanding import QueryUnderstanding, _REFERENCE_PRONOUNS

logger = logging.getLogger("nexia.ctx")


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


def _is_pronoun_entity(entity: Optional[str]) -> bool:
    """Return True if the entity string is actually a pronoun (not a real entity)."""
    if not entity:
        return False
    first_word = entity.strip().split()[0].lower()
    return first_word in _REFERENCE_PRONOUNS


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

        entity_before = state.current_entity
        topic_before = state.current_topic

        # Update state entity ONLY if QU produced a clean, non-pronoun entity.
        # If qu.entity starts with a pronoun (e.g. "he born"), it is NOT a real entity
        # and must not overwrite the established current_entity.
        if qu.entity and not _is_pronoun_entity(qu.entity):
            state.current_entity = qu.entity
            state.current_topic = qu.entity

        # Check for explicit ambiguity flag
        if qu.metadata.get("is_ambiguous", False):
            self._log_trace(
                original_q=original_q,
                resolved_q=original_q,
                entity_before=entity_before,
                entity_after=state.current_entity,
                matches=[],
                resolved_refs={},
                status=ResolutionStatus.AMBIGUOUS,
            )
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
            # No reference pronoun present — no resolution needed
            self._log_trace(
                original_q=original_q,
                resolved_q=original_q,
                entity_before=entity_before,
                entity_after=state.current_entity,
                matches=[],
                resolved_refs={},
                status=ResolutionStatus.NOT_REQUIRED,
            )
            return ContextResolution(
                original_query=original_q,
                resolved_query=original_q,
                resolved_entity=state.current_entity,
                resolution_status=ResolutionStatus.NOT_REQUIRED,
                confidence=1.0,
            )

        # Reference pronouns detected but no established current entity
        if not state.current_entity:
            self._log_trace(
                original_q=original_q,
                resolved_q=original_q,
                entity_before=entity_before,
                entity_after=None,
                matches=matches,
                resolved_refs={},
                status=ResolutionStatus.UNRESOLVED,
            )
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

        self._log_trace(
            original_q=original_q,
            resolved_q=resolved_q,
            entity_before=entity_before,
            entity_after=state.current_entity,
            matches=matches,
            resolved_refs=resolved_refs,
            status=ResolutionStatus.RESOLVED,
        )

        return ContextResolution(
            original_query=original_q,
            resolved_query=resolved_q,
            resolved_entity=current_ent,
            resolved_references=resolved_refs,
            resolution_status=ResolutionStatus.RESOLVED,
            confidence=0.95,
        )

    def _log_trace(
        self,
        original_q: str,
        resolved_q: str,
        entity_before: Optional[str],
        entity_after: Optional[str],
        matches: list,
        resolved_refs: Dict,
        status: str,
    ) -> None:
        logger.debug(
            "[CTX] original_query=%r  resolved_query=%r  "
            "entity_before=%r  entity_after=%r  "
            "detected_refs=%r  resolved_refs=%r  status=%s",
            original_q,
            resolved_q,
            entity_before,
            entity_after,
            matches,
            resolved_refs,
            status,
        )


context_manager = ContextManager()

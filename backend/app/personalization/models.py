"""
Models for Stage 11 Adaptive Personalization & Interaction.
"""
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field


class PreferenceCategory(str, Enum):
    """Supported preference categories for adaptive personalization."""
    RESPONSE_STYLE = "response_style"
    RESPONSE_LENGTH = "response_length"
    FORMATTING = "formatting"
    EXPLANATION_DEPTH = "explanation_depth"
    RESEARCH_DEPTH = "research_depth"
    SOURCE_PRESENTATION = "source_presentation"
    INTERACTION_BEHAVIOR = "interaction_behavior"


class PreferenceSource(str, Enum):
    """Source origin of a user preference."""
    EXPLICIT = "EXPLICIT"    # Direct user command/instruction
    INFERRED = "INFERRED"    # Derived from repeated interaction patterns
    DEFAULT = "DEFAULT"      # System fallback


class PreferenceStatus(str, Enum):
    """Lifecycle status of a preference."""
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    DELETED = "DELETED"


class UserPreference(BaseModel):
    """
    Structured persistent user preference record.
    Influences presentation and interaction behavior while preserving factual truth.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str = "default"
    category: PreferenceCategory
    key: str
    value: str
    source: PreferenceSource = PreferenceSource.EXPLICIT
    confidence: float = 1.0
    status: PreferenceStatus = PreferenceStatus.ACTIVE
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EffectiveInteractionProfile(BaseModel):
    """
    Transient, request-level interaction profile.
    Derived dynamically from current request, conversation context, and active preferences.
    """
    response_length: Optional[str] = None       # "concise", "detailed", "standard"
    explanation_depth: Optional[str] = None     # "low", "medium", "high"
    formatting: Optional[str] = None            # "bullets", "structured_sections", "paragraphs", "code_first"
    research_depth: Optional[str] = None        # "quick_overview", "thorough", "standard"
    source_presentation: Optional[str] = None   # "prominent", "standard", "compact"
    interaction_preferences: Dict[str, Any] = Field(default_factory=dict)
    applied_preferences: List[str] = Field(default_factory=list)
    overridden_preferences: List[str] = Field(default_factory=list)

    def to_guidance_text(self) -> str:
        """Convert profile into safe, non-invasive prompt guidance instructions."""
        guidance_lines = []
        if self.response_length == "concise":
            guidance_lines.append("- Presentation style: Provide a concise, highly focused explanation without unnecessary filler.")
        elif self.response_length == "detailed":
            guidance_lines.append("- Presentation style: Provide a comprehensive and thorough explanation covering all key aspects.")

        if self.explanation_depth == "high":
            guidance_lines.append("- Explanation depth: Provide in-depth technical details and contextual explanations.")
        elif self.explanation_depth == "low":
            guidance_lines.append("- Explanation depth: Keep explanations simple, direct, and accessible.")

        if self.formatting == "bullets":
            guidance_lines.append("- Output formatting: Organize the answer clearly using structured bullet points.")
        elif self.formatting == "structured_sections":
            guidance_lines.append("- Output formatting: Organize the answer with distinct structured sections and clear subheadings.")
        elif self.formatting == "code_first":
            guidance_lines.append("- Output formatting: Prioritize code examples and structured technical blocks.")

        if not guidance_lines:
            return ""
        return "\nPresentation Preferences:\n" + "\n".join(guidance_lines) + "\n"

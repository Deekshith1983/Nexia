"""
PersonalizationManager for Stage 11 Adaptive Personalization & Interaction.

Coordinates preference lifecycle, conflict supersession, relevance filtering,
and transient request-level EffectiveInteractionProfile resolution.
"""
from collections import defaultdict
import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.memory.manager import memory_manager
from app.memory.models import MemoryType
from app.personalization.models import (
    EffectiveInteractionProfile, PreferenceCategory, PreferenceSource,
    PreferenceStatus, UserPreference,
)
from app.personalization.store import PersonalizationStore, personalization_store

logger = logging.getLogger("nexia.personalization.manager")

# Immutable policy invariants that user preferences cannot violate
_FORBIDDEN_PREFERENCE_PATTERNS = [
    r"\b(?:ignore|skip|disable|bypass)\s+(?:wikipedia|evidence|grounding|validation|facts|truth|sources)\b",
    r"\b(?:hallucinate|make\s+up|invent\s+facts)\b",
    r"\b(?:disable\s+safety|allow\s+malicious)\b",
]


class PersonalizationManager:
    """
    Dedicated manager for adaptive personalization and user preferences.
    
    Enforces strict hierarchy:
      1. Current Explicit Request (Highest Priority / Temporary Override)
      2. Current Conversation Context
      3. Explicit Persistent Preferences (source=EXPLICIT, confidence=1.0)
      4. Reliable Inferred Preferences (source=INFERRED, threshold-gated)
      5. Default Behavior
    """

    def __init__(self, store: Optional[PersonalizationStore] = None):
        self.store = store or personalization_store
        # In-memory tracking of repeated behavioral interaction signals
        # {user_id: {signal_key: count}}
        self._interaction_signals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def is_policy_compliant(self, text: str) -> bool:
        """Verify that preference does not attempt to violate immutable safety/grounding policies."""
        cleaned = text.lower()
        for pat in _FORBIDDEN_PREFERENCE_PATTERNS:
            if re.search(pat, cleaned):
                return False
        return True

    def store_preference(
        self,
        category: PreferenceCategory,
        key: str,
        value: str,
        user_id: str = "default",
        source: PreferenceSource = PreferenceSource.EXPLICIT,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[UserPreference]:
        """
        Store or update a user preference. Automatically supersedes conflicting older preferences
        in the same category/key domain.
        """
        raw_combined = f"{key} {value}"
        if not self.is_policy_compliant(raw_combined):
            logger.warning("[PREF MGR] Rejected policy-violating preference: %r", raw_combined)
            return None

        # 1. Supersede existing active preferences in the same category & key
        active_prefs = self.store.list_active(user_id=user_id, category=category)
        for old_pref in active_prefs:
            if old_pref.key.lower() == key.lower():
                logger.info(
                    "[PREF MGR] Superseding older preference id=%s key=%s old_val=%r with new_val=%r",
                    old_pref.id, old_pref.key, old_pref.value, value
                )
                self.store.update_status(old_pref.id, PreferenceStatus.SUPERSEDED, user_id=user_id)

        # 2. Persist new preference
        pref = UserPreference(
            user_id=user_id,
            category=category,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
            status=PreferenceStatus.ACTIVE,
            metadata=metadata or {},
        )
        saved = self.store.save(pref)

        # 3. Synchronize with Stage 4 MemoryManager
        try:
            mem_content = f"User preference for {category.value} ({key}): {value}"
            memory_manager.store_memory(
                content=mem_content,
                user_id=user_id,
                memory_type=MemoryType.INTERACTION_PREFERENCE if category in (PreferenceCategory.RESPONSE_STYLE, PreferenceCategory.RESPONSE_LENGTH, PreferenceCategory.FORMATTING) else MemoryType.PREFERENCE,
                source="user_explicit" if source == PreferenceSource.EXPLICIT else "learned",
                metadata={"preference_id": saved.id, "category": category.value, "key": key},
            )
        except Exception as exc:
            logger.debug("[PREF MGR] Memory sync note: %s", exc)

        return saved

    def get_preference(self, preference_id: str, user_id: str = "default") -> Optional[UserPreference]:
        """Fetch preference by ID."""
        return self.store.get(preference_id, user_id=user_id)

    def list_preferences(
        self,
        user_id: str = "default",
        active_only: bool = True,
        category: Optional[PreferenceCategory] = None,
    ) -> List[UserPreference]:
        """List preferences for user."""
        if active_only:
            return self.store.list_active(user_id=user_id, category=category)
        return [p for p in self.store.list_all(user_id=user_id) if (category is None or p.category == category)]

    def delete_preference(self, preference_id: str, user_id: str = "default") -> bool:
        """Mark preference as deleted."""
        return self.store.delete(preference_id, user_id=user_id)

    def forget_preference(self, pattern: str, user_id: str = "default") -> int:
        """Delete preferences matching pattern."""
        return self.store.delete_by_pattern(pattern, user_id=user_id)

    # ── Behavioral Inference Signal Tracking ──────────────────────────────────

    def record_interaction_signal(self, signal_key: str, user_id: str = "default", threshold: int = 3) -> Optional[UserPreference]:
        """
        Record an interaction behavior signal.
        If the signal repeats >= threshold times, candidate inferred preference is established.
        Single interactions never create permanent preferences.
        """
        self._interaction_signals[user_id][signal_key] += 1
        count = self._interaction_signals[user_id][signal_key]

        if count >= threshold:
            # Map signal to inferred preference
            if signal_key == "shorter_answer":
                return self.store_preference(
                    category=PreferenceCategory.RESPONSE_LENGTH,
                    key="length",
                    value="concise",
                    user_id=user_id,
                    source=PreferenceSource.INFERRED,
                    confidence=0.75,
                    metadata={"signal_count": count},
                )
            elif signal_key == "bullet_format":
                return self.store_preference(
                    category=PreferenceCategory.FORMATTING,
                    key="format",
                    value="bullets",
                    user_id=user_id,
                    source=PreferenceSource.INFERRED,
                    confidence=0.75,
                    metadata={"signal_count": count},
                )
        return None

    # ── Natural Language Preference Parsing ────────────────────────────────────

    def parse_preference_from_query(self, text: str) -> List[Tuple[PreferenceCategory, str, str]]:
        """
        Deterministic extraction of structured preference directives from text.
        Returns list of (category, key, value).
        """
        cleaned = text.lower().strip()
        results: List[Tuple[PreferenceCategory, str, str]] = []

        # 1. Response length
        if re.search(r"\b(?:prefer|want|like|give\s+me|use|from\s+now\s+on)\s+(?:concise|brief|short)\s+(?:answers?|responses?|explanations?)?\b", cleaned) or \
           re.search(r"\b(?:keep\s+(?:it|answers?|responses?)\s+(?:concise|brief|short))\b", cleaned):
            results.append((PreferenceCategory.RESPONSE_LENGTH, "length", "concise"))

        elif re.search(r"\b(?:prefer|want|like|give\s+me|use|from\s+now\s+on)\s+(?:detailed|comprehensive|thorough|in-depth)\s+(?:answers?|responses?|explanations?)?\b", cleaned):
            results.append((PreferenceCategory.RESPONSE_LENGTH, "length", "detailed"))
            results.append((PreferenceCategory.EXPLANATION_DEPTH, "depth", "high"))

        # 2. Explanation depth
        if re.search(r"\b(?:prefer|want|like|give\s+me)\s+(?:simple|accessible|beginner-friendly|high-level)\s+(?:explanations?|answers?)\b", cleaned):
            results.append((PreferenceCategory.EXPLANATION_DEPTH, "depth", "low"))
        elif re.search(r"\b(?:prefer|want|like|give\s+me)\s+(?:technical|deep|expert|advanced)\s+(?:explanations?|answers?)\b", cleaned):
            results.append((PreferenceCategory.EXPLANATION_DEPTH, "depth", "high"))

        # 3. Formatting
        if re.search(r"\b(?:prefer|want|like|use|format\s+in|give\s+me|from\s+now\s+on)\s+(?:bullet\s+points?|bullets?|list\s+format)\b", cleaned):
            results.append((PreferenceCategory.FORMATTING, "format", "bullets"))
        elif re.search(r"\b(?:prefer|want|like|use|format\s+in)\s+(?:structured\s+sections?|sections?|headings?)\b", cleaned):
            results.append((PreferenceCategory.FORMATTING, "format", "structured_sections"))
        elif re.search(r"\b(?:prefer|want|like|use|format\s+in)\s+(?:paragraphs?|prose)\b", cleaned):
            results.append((PreferenceCategory.FORMATTING, "format", "paragraphs"))

        # 4. Source presentation
        if re.search(r"\b(?:always\s+show\s+sources|include\s+sources\s+prominently|highlight\s+sources)\b", cleaned):
            results.append((PreferenceCategory.SOURCE_PRESENTATION, "presentation", "prominent"))

        # 5. Research depth
        if re.search(r"\b(?:prefer|want|give\s+me)\s+(?:quick\s+overviews?|fast\s+summaries?)\b", cleaned):
            results.append((PreferenceCategory.RESEARCH_DEPTH, "research_depth", "quick_overview"))
        elif re.search(r"\b(?:prefer|want|conduct)\s+(?:deep\s+research|thorough\s+research|exhaustive\s+research)\b", cleaned):
            results.append((PreferenceCategory.RESEARCH_DEPTH, "research_depth", "thorough"))

        return results

    # ── Dynamic Profile Resolution ────────────────────────────────────────────

    def resolve_effective_profile(
        self,
        current_query: str,
        user_id: str = "default",
        conversation_state: Optional[Any] = None,
    ) -> EffectiveInteractionProfile:
        """
        Dynamically construct the request-level EffectiveInteractionProfile.
        
        Priority:
          1. Current Explicit Request (Highest / Temporary Override)
          2. Explicit Persistent Preferences (source=EXPLICIT)
          3. Inferred Persistent Preferences (source=INFERRED)
          4. Default Fallback
        """
        profile = EffectiveInteractionProfile()
        applied_set: List[str] = []
        overridden_set: List[str] = []

        # 1. Fetch active stored preferences
        stored_prefs = self.store.list_active(user_id=user_id)
        prefs_by_cat: Dict[PreferenceCategory, UserPreference] = {}
        for p in stored_prefs:
            if p.category not in prefs_by_cat:
                prefs_by_cat[p.category] = p
            else:
                # If existing is inferred and new is explicit, explicit wins
                existing = prefs_by_cat[p.category]
                if existing.source == PreferenceSource.INFERRED and p.source == PreferenceSource.EXPLICIT:
                    prefs_by_cat[p.category] = p
                elif p.confidence > existing.confidence:
                    prefs_by_cat[p.category] = p

        # Apply stored preferences to baseline
        for cat, pref in prefs_by_cat.items():
            val = pref.value.lower()
            if cat == PreferenceCategory.RESPONSE_LENGTH:
                profile.response_length = val
                applied_set.append(f"stored:response_length={val}")
            elif cat == PreferenceCategory.EXPLANATION_DEPTH:
                profile.explanation_depth = val
                applied_set.append(f"stored:explanation_depth={val}")
            elif cat == PreferenceCategory.FORMATTING:
                profile.formatting = val
                applied_set.append(f"stored:formatting={val}")
            elif cat == PreferenceCategory.RESEARCH_DEPTH:
                profile.research_depth = val
                applied_set.append(f"stored:research_depth={val}")
            elif cat == PreferenceCategory.SOURCE_PRESENTATION:
                profile.source_presentation = val
                applied_set.append(f"stored:source_presentation={val}")
            else:
                profile.interaction_preferences[pref.key] = val

        # 2. Temporary Overrides from Current Request
        q_lower = current_query.lower()

        # Temporary length override
        if re.search(r"\b(?:in\s+detail|detailed\s+explanation|explain\s+in\s+detail|elaborate|comprehensive)\b", q_lower):
            if profile.response_length and profile.response_length != "detailed":
                overridden_set.append(f"response_length:{profile.response_length}->detailed")
            profile.response_length = "detailed"
            profile.explanation_depth = "high"
            applied_set.append("override:response_length=detailed")

        elif re.search(r"\b(?:concise|briefly|in\s+brief|short\s+answer|quick\s+summary|in\s+short)\b", q_lower):
            if profile.response_length and profile.response_length != "concise":
                overridden_set.append(f"response_length:{profile.response_length}->concise")
            profile.response_length = "concise"
            applied_set.append("override:response_length=concise")

        # Temporary formatting override
        if re.search(r"\b(?:in\s+bullet\s+points|use\s+bullets|as\s+a\s+bulleted\s+list|as\s+bullets)\b", q_lower):
            if profile.formatting and profile.formatting != "bullets":
                overridden_set.append(f"formatting:{profile.formatting}->bullets")
            profile.formatting = "bullets"
            applied_set.append("override:formatting=bullets")
        elif re.search(r"\b(?:in\s+sections|structured\s+sections|with\s+headings)\b", q_lower):
            if profile.formatting and profile.formatting != "structured_sections":
                overridden_set.append(f"formatting:{profile.formatting}->structured_sections")
            profile.formatting = "structured_sections"
            applied_set.append("override:formatting=structured_sections")

        # Temporary research depth override
        if re.search(r"\b(?:quick\s+overview|brief\s+overview|quick\s+look)\b", q_lower):
            if profile.research_depth and profile.research_depth != "quick_overview":
                overridden_set.append(f"research_depth:{profile.research_depth}->quick_overview")
            profile.research_depth = "quick_overview"
            profile.response_length = "concise"
            applied_set.append("override:research_depth=quick_overview")
        elif re.search(r"\b(?:research\s+this\s+thoroughly|deep\s+research|in-depth\s+investigation)\b", q_lower):
            if profile.research_depth and profile.research_depth != "thorough":
                overridden_set.append(f"research_depth:{profile.research_depth}->thorough")
            profile.research_depth = "thorough"
            profile.explanation_depth = "high"
            applied_set.append("override:research_depth=thorough")

        profile.applied_preferences = applied_set
        profile.overridden_preferences = overridden_set
        return profile


personalization_manager = PersonalizationManager()

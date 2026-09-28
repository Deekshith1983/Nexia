"""
MemoryManager — Central coordinator for Nexia's persistent long-term memory.
"""
import logging
from typing import Any, Dict, List, Optional

from app.memory.models import Memory, MemoryStatus, MemoryType
from app.memory.policy import MemoryPolicy, memory_policy
from app.memory.store import MemoryStore

logger = logging.getLogger("nexia.memory.manager")


class MemoryManager:
    """
    Coordinates persistent storage, policy enforcement, relevance retrieval,
    and lifecycle management for structured user memory.
    """

    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        policy: Optional[MemoryPolicy] = None,
    ):
        self.store = store or MemoryStore()
        self.policy = policy or memory_policy

    def store_memory(
        self,
        content: str,
        user_id: str = "default",
        memory_type: Optional[MemoryType] = None,
        source: str = "user_explicit",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Memory]:
        """
        Store a new memory according to policy rules, superseding conflicting older records.
        """
        cleaned_content = content.strip()
        if not cleaned_content:
            return None

        # If learned/extracted (not explicitly commanded), apply policy filter
        if source != "user_explicit" and not self.policy.should_store(cleaned_content):
            logger.debug(
                "[MEMORY] Candidate rejected by policy: %r", cleaned_content[:60]
            )
            return None

        # Determine type if not specified
        final_type = memory_type or self.policy.extract_memory_type(cleaned_content)

        # Look up existing active memories for supersession check
        active = self.store.list_active(user_id=user_id)
        superseded_records = self.policy.find_superseded(
            cleaned_content, final_type, active
        )
        for old_mem in superseded_records:
            logger.info(
                "[MEMORY] Superseding older memory id=%s content=%r with new content=%r",
                old_mem.id[:8], old_mem.content[:40], cleaned_content[:40]
            )
            self.store.update_status(old_mem.id, MemoryStatus.SUPERSEDED, user_id=user_id)

        # Create and persist new record
        new_memory = Memory(
            user_id=user_id,
            type=final_type,
            content=cleaned_content,
            status=MemoryStatus.ACTIVE,
            source=source,
            metadata=metadata or {},
        )
        persisted = self.store.add(new_memory)
        return persisted

    def forget_memory(
        self,
        query_or_pattern: str,
        user_id: str = "default",
    ) -> int:
        """
        Deactivate matching active memories (mark as DELETED).
        Returns number of memories forgotten.
        """
        count = self.store.delete_by_pattern(query_or_pattern, user_id=user_id)
        logger.info(
            "[MEMORY] Forgot %d memories matching pattern=%r user_id=%s",
            count, query_or_pattern, user_id
        )
        return count

    def recall_memories(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 10,
    ) -> List[Memory]:
        """
        Recall active memories relevant to a user request or broad category.
        """
        q_lower = query.strip().lower()

        # Broad recall categories
        if q_lower in ("all", "everything", "memories", "saved memories", "list", ""):
            return self.store.list_active(user_id=user_id)[:limit]

        if "preference" in q_lower:
            prefs = self.store.list_active(user_id=user_id, memory_type=MemoryType.PREFERENCE)
            inter_prefs = self.store.list_active(
                user_id=user_id, memory_type=MemoryType.INTERACTION_PREFERENCE
            )
            combined = prefs + [m for m in inter_prefs if m.id not in {p.id for p in prefs}]
            return combined[:limit]

        if "project" in q_lower:
            proj = self.store.list_active(user_id=user_id, memory_type=MemoryType.PROJECT_CONTEXT)
            if proj:
                return proj[:limit]

        if "profile" in q_lower or "about me" in q_lower:
            prof = self.store.list_active(user_id=user_id, memory_type=MemoryType.PROFILE_FACT)
            if prof:
                return prof[:limit]

        # Targeted search
        return self.store.search_relevant(query, user_id=user_id, limit=limit)

    def get_relevant_memories(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 3,
    ) -> List[Memory]:
        """
        Retrieve active memories relevant to the query for background context injection.
        """
        return self.store.search_relevant(query, user_id=user_id, limit=limit)

    def is_memory_relevant(
        self,
        query: str,
        user_id: str = "default",
    ) -> bool:
        """
        Check if any active memory is relevant to the query.
        """
        relevant = self.get_relevant_memories(query, user_id=user_id, limit=1)
        return len(relevant) > 0

    # ── Inspection & CRUD API ──────────────────────────────────────────────────

    def get_memory(self, memory_id: str, user_id: str = "default") -> Optional[Memory]:
        """Get a memory by ID."""
        return self.store.get(memory_id, user_id=user_id)

    def list_active(
        self,
        user_id: str = "default",
        memory_type: Optional[MemoryType] = None,
    ) -> List[Memory]:
        """List all active memories for a user."""
        return self.store.list_active(user_id=user_id, memory_type=memory_type)

    def list_all(self, user_id: str = "default") -> List[Memory]:
        """List all memories for a user, including superseded and deleted."""
        return self.store.list_all(user_id=user_id)

    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        user_id: str = "default",
    ) -> Optional[Memory]:
        """Update the content of an active memory."""
        return self.store.update_content(memory_id, new_content, user_id=user_id)

    def delete_memory(self, memory_id: str, user_id: str = "default") -> bool:
        """Mark a memory as DELETED."""
        return self.store.delete(memory_id, user_id=user_id)

    def clear_all(self, user_id: Optional[str] = None) -> None:
        """Clear memories (testing/reset)."""
        self.store.clear_all(user_id=user_id)


# Global singleton
memory_manager = MemoryManager()

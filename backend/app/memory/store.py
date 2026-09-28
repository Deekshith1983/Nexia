"""
SQLite persistent storage for Nexia's long-term memory.
"""
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.config import settings
from app.memory.models import Memory, MemoryStatus, MemoryType

logger = logging.getLogger("nexia.memory.store")

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "to", "of", "in",
    "on", "at", "by", "for", "with", "about", "that", "this", "it", "my", "your",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "what", "who", "which", "where", "when", "why", "how", "do", "does", "did",
    "have", "has", "had", "be", "been", "being", "tell", "show", "know", "remember",
    "please", "can", "could", "would", "should", "am", "as", "if"
}


class MemoryStore:
    """
    Persistent SQLite storage engine for structured long-term memories.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = settings.memory_db_path

        # If relative path, resolve relative to backend root
        path_obj = Path(db_path)
        if not path_obj.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent.parent
            path_obj = backend_root / db_path

        self.db_path = str(path_obj)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and row factory."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema and indexes."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_user_type ON memories(user_id, type);"
            )
            conn.commit()
        logger.debug("[MEMORY STORE] Initialized DB at %s", self.db_path)

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert a SQLite row to a Memory model instance."""
        meta = {}
        try:
            meta = json.loads(row["metadata"])
        except Exception:
            meta = {}

        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            type=MemoryType(row["type"]),
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            status=MemoryStatus(row["status"]),
            source=row["source"],
            metadata=meta,
        )

    def add(self, memory: Memory) -> Memory:
        """Persist a new memory record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, user_id, type, content, created_at, updated_at, status, source, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    memory.id,
                    memory.user_id,
                    memory.type.value,
                    memory.content,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.status.value,
                    memory.source,
                    json.dumps(memory.metadata),
                ),
            )
            conn.commit()
        logger.info(
            "[MEMORY STORE] Added memory id=%s user_id=%s type=%s content=%r",
            memory.id[:8], memory.user_id, memory.type.value, memory.content[:60]
        )
        return memory

    def get(self, memory_id: str, user_id: str = "default") -> Optional[Memory]:
        """Fetch a specific memory record by ID and user_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?;",
                (memory_id, user_id),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_memory(row)
        return None

    def list_active(
        self,
        user_id: str = "default",
        memory_type: Optional[MemoryType] = None,
    ) -> List[Memory]:
        """List active memories for a given user."""
        with self._get_connection() as conn:
            if memory_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND status = ? AND type = ?
                    ORDER BY updated_at DESC;
                    """,
                    (user_id, MemoryStatus.ACTIVE.value, memory_type.value),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE user_id = ? AND status = ?
                    ORDER BY updated_at DESC;
                    """,
                    (user_id, MemoryStatus.ACTIVE.value),
                )
            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def list_all(self, user_id: str = "default") -> List[Memory]:
        """List all memories (including SUPERSEDED and DELETED) for inspection/audit."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC;",
                (user_id,),
            )
            return [self._row_to_memory(row) for row in cursor.fetchall()]

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        user_id: str = "default",
    ) -> bool:
        """Update the status of a memory record (e.g. SUPERSEDED or DELETED)."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET status = ?, updated_at = ?
                WHERE id = ? AND user_id = ?;
                """,
                (status.value, now, memory_id, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_content(
        self,
        memory_id: str,
        new_content: str,
        user_id: str = "default",
    ) -> Optional[Memory]:
        """Update the content of an active memory."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET content = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND status = ?;
                """,
                (new_content, now, memory_id, user_id, MemoryStatus.ACTIVE.value),
            )
            conn.commit()
            if cursor.rowcount > 0:
                return self.get(memory_id, user_id)
        return None

    def delete(self, memory_id: str, user_id: str = "default") -> bool:
        """Mark a memory record as DELETED."""
        return self.update_status(memory_id, MemoryStatus.DELETED, user_id)

    def search_relevant(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
    ) -> List[Memory]:
        """
        Deterministic, token-based relevance search across active memories for a user.
        Scores memories based on token overlap, exact substring match, and metadata matching.
        """
        active_memories = self.list_active(user_id=user_id)
        if not active_memories:
            return []

        # Extract tokens from query excluding stop words
        query_cleaned = re.sub(r"[^\w\s]", " ", query.lower())
        query_tokens = {t for t in query_cleaned.split() if t and t not in _STOP_WORDS}
        if not query_tokens:
            return []

        scored_memories = []
        for mem in active_memories:
            mem_text_clean = re.sub(r"[^\w\s]", " ", mem.content.lower())
            mem_tokens = set(mem_text_clean.split())
            
            # 1. Exact overlap count
            overlap = query_tokens.intersection(mem_tokens)
            if not overlap:
                # Also check substring match (e.g. "nexia" in "nexia project")
                score = 0
                for q_tok in query_tokens:
                    if len(q_tok) >= 3 and q_tok in mem_text_clean:
                        score += 1
                if score == 0:
                    continue
            else:
                score = len(overlap) * 2

            # 2. Check metadata keywords or category match
            if mem.type.value in query.lower():
                score += 2

            scored_memories.append((score, mem))

        # Sort descending by score, then by recency
        scored_memories.sort(key=lambda x: (x[0], x[1].updated_at), reverse=True)
        return [mem for score, mem in scored_memories[:limit]]

    def delete_by_pattern(
        self,
        pattern: str,
        user_id: str = "default",
    ) -> int:
        """
        Find active memories matching a pattern or topic and mark them DELETED.
        Returns the count of deleted memories.
        """
        active_memories = self.list_active(user_id=user_id)
        if not active_memories:
            return 0

        pattern_cleaned = re.sub(r"[^\w\s]", " ", pattern.lower()).strip()
        pattern_tokens = {t for t in pattern_cleaned.split() if t and t not in _STOP_WORDS}

        deleted_count = 0
        for mem in active_memories:
            mem_text_clean = re.sub(r"[^\w\s]", " ", mem.content.lower())
            mem_tokens = set(mem_text_clean.split())

            # Check exact substring match or significant token overlap
            matches = False
            if pattern_cleaned and pattern_cleaned in mem_text_clean:
                matches = True
            elif pattern_tokens and pattern_tokens.issubset(mem_tokens):
                matches = True
            elif pattern_tokens and len(pattern_tokens.intersection(mem_tokens)) >= max(1, len(pattern_tokens) - 1):
                matches = True
            elif pattern_cleaned in mem.type.value:
                matches = True

            if matches:
                self.delete(mem.id, user_id=user_id)
                deleted_count += 1

        return deleted_count

    def clear_all(self, user_id: Optional[str] = None) -> None:
        """Clear memories (primarily for testing)."""
        with self._get_connection() as conn:
            if user_id:
                conn.execute("DELETE FROM memories WHERE user_id = ?;", (user_id,))
            else:
                conn.execute("DELETE FROM memories;")
            conn.commit()

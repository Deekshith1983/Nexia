"""
SQLite persistent storage for Nexia Stage 11 User Preferences.
Survives application restarts and maintains multi-user isolation.
"""
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.personalization.models import PreferenceCategory, PreferenceSource, PreferenceStatus, UserPreference

logger = logging.getLogger("nexia.personalization.store")


class PersonalizationStore:
    """
    Persistent SQLite storage engine for structured user preferences.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = settings.preferences_db_path

        path_obj = Path(db_path)
        if not path_obj.is_absolute():
            backend_root = Path(__file__).resolve().parent.parent.parent
            path_obj = backend_root / db_path

        self.db_path = str(path_obj)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prefs_user_status ON user_preferences(user_id, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_prefs_user_cat ON user_preferences(user_id, category, status);"
            )
            conn.commit()

    def save(self, preference: UserPreference) -> UserPreference:
        """Insert or update a preference record."""
        preference.updated_at = time.time()
        payload = preference.model_dump()
        payload_json = json.dumps(payload)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_preferences (
                    id, user_id, category, key, value, source, confidence,
                    status, created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    category = excluded.category,
                    key = excluded.key,
                    value = excluded.value,
                    source = excluded.source,
                    confidence = excluded.confidence,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    payload_json = excluded.payload_json;
                """,
                (
                    preference.id,
                    preference.user_id,
                    preference.category.value,
                    preference.key,
                    preference.value,
                    preference.source.value,
                    preference.confidence,
                    preference.status.value,
                    preference.created_at,
                    preference.updated_at,
                    payload_json,
                ),
            )
            conn.commit()

        logger.debug(
            "[PREF STORE] Saved preference id=%s user=%s cat=%s key=%s val=%s status=%s",
            preference.id, preference.user_id, preference.category.value, preference.key, preference.value, preference.status.value
        )
        return preference

    def get(self, preference_id: str, user_id: str = "default") -> Optional[UserPreference]:
        """Get preference by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM user_preferences WHERE id = ? AND user_id = ?",
                (preference_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["payload_json"])
            return UserPreference.model_validate(data)

    def list_active(
        self, user_id: str = "default", category: Optional[PreferenceCategory] = None
    ) -> List[UserPreference]:
        """List active preferences for a user, optionally filtered by category."""
        query = "SELECT payload_json FROM user_preferences WHERE user_id = ? AND status = 'ACTIVE'"
        params: List[Any] = [user_id]
        if category:
            query += " AND category = ?"
            cat_val = category.value if hasattr(category, "value") else str(category)
            params.append(cat_val)

        query += " ORDER BY updated_at DESC"

        results: List[UserPreference] = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["payload_json"])
                    results.append(UserPreference.model_validate(data))
                except Exception as exc:
                    logger.warning("[PREF STORE] Failed parsing preference row: %s", exc)
        return results

    def list_all(self, user_id: str = "default") -> List[UserPreference]:
        """List all preferences for a user."""
        query = "SELECT payload_json FROM user_preferences WHERE user_id = ? ORDER BY updated_at DESC"
        results: List[UserPreference] = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, (user_id,))
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["payload_json"])
                    results.append(UserPreference.model_validate(data))
                except Exception as exc:
                    logger.warning("[PREF STORE] Failed parsing preference row: %s", exc)
        return results

    def update_status(
        self, preference_id: str, status: PreferenceStatus, user_id: str = "default"
    ) -> bool:
        """Update lifecycle status of a preference."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM user_preferences WHERE id = ? AND user_id = ?",
                (preference_id, user_id),
            )
            row = cursor.fetchone()
            if not row:
                return False
            data = json.loads(row["payload_json"])
            data["status"] = status.value
            data["updated_at"] = time.time()
            payload_json = json.dumps(data)

            conn.execute(
                "UPDATE user_preferences SET status = ?, updated_at = ?, payload_json = ? WHERE id = ? AND user_id = ?",
                (status.value, data["updated_at"], payload_json, preference_id, user_id),
            )
            conn.commit()
            return True

    def delete(self, preference_id: str, user_id: str = "default") -> bool:
        """Mark a preference as DELETED."""
        return self.update_status(preference_id, PreferenceStatus.DELETED, user_id=user_id)

    def delete_by_pattern(self, pattern: str, user_id: str = "default") -> int:
        """Delete active preferences matching a keyword pattern."""
        active = self.list_active(user_id=user_id)
        pat_lower = pattern.lower().strip()
        count = 0
        for p in active:
            if pat_lower in p.key.lower() or pat_lower in p.value.lower() or pat_lower in p.category.value.lower():
                if self.delete(p.id, user_id=user_id):
                    count += 1
        return count

    def clear_all(self, user_id: Optional[str] = None) -> None:
        """Clear all preferences."""
        with self._get_connection() as conn:
            if user_id:
                conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
            else:
                conn.execute("DELETE FROM user_preferences")
            conn.commit()


personalization_store = PersonalizationStore()

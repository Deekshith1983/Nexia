"""
SQLite persistent storage for Nexia Stage 10 ResearchSessions.
Survives application restarts and enforces optimistic concurrency versioning.
"""
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional

from app.config import settings
from app.sessions.models import ResearchSession, SessionStatus

logger = logging.getLogger("nexia.sessions.store")


class StaleSessionError(Exception):
    """Raised when updating a session whose persisted version is newer than the in-memory version."""
    pass


class ResearchSessionStore:
    """
    Persistent SQLite storage engine for long-horizon ResearchSessions.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = "data/research_sessions.db"

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
                CREATE TABLE IF NOT EXISTS research_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conversation_id TEXT,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON research_sessions(user_id, status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_updated ON research_sessions(updated_at);"
            )
            conn.commit()

    def save(self, session: ResearchSession) -> ResearchSession:
        """Insert or update a session with version increment and concurrency check."""
        session.updated_at = time.time()
        payload = session.model_dump()
        payload_json = json.dumps(payload)

        with self._get_connection() as conn:
            # Check existing version if updating
            cursor = conn.execute(
                "SELECT version FROM research_sessions WHERE session_id = ?",
                (session.session_id,),
            )
            row = cursor.fetchone()
            if row:
                existing_version = row["version"]
                if existing_version > session.version:
                    raise StaleSessionError(
                        f"Cannot overwrite session {session.session_id} (version {session.version} < persisted {existing_version})"
                    )
                new_version = session.version + 1
                session.version = new_version
                payload["version"] = new_version
                payload_json = json.dumps(payload)

                conn.execute(
                    """
                    UPDATE research_sessions
                    SET user_id = ?, conversation_id = ?, title = ?, objective = ?, status = ?,
                        version = ?, updated_at = ?, payload_json = ?
                    WHERE session_id = ?
                    """,
                    (
                        session.user_id,
                        session.conversation_id,
                        session.title,
                        session.objective,
                        session.status.value,
                        new_version,
                        session.updated_at,
                        payload_json,
                        session.session_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO research_sessions (
                        session_id, user_id, conversation_id, title, objective, status,
                        version, created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        session.user_id,
                        session.conversation_id,
                        session.title,
                        session.objective,
                        session.status.value,
                        session.version,
                        session.created_at,
                        session.updated_at,
                        payload_json,
                    ),
                )
            conn.commit()

        logger.debug("[SESSION STORE] Saved session %r (version=%d status=%s)", session.session_id, session.version, session.status.value)
        return session

    def get(self, session_id: str) -> Optional[ResearchSession]:
        """Fetch a single session by session_id."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT payload_json FROM research_sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["payload_json"])
            return ResearchSession.model_validate(data)

    def list(
        self,
        user_id: str = "default_user",
        status: Optional[SessionStatus] = None,
        limit: int = 50,
    ) -> List[ResearchSession]:
        """List sessions for a given user, optionally filtered by status."""
        query = "SELECT payload_json FROM research_sessions WHERE user_id = ?"
        params: List[Any] = [user_id]

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        results: List[ResearchSession] = []
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                try:
                    data = json.loads(row["payload_json"])
                    results.append(ResearchSession.model_validate(data))
                except Exception as exc:
                    logger.warning("[SESSION STORE] Failed parsing session row: %s", exc)

        return results

    def delete(self, session_id: str) -> bool:
        """Permanently remove a single session."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM research_sessions WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount > 0


research_session_store = ResearchSessionStore()

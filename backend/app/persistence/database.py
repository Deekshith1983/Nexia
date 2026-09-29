"""
Nexia Centralized Persistence Infrastructure.
Provides robust SQLite connection management, transaction safety, WAL mode, and path sanitization.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, Optional

from app.config import settings
from app.core.errors import PersistenceError


def resolve_safe_path(db_path: str) -> str:
    """Safely resolve relative and absolute database paths within the project directory."""
    path_obj = Path(db_path)
    if not path_obj.is_absolute():
        backend_root = Path(__file__).resolve().parent.parent.parent
        path_obj = (backend_root / db_path).resolve()
    else:
        path_obj = path_obj.resolve()

    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    return str(path_obj)


@contextmanager
def get_db_connection(db_path: str, timeout: float = 15.0) -> Generator[sqlite3.Connection, None, None]:
    """Provide a configured SQLite connection with row factory, foreign keys, and WAL journal."""
    safe_path = resolve_safe_path(db_path)
    try:
        conn = sqlite3.connect(safe_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    except sqlite3.Error as e:
        raise PersistenceError(f"Database connection error on '{safe_path}': {e}") from e
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def transaction(db_path: str, timeout: float = 15.0) -> Generator[sqlite3.Connection, None, None]:
    """Execute operations inside an atomic ACID transaction. Automatically commits on success and rolls back on failure."""
    with get_db_connection(db_path, timeout=timeout) as conn:
        try:
            conn.execute("BEGIN TRANSACTION;")
            yield conn
            conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            if isinstance(e, PersistenceError):
                raise
            raise PersistenceError(f"Database transaction error: {e}") from e


def verify_sqlite_health(db_path: str) -> bool:
    """Verify read and write accessibility of a database file."""
    try:
        with get_db_connection(db_path) as conn:
            conn.execute("SELECT 1;")
        return True
    except Exception:
        return False


def get_persistence_status() -> Dict[str, Any]:
    """Return health status of all registered persistence databases."""
    return {
        "memory_db": verify_sqlite_health(settings.memory_db_path),
        "sessions_db": verify_sqlite_health(settings.sessions_db_path),
        "preferences_db": verify_sqlite_health(settings.preferences_db_path),
    }

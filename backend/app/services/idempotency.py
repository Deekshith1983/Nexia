"""Request Deduplication and Idempotency Management for Nexia."""

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any, Dict, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger("nexia.idempotency")


class IdempotencyState(str, Enum):
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class IdempotencyRecord:
    key: str
    state: IdempotencyState
    response_data: Optional[Dict[str, Any]]
    created_at: float
    future: Optional[asyncio.Future] = None


class IdempotencyService:
    """Manages request idempotency, in-flight deduplication, and completed response replay."""

    _instance: Optional["IdempotencyService"] = None

    def __init__(self, ttl_seconds: Optional[float] = None):
        self.ttl_seconds = ttl_seconds or float(settings.IDEMPOTENCY_TTL_SECONDS)
        self.enabled = settings.IDEMPOTENCY_ENABLED
        self._records: Dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> "IdempotencyService":
        if cls._instance is None:
            cls._instance = IdempotencyService()
        return cls._instance

    async def start_request(self, key: str) -> Tuple[bool, Optional[IdempotencyRecord]]:
        """Register or check an idempotency key.
        
        Returns:
            Tuple[is_new, record]:
            If is_new is True, the caller owns execution.
            If is_new is False, another request owns it (either running or completed).
        """
        if not self.enabled or not key:
            return True, None

        now = time.monotonic()
        async with self._lock:
            # Check existing record
            existing = self._records.get(key)
            if existing:
                # Check if expired
                if (now - existing.created_at) > self.ttl_seconds:
                    self._records.pop(key, None)
                else:
                    return False, existing

            # New request
            loop = asyncio.get_running_loop()
            record = IdempotencyRecord(
                key=key,
                state=IdempotencyState.RUNNING,
                response_data=None,
                created_at=now,
                future=loop.create_future(),
            )
            self._records[key] = record
            return True, record

    async def complete_request(self, key: str, response_data: Dict[str, Any]) -> None:
        """Mark an idempotency record as COMPLETED with cached response payload."""
        if not self.enabled or not key:
            return

        async with self._lock:
            rec = self._records.get(key)
            if rec:
                rec.state = IdempotencyState.COMPLETED
                rec.response_data = response_data
                if rec.future and not rec.future.done():
                    rec.future.set_result(response_data)

    async def fail_request(self, key: str, error: Exception) -> None:
        """Mark an idempotency record as FAILED."""
        if not self.enabled or not key:
            return

        async with self._lock:
            rec = self._records.get(key)
            if rec:
                rec.state = IdempotencyState.FAILED
                if rec.future and not rec.future.done():
                    rec.future.set_exception(error)
                # Remove failed record after short delay so user can retry safely
                self._records.pop(key, None)

    async def cancel_request(self, key: str) -> None:
        """Mark an idempotency record as CANCELLED."""
        if not self.enabled or not key:
            return

        async with self._lock:
            rec = self._records.get(key)
            if rec:
                rec.state = IdempotencyState.CANCELLED
                self._records.pop(key, None)

    async def wait_for_result(self, record: IdempotencyRecord, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Wait for an in-flight duplicate request to complete."""
        if record.state == IdempotencyState.COMPLETED:
            return record.response_data

        if record.future:
            try:
                return await asyncio.wait_for(asyncio.shield(record.future), timeout=timeout)
            except Exception as e:
                logger.warning("Error waiting for in-flight idempotent request %s: %s", record.key, e)
                return None
        return None

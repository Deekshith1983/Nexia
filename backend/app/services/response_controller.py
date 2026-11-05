"""Response Controller: Concurrency management, task cancellation, and request deduplication."""

import asyncio
from datetime import datetime
import logging
from typing import Any, Callable, Coroutine, Dict, Optional
from uuid import UUID, uuid4

from app.core.config import settings

logger = logging.getLogger("nexia.response_controller")


class RequestCancelledError(Exception):
    """Raised when an in-flight LLM generation request is cancelled."""
    pass


class ResponseController:
    """Controls request lifecycles, bounded local concurrency, deduplication, and cancellation."""

    _instance: Optional["ResponseController"] = None

    def __init__(self, max_concurrent: int = 1):
        # Default concurrency = 1 on CPU to prevent CPU thread thrashing and memory spikes
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.in_flight_queries: Dict[str, asyncio.Future] = {}
        self.cancelled_requests: set[str] = set()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        """Lazily initialize semaphore inside the active event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    @classmethod
    def get_instance(cls) -> "ResponseController":
        if cls._instance is None:
            cls._instance = ResponseController(max_concurrent=1)
        return cls._instance

    def register_task(self, request_id: str, task: asyncio.Task) -> None:
        """Register an active generation task for cancellation management."""
        self.active_tasks[request_id] = task

    def unregister_task(self, request_id: str) -> None:
        """Unregister a completed task."""
        self.active_tasks.pop(request_id, None)
        self.cancelled_requests.discard(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """Explicitly cancel an in-flight request."""
        task = self.active_tasks.get(request_id)
        if task and not task.done():
            self.cancelled_requests.add(request_id)
            task.cancel()
            logger.info("Explicitly cancelled generation task for request_id: %s", request_id)
            return True
        return False

    def is_cancelled(self, request_id: str) -> bool:
        """Check if request cancellation has been requested."""
        return request_id in self.cancelled_requests

    async def execute_bounded(
        self,
        request_id: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute inference within bounded concurrency semaphore and timeout."""
        current_task = asyncio.current_task()
        if current_task:
            self.register_task(request_id, current_task)

        try:
            async with self.semaphore:
                if self.is_cancelled(request_id):
                    raise RequestCancelledError(f"Request {request_id} was cancelled before starting")

                if timeout:
                    return await asyncio.wait_for(coro_func(), timeout=timeout)
                return await coro_func()

        except asyncio.CancelledError:
            logger.warning("Generation for request %s was cancelled", request_id)
            raise RequestCancelledError(f"Request {request_id} cancelled")
        finally:
            self.unregister_task(request_id)

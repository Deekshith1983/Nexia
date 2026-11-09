"""Centralized Inference Admission and Concurrency Manager for local Ollama LLM execution."""

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Set
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger("nexia.inference_manager")


class Priority(IntEnum):
    """Execution priority levels (lower numeric value = higher priority)."""
    INTERACTIVE = 1   # User-facing answers and streaming
    BACKGROUND = 2    # Automated title generation and summarization


class RequestCancelledError(Exception):
    """Raised when an in-flight LLM generation request is cancelled."""
    pass


class InferenceQueueFullError(Exception):
    """Raised when the inference admission queue exceeds maximum capacity (HTTP 503)."""
    pass


class InferenceQueueTimeoutError(Exception):
    """Raised when request waiting in inference admission queue exceeds timeout (HTTP 504)."""
    pass


@dataclass(order=True)
class QueuedItem:
    """Item placed in the bounded inference priority queue."""
    priority: int
    enqueued_at: float
    request_id: str = field(compare=False)
    future: asyncio.Future = field(compare=False)
    cancelled: bool = field(default=False, compare=False)


class ResponseController:
    """Centralized inference admission manager.
    
    Enforces strictly bounded local LLM concurrency, queue depth limits,
    priority scheduling, queue timeouts, and task cancellation.
    """

    _instance: Optional["ResponseController"] = None

    def __init__(
        self,
        max_active: Optional[int] = None,
        max_queue: Optional[int] = None,
        queue_timeout: Optional[float] = None,
    ):
        self.max_active = max_active or settings.INFERENCE_MAX_ACTIVE
        self.max_queue = max_queue or settings.INFERENCE_MAX_QUEUE
        self.queue_timeout = queue_timeout or settings.INFERENCE_QUEUE_TIMEOUT

        self._active_count = 0
        self._queue: list[QueuedItem] = []
        self._lock = asyncio.Lock()

        # Cancellation tracking
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.cancelled_requests: Set[str] = set()

        # Operational metrics
        self.metrics = {
            "total_admitted": 0,
            "total_rejected_queue_full": 0,
            "total_timeout_queue": 0,
            "total_cancelled": 0,
            "total_completed": 0,
            "max_queue_depth_seen": 0,
        }

    @classmethod
    def get_instance(cls) -> "ResponseController":
        if cls._instance is None:
            cls._instance = ResponseController()
        return cls._instance

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def active_generations(self) -> int:
        return self._active_count

    def register_task(self, request_id: str, task: asyncio.Task) -> None:
        """Register an active generation task for cancellation management."""
        self.active_tasks[request_id] = task

    def unregister_task(self, request_id: str) -> None:
        """Unregister a completed or cancelled task."""
        self.active_tasks.pop(request_id, None)
        self.cancelled_requests.discard(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """Explicitly cancel an in-flight or queued request."""
        logger.info("Cancellation requested for request_id: %s", request_id)
        self.cancelled_requests.add(request_id)

        # 1. Check if queued: dequeue immediately
        for item in list(self._queue):
            if item.request_id == request_id:
                item.cancelled = True
                try:
                    self._queue.remove(item)
                except ValueError:
                    pass
                if not item.future.done():
                    item.future.set_exception(RequestCancelledError(f"Request {request_id} cancelled while queued"))
                self.metrics["total_cancelled"] += 1
                return True

        # 2. Check if currently executing: cancel active task
        task = self.active_tasks.get(request_id)
        if task and not task.done():
            task.cancel()
            self.metrics["total_cancelled"] += 1
            logger.info("Cancelled active running task for request_id: %s", request_id)
            return True

        return False

    def is_cancelled(self, request_id: str) -> bool:
        """Check if request cancellation has been requested."""
        return request_id in self.cancelled_requests

    async def acquire_slot(
        self,
        request_id: str,
        priority: Priority = Priority.INTERACTIVE,
    ) -> float:
        """Acquire an execution slot for inference, waiting in queue if busy.
        
        Returns:
            queue_wait_ms (float): Milliseconds spent waiting in the queue.
        """
        t_start = time.perf_counter()

        if self.is_cancelled(request_id):
            raise RequestCancelledError(f"Request {request_id} was cancelled before acquisition")

        async with self._lock:
            # Slot available immediately
            if self._active_count < self.max_active:
                self._active_count += 1
                self.metrics["total_admitted"] += 1
                return 0.0

            # Capacity exceeded — evaluate bounded queue
            active_queued = [item for item in self._queue if not item.cancelled]
            if len(active_queued) >= self.max_queue:
                self.metrics["total_rejected_queue_full"] += 1
                logger.warning("Inference queue full (%d/%d). Rejecting request %s with HTTP 503",
                               len(active_queued), self.max_queue, request_id)
                raise InferenceQueueFullError(
                    f"Inference server busy: max queue capacity of {self.max_queue} reached. Please try again later."
                )

            # Create queued ticket
            loop = asyncio.get_running_loop()
            ticket = QueuedItem(
                priority=int(priority),
                enqueued_at=time.monotonic(),
                request_id=request_id,
                future=loop.create_future(),
            )
            # Insert maintaining sorted order: priority asc, enqueued_at asc
            self._queue.append(ticket)
            self._queue.sort(key=lambda x: (x.priority, x.enqueued_at))
            self.metrics["max_queue_depth_seen"] = max(self.metrics["max_queue_depth_seen"], len(self._queue))
            logger.info("Request %s queued at position %d (priority=%s)",
                        request_id, len(self._queue), priority.name)

        # Wait for ticket admission or timeout
        try:
            await asyncio.wait_for(ticket.future, timeout=self.queue_timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                if ticket in self._queue:
                    self._queue.remove(ticket)
            self.metrics["total_timeout_queue"] += 1
            logger.warning("Request %s timed out after %.1fs waiting in inference queue",
                           request_id, self.queue_timeout)
            raise InferenceQueueTimeoutError(
                f"Inference queue timeout: wait exceeded {self.queue_timeout}s without an available slot."
            )
        except asyncio.CancelledError:
            async with self._lock:
                if ticket in self._queue:
                    self._queue.remove(ticket)
            raise RequestCancelledError(f"Request {request_id} cancelled while waiting in queue")

        wait_ms = round((time.perf_counter() - t_start) * 1000, 2)
        self.metrics["total_admitted"] += 1
        return wait_ms

    async def release_slot(self, request_id: str) -> None:
        """Release execution slot and admit the next valid queued request."""
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self.metrics["total_completed"] += 1

            # Admit next non-cancelled queued item
            while self._queue:
                next_item = self._queue.pop(0)
                if not next_item.cancelled and not next_item.future.done():
                    self._active_count += 1
                    next_item.future.set_result(True)
                    logger.debug("Admitted queued request %s from queue", next_item.request_id)
                    break

        self.unregister_task(request_id)

    async def execute_bounded(
        self,
        request_id: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        timeout: Optional[float] = None,
        priority: Priority = Priority.INTERACTIVE,
    ) -> Any:
        """Execute a coroutine within admission queue bounds and timeouts."""
        current_task = asyncio.current_task()
        if current_task:
            self.register_task(request_id, current_task)

        # 1. Acquire admission slot (respecting queue limit & timeout)
        await self.acquire_slot(request_id=request_id, priority=priority)

        try:
            if self.is_cancelled(request_id):
                raise RequestCancelledError(f"Request {request_id} was cancelled before starting execution")

            if timeout:
                return await asyncio.wait_for(coro_func(), timeout=timeout)
            return await coro_func()

        except asyncio.CancelledError:
            logger.warning("Execution for request %s was cancelled", request_id)
            raise RequestCancelledError(f"Request {request_id} cancelled")
        finally:
            await self.release_slot(request_id)

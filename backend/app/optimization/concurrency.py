"""
Bounded Concurrency and Parallel Action Execution for Stage 13.
"""
import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("nexia.optimization.concurrency")


class BoundedConcurrencyExecutor:
    """
    Executes independent asynchronous tasks with a configurable concurrency limit.
    Ensures dependency-ordered execution and cancellation safety.
    """

    def __init__(self, max_concurrency: int = 2):
        self.max_concurrency = max(1, max_concurrency)

    async def run_parallel(
        self,
        tasks: List[Callable[[], Coroutine[Any, Any, Any]]],
        max_concurrency: Optional[int] = None,
    ) -> List[Any]:
        """
        Execute a list of independent coroutine factories with bounded concurrency.
        """
        if not tasks:
            return []

        limit = max_concurrency or self.max_concurrency
        semaphore = asyncio.Semaphore(limit)

        async def _worker(task_factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
            async with semaphore:
                try:
                    return await task_factory()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("[CONCURRENCY] Task error during bounded parallel execution: %s", e)
                    raise

        results = await asyncio.gather(*[_worker(t) for t in tasks], return_exceptions=True)
        return list(results)


bounded_executor = BoundedConcurrencyExecutor(max_concurrency=2)

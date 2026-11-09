"""Operational performance metrics and system diagnostics endpoint."""

from fastapi import APIRouter
from app.core.config import settings
from app.services.cache_service import CacheService
from app.services.rate_limiter import RateLimiterService
from app.services.response_controller import ResponseController

router = APIRouter(tags=["Performance & Diagnostics"])


@router.get("/metrics")
async def get_metrics():
    """Retrieve runtime performance telemetry and diagnostics."""
    rc = ResponseController.get_instance()
    cache = CacheService.get_instance()
    rl = RateLimiterService.get_instance()

    return {
        "status": "healthy",
        "version": settings.VERSION,
        "inference": {
            "model": settings.LLM_MODEL,
            "max_active": rc.max_active,
            "max_queue": rc.max_queue,
            "queue_depth": rc.queue_depth,
            "active_generations": rc.active_generations,
            "total_admitted": rc.metrics["total_admitted"],
            "total_rejected_queue_full": rc.metrics["total_rejected_queue_full"],
            "total_timeout_queue": rc.metrics["total_timeout_queue"],
            "total_cancelled": rc.metrics["total_cancelled"],
            "total_completed": rc.metrics["total_completed"],
            "max_queue_depth_seen": rc.metrics["max_queue_depth_seen"],
        },
        "cache": {
            "enabled": cache.enabled,
            "backend": cache.backend_type,
            "hits": cache.stats["hits"],
            "misses": cache.stats["misses"],
            "hit_ratio": cache.hit_ratio,
            "sets": cache.stats["sets"],
            "deletes": cache.stats["deletes"],
        },
        "rate_limiting": {
            "enabled": rl.enabled,
            "backend": rl.backend_type,
            "answer_capacity": settings.RATE_LIMIT_ANSWER_CAPACITY,
            "answer_refill_seconds": settings.RATE_LIMIT_ANSWER_REFILL_SECONDS,
            "conv_create_capacity": settings.RATE_LIMIT_CONV_CREATE_CAPACITY,
            "conv_read_capacity": settings.RATE_LIMIT_CONV_READ_CAPACITY,
        },
        "database_pool": {
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_recycle": settings.DB_POOL_RECYCLE,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pre_ping": settings.DB_PRE_PING,
        },
    }


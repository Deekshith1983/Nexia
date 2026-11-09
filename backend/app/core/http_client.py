"""Shared Asynchronous HTTP Client Pool for Nexia backend services."""

import logging
from typing import Optional
import httpx

from app.core.config import settings

logger = logging.getLogger("nexia.http_client")

_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Retrieve the global pooled AsyncClient instance.
    
    Lazily initializes if accessed before lifespan startup.
    """
    global _client
    if _client is None or _client.is_closed:
        limits = httpx.Limits(
            max_connections=settings.HTTP_POOL_MAX_CONNECTIONS,
            max_keepalive_connections=settings.HTTP_POOL_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=settings.HTTP_POOL_KEEPALIVE_EXPIRY,
        )
        timeout = httpx.Timeout(
            connect=10.0,
            read=float(settings.LLM_TIMEOUT_SECONDS),
            write=15.0,
            pool=10.0,
        )
        _client = httpx.AsyncClient(limits=limits, timeout=timeout)
        logger.info("Initialized shared HTTP client pool (max_conn=%d, keepalive=%d)",
                    settings.HTTP_POOL_MAX_CONNECTIONS, settings.HTTP_POOL_MAX_KEEPALIVE_CONNECTIONS)
    return _client


async def init_http_client() -> httpx.AsyncClient:
    """Explicitly initialize HTTP client pool during application startup."""
    return get_http_client()


async def close_http_client() -> None:
    """Gracefully close HTTP client pool and all connections on application shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        try:
            await _client.aclose()
            logger.info("Shared HTTP client pool gracefully closed.")
        except Exception as e:
            logger.warning("Error closing HTTP client pool: %s", e)
        finally:
            _client = None


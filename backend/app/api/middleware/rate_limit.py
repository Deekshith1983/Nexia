"""FastAPI rate limiting dependencies and exception handling."""

import logging
from typing import Callable
from fastapi import HTTPException, Request, Response, status

from app.services.rate_limiter import RateLimiterService

logger = logging.getLogger("nexia.rate_limit_middleware")


def get_client_ip(request: Request) -> str:
    """Extract canonical client IP from headers or connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


def rate_limit(policy_name: str, cost: int = 1) -> Callable:
    """FastAPI dependency to enforce rate limits per endpoint."""

    async def _dependency(request: Request, response: Response):
        limiter = RateLimiterService.get_instance()
        client_ip = get_client_ip(request)

        result = await limiter.check(policy_name=policy_name, client_ip=client_ip, cost=cost)

        # Set standard rate-limiting headers
        response.headers["X-RateLimit-Limit"] = str(result.capacity)
        response.headers["X-RateLimit-Remaining"] = str(max(0, int(result.remaining_tokens)))

        if not result.allowed:
            retry_seconds = max(1, int(result.retry_after_seconds + 0.99))
            logger.warning("Rate limit exceeded for %s on policy '%s'. Retry after %ss",
                           client_ip, policy_name, retry_seconds)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Please wait {retry_seconds} seconds before retrying.",
                    "retry_after_seconds": result.retry_after_seconds,
                },
                headers={"Retry-After": str(retry_seconds)},
            )

    return _dependency


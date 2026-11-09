"""Correlation ID Middleware for end-to-end request tracing and structured logging."""

import logging
import time
from uuid import uuid4
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import settings

logger = logging.getLogger("nexia.correlation")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures every request has a correlation ID in state and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        header_name = settings.CORRELATION_HEADER_NAME
        request_id = (
            request.headers.get(header_name)
            or request.headers.get("Idempotency-Key")
            or str(uuid4())
        )
        request.state.request_id = request_id

        t_start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 2)

        response.headers[header_name] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        if not request.url.path.endswith("/health"):
            logger.info("[%s] %s %s - %d (%.2fms)",
                        request_id, request.method, request.url.path, response.status_code, duration_ms)

        return response


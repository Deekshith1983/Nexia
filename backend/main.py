"""
Nexia API Core — Production Application Entrypoint.
Orchestrates FastAPI routing, correlation middleware, lifecycle management, and error handling.
"""
from contextlib import asynccontextmanager
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.core.errors import NexiaError
from app.core.logging import current_request_id, get_logger, setup_logging
from app.core.nexia_core import nexia_core
from app.persistence.database import get_persistence_status
from app.providers import get_knowledge_provider, get_model_provider
from app.routes import chat, memory

# Initialize structured logging subsystem
setup_logging(settings.logging_level)
logger = get_logger("app")

_app_start_time = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager for graceful startup and shutdown."""
    logger.info(f"Starting Nexia API Core (env={settings.app_env}, model={settings.ollama_model})...")
    # Verify persistence accessibility on startup
    p_status = get_persistence_status()
    logger.info(f"Persistence status initialized: {p_status}")

    yield

    logger.info("Shutting down Nexia API Core gracefully...")


app = FastAPI(
    title="Nexia Production API Core",
    description="Backend API for Nexia production-grade grounded conversational agent",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Conversation-Id", "X-Nexia-Sources", "X-Request-Id"],
)


@app.middleware("http")
async def correlation_and_timing_middleware(request: Request, call_next) -> Response:
    """Assigns or propagates request correlation ID and records request duration."""
    req_id = request.headers.get("X-Request-Id") or f"req_{uuid.uuid4().hex[:10]}"
    token = current_request_id.set(req_id)
    start_time = time.monotonic()

    try:
        response: Response = await call_next(request)
        duration_ms = (time.monotonic() - start_time) * 1000.0
        response.headers["X-Request-Id"] = req_id
        if request.url.path not in ("/api/live", "/api/health"):
            logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
        return response
    except Exception as exc:
        duration_ms = (time.monotonic() - start_time) * 1000.0
        logger.error(f"{request.method} {request.url.path} unhandled error ({duration_ms:.1f}ms): {exc}")
        raise
    finally:
        current_request_id.reset(token)


# ── Structured Global Error Handlers ──────────────────────────────────────────

@app.exception_handler(NexiaError)
async def nexia_error_handler(request: Request, exc: NexiaError) -> JSONResponse:
    """Handle structured Nexia domain errors with safe public JSON format."""
    req_id = current_request_id.get() or "unknown"
    logger.warning(f"Domain error [{exc.code}]: {exc.message} (status={exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
                "request_id": req_id,
            }
        },
        headers={"X-Request-Id": req_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request payload validation errors safely."""
    req_id = current_request_id.get() or "unknown"
    logger.warning(f"Request validation failed: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request schema or payload.",
                "details": {"validation_errors": exc.errors()},
                "request_id": req_id,
            }
        },
        headers={"X-Request-Id": req_id},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle standard FastAPI HTTPExceptions."""
    req_id = current_request_id.get() or "unknown"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": str(exc.detail),
                "request_id": req_id,
            }
        },
        headers={"X-Request-Id": req_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unexpected internal exceptions (prevents internal stack leaks)."""
    req_id = current_request_id.get() or "unknown"
    logger.error(f"Internal server error: {exc}", exc_info=settings.debug)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. Please try again later.",
                "request_id": req_id,
            }
        },
        headers={"X-Request-Id": req_id},
    )


# ── Route Registration ────────────────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(memory.router)


# ── System Health & Readiness Schemas & Endpoints ─────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    llm_target: str
    ollama_reachable: bool
    knowledge_provider: str
    persistence: Dict[str, bool]
    uptime_seconds: float


@app.get("/")
def root():
    return {"message": "Welcome to Nexia Production Agent Core API", "version": "1.0.0"}


@app.get("/api/live")
def liveness_probe():
    """Liveness probe for process monitoring."""
    return {"status": "alive"}


@app.get("/api/ready")
async def readiness_probe():
    """Readiness probe checking critical dependencies."""
    p_status = get_persistence_status()
    all_dbs_ok = all(p_status.values())
    model_ok = await get_model_provider().health()

    if not all_dbs_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persistence subsystem not ready.",
        )
    return {
        "status": "ready",
        "model_available": model_ok,
        "persistence_ready": all_dbs_ok,
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    model_online = await get_model_provider().health()
    p_status = get_persistence_status()
    k_provider = get_knowledge_provider()

    return HealthResponse(
        status="online" if model_online else "degraded",
        version="1.0.0",
        llm_target=f"Gemma 3 4B ({settings.ollama_model})",
        ollama_reachable=model_online,
        knowledge_provider=k_provider.get_provider_name(),
        persistence=p_status,
        uptime_seconds=round(time.monotonic() - _app_start_time, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=settings.debug)

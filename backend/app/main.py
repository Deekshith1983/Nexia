"""Nexia Backend - FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.api.middleware.correlation import CorrelationIdMiddleware
from app.api.routes import health, answer, conversations, metrics
from app.core.config import settings
from app.core.http_client import init_http_client, close_http_client
from app.db.connection import dispose_engine
from app.db.migrations import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nexia")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle hooks."""
    try:
        await init_db()
        logger.info("Nexia Stage 7 PostgreSQL and vector schema initialized.")
    except Exception as e:
        logger.warning("Database schema init error during startup: %s", e)

    # Initialize shared Async HTTP client pool
    try:
        await init_http_client()
    except Exception as e:
        logger.warning("HTTP client pool startup error: %s", e)

    yield

    # Graceful shutdown
    await close_http_client()
    await dispose_engine()
    logger.info("Nexia application shutdown completed.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Nexia — Intelligent Knowledge Retrieval Backend",
    lifespan=lifespan,
)

# Correlation ID and Request Tracing Middleware
app.add_middleware(CorrelationIdMiddleware)

# Response Compression for compressible payloads >= 1KB (does not buffer SSE streams)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enable CORS for local frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Approved API v1 routers for frontend integration
app.include_router(health.router)
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(metrics.router, prefix=settings.API_V1_PREFIX)
app.include_router(conversations.router, prefix=settings.API_V1_PREFIX)
app.include_router(answer.router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["Index"])
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "health": f"{settings.API_V1_PREFIX}/health",
            "conversations": f"{settings.API_V1_PREFIX}/conversations",
            "answer": f"{settings.API_V1_PREFIX}/retrieval/answer",
            "answer_stream": f"{settings.API_V1_PREFIX}/retrieval/answer/stream",
            "cancel": f"{settings.API_V1_PREFIX}/retrieval/cancel/{{request_id}}",
        },
    }



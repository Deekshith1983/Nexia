"""Nexia Backend - FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from app.api.routes import health, retrieval, relationships, evidence, answer, conversations
from app.core.config import settings
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
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Nexia — Intelligent Knowledge Retrieval Backend",
    lifespan=lifespan,
)

# Include API v1 router
app.include_router(health.router)
app.include_router(retrieval.router, prefix=settings.API_V1_PREFIX)
app.include_router(relationships.router, prefix=settings.API_V1_PREFIX)
app.include_router(evidence.router, prefix=settings.API_V1_PREFIX)
app.include_router(answer.router, prefix=settings.API_V1_PREFIX)
app.include_router(conversations.router, prefix=settings.API_V1_PREFIX)

# Also include at root level for direct access
app.include_router(retrieval.router)
app.include_router(relationships.router)
app.include_router(evidence.router)
app.include_router(answer.router)
app.include_router(conversations.router)


@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
        "endpoints": {
            "search": f"{settings.API_V1_PREFIX}/retrieval/search",
            "document": f"{settings.API_V1_PREFIX}/retrieval/document",
            "passages": f"{settings.API_V1_PREFIX}/retrieval/passages",
            "multi_document": f"{settings.API_V1_PREFIX}/retrieval/multi-document",
            "relationships": f"{settings.API_V1_PREFIX}/retrieval/relationships",
            "evidence": f"{settings.API_V1_PREFIX}/retrieval/evidence",
            "answer": f"{settings.API_V1_PREFIX}/retrieval/answer",
        },
    }



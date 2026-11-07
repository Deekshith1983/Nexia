"""Nexia Backend - FastAPI Application Entrypoint."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, answer, conversations
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



"""Nexia Backend - FastAPI Application Entrypoint."""

from fastapi import FastAPI
from app.api.routes import health, retrieve
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Nexia — Intelligent Knowledge Retrieval Backend",
)

# Include routes
app.include_router(health.router)
app.include_router(retrieve.router)


@app.get("/")
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": "/health",
    }

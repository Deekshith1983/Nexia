"""Nexia Backend - FastAPI Application Entrypoint."""

from fastapi import FastAPI
from app.api.routes import health, retrieval
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Nexia — Intelligent Knowledge Retrieval Backend",
)

# Include API v1 router
app.include_router(health.router)
app.include_router(retrieval.router, prefix=settings.API_V1_PREFIX)
# Also include at root level for direct access (/health, /retrieval/document, /retrieve/document)
app.include_router(retrieval.router)


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
        },
    }

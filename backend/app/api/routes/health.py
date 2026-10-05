"""Health check endpoint router."""

from fastapi import APIRouter
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health status of the Nexia backend service."""
    return HealthResponse(status="ok")

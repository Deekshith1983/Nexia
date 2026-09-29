"""
Nexia Chat API Endpoint.
Handles conversational turns, streaming execution, source extraction, and client disconnects.
"""
import asyncio
import json
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent import agent_runtime
from app.config import settings
from app.core.errors import CancellationError, ModelTimeoutError, ModelUnavailableError
from app.core.logging import current_request_id, get_logger
from app.providers import get_model_provider

logger = get_logger("chat_endpoint")
router = APIRouter(prefix="/api", tags=["Chat"])

# ── Startup / Runtime Model Health Cache ───────────────────────────────────────
_model_healthy: Optional[bool] = None
_model_last_check: float = 0.0
_HEALTH_CACHE_TTL = 30.0  # seconds


async def _is_model_healthy() -> bool:
    """Return cached model health, refreshing at most once per TTL."""
    global _model_healthy, _model_last_check
    now = time.monotonic()
    if _model_healthy is None or (now - _model_last_check) > _HEALTH_CACHE_TTL:
        _model_healthy = await get_model_provider().health()
        _model_last_check = now
    return _model_healthy


def _invalidate_health_cache() -> None:
    """Force a fresh health check on the next call."""
    global _model_healthy, _model_last_check
    _model_healthy = None
    _model_last_check = 0.0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User prompt or question")
    conversation_id: Optional[str] = Field(None, max_length=128, description="Conversation session identifier")
    user_id: Optional[str] = Field("default", max_length=128, description="User identifier for isolation")


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty.",
        )

    # Use cached model health — avoids an expensive HTTP check before every message
    is_healthy = await _is_model_healthy()
    if not is_healthy:
        _invalidate_health_cache()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Nexia LLM infrastructure ({settings.ollama_base_url}). Please ensure Ollama is running.",
        )

    req_id = current_request_id.get() or "chat_req"

    # Process request through the Nexia Agent Runtime
    try:
        result, active_cid = await agent_runtime.run(
            user_message=user_message,
            conversation_id=request.conversation_id,
            user_id=request.user_id or "default",
        )
    except Exception as e:
        logger.error(f"Failed to execute agent runtime: {e}")
        raise

    stream_gen = result.stream
    sources = result.sources
    sources_header_val = json.dumps([s.model_dump() for s in sources]) if sources else ""

    async def event_generator():
        try:
            async for token in stream_gen:
                if await raw_request.is_disconnected():
                    logger.info("Client disconnected. Cancelling stream generation.")
                    break
                yield token
        except asyncio.CancelledError:
            logger.info("Stream cancelled by client disconnect.")
        except ModelUnavailableError as e:
            _invalidate_health_cache()
            yield f"\n[ERROR: {str(e)}]"
        except ModelTimeoutError as e:
            yield f"\n[ERROR: {str(e)}]"
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            yield f"\n[ERROR: Unexpected error during streaming - {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "X-Conversation-Id": active_cid,
            "X-Nexia-Sources": sources_header_val,
            "X-Request-Id": req_id,
        },
    )

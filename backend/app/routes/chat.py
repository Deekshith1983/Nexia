import asyncio
import json
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.agent import agent_runtime
from app.core.llm_interface import LLMOfflineError, LLMServiceError
from app.config import settings

router = APIRouter(prefix="/api", tags=["Chat"])

# ── Startup health cache ──────────────────────────────────────────────────────
# We cache Ollama reachability at startup (and re-validate on explicit /api/health
# requests) rather than calling /api/tags before every single chat request.
# The chat endpoint uses the cached value and only returns 503 if Ollama was
# already known to be unreachable.  A fresh health-check hit will reset the cache.
_ollama_healthy: Optional[bool] = None  # None = not yet checked
_ollama_last_check: float = 0.0
_HEALTH_CACHE_TTL = 30.0  # seconds; re-check at most once per 30s on chat path


async def _is_ollama_healthy() -> bool:
    """Return cached Ollama health, refreshing at most once per TTL."""
    global _ollama_healthy, _ollama_last_check
    now = time.monotonic()
    if _ollama_healthy is None or (now - _ollama_last_check) > _HEALTH_CACHE_TTL:
        from app.core.nexia_core import nexia_core
        _ollama_healthy = await nexia_core.check_health()
        _ollama_last_check = now
    return _ollama_healthy


def _invalidate_health_cache() -> None:
    """Force a fresh health check on the next call."""
    global _ollama_healthy, _ollama_last_check
    _ollama_healthy = None
    _ollama_last_check = 0.0


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "default"


@router.post("/chat")
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )

    # Use cached health — avoids a /api/tags round-trip on every message
    core_online = await _is_ollama_healthy()
    if not core_online:
        _invalidate_health_cache()  # force re-check next time
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Nexia LLM infrastructure at {settings.ollama_base_url}. Please ensure Ollama is running."
        )

    # Process request through the Nexia Agent Runtime
    result, active_cid = await agent_runtime.run(
        user_message=user_message,
        conversation_id=request.conversation_id,
        user_id=request.user_id or "default",
    )
    stream_gen = result.stream
    sources = result.sources

    sources_header_val = json.dumps([s.model_dump() for s in sources]) if sources else ""

    async def event_generator():
        try:
            async for token in stream_gen:
                if await raw_request.is_disconnected():
                    print("[INFO] Client disconnected. Stopping stream generator.")
                    break
                yield token
        except asyncio.CancelledError:
            print("[INFO] Event generator cancelled by client disconnect.")
        except LLMOfflineError as e:
            _invalidate_health_cache()
            yield f"\n[ERROR: {str(e)}]"
        except LLMServiceError as e:
            yield f"\n[ERROR: {str(e)}]"
        except Exception as e:
            yield f"\n[ERROR: Unexpected error during Nexia core streaming - {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={
            "X-Conversation-Id": active_cid,
            "X-Nexia-Sources": sources_header_val,
        }
    )

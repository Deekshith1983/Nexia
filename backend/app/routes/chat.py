import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.core.nexia_core import nexia_core
from app.core.llm_interface import LLMOfflineError, LLMServiceError
from app.config import settings

router = APIRouter(prefix="/api", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )

    # Verify Nexia Core & LLM service reachability before streaming
    core_online = await nexia_core.check_health()
    if not core_online:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Nexia LLM infrastructure at {settings.ollama_base_url}. Please ensure Ollama is running."
        )

    # Process request through Nexia Core & Conversation Manager
    stream_gen, active_cid = await nexia_core.process_chat_stream(
        user_message=user_message,
        conversation_id=request.conversation_id
    )

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
            yield f"\n[ERROR: {str(e)}]"
        except LLMServiceError as e:
            yield f"\n[ERROR: {str(e)}]"
        except Exception as e:
            yield f"\n[ERROR: Unexpected error during Nexia core streaming - {str(e)}]"

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"X-Conversation-Id": active_cid}
    )

import asyncio
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.ollama_service import ollama_service, OllamaOfflineError, OllamaServiceError
from app.config import settings

router = APIRouter(prefix="/api", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_endpoint(request: ChatRequest, raw_request: Request):
    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )

    # First verify Ollama is reachable before beginning stream response
    ollama_online = await ollama_service.check_health()
    if not ollama_online:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot connect to Ollama server at {settings.ollama_base_url}. Please ensure Ollama is running."
        )

    async def event_generator():
        try:
            async for token in ollama_service.stream_response(user_message):
                if await raw_request.is_disconnected():
                    print("[INFO] Client disconnected. Stopping stream generator.")
                    break
                yield token
        except asyncio.CancelledError:
            print("[INFO] Event generator cancelled by client disconnect.")
        except OllamaOfflineError as e:
            yield f"\n[ERROR: {str(e)}]"
        except OllamaServiceError as e:
            yield f"\n[ERROR: {str(e)}]"
        except Exception as e:
            yield f"\n[ERROR: Unexpected error during streaming - {str(e)}]"

    return StreamingResponse(event_generator(), media_type="text/plain")

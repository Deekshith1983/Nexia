import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.routes import chat, memory
from app.core.nexia_core import nexia_core

# Configure Nexia Core logger — traces pipeline timing to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Core pipeline trace at INFO level (always shown)
logging.getLogger("nexia.core").setLevel(logging.INFO)
# Per-stage structured debug traces (shown at DEBUG level)
# Set to logging.DEBUG to see full pipeline data flow for each request.
for _debug_logger in ("nexia.pipeline", "nexia.qu", "nexia.ctx", "nexia.wiki", "nexia.memory"):
    logging.getLogger(_debug_logger).setLevel(logging.DEBUG)

app = FastAPI(
    title="Nexia API Core",
    description="Backend API for Nexia grounded conversational agent",
    version="0.2.2",
)

# Enable CORS for local development (React frontend on localhost:5173 / localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose custom response headers so the browser can read them via fetch()
    # Without this, response.headers.get('X-Conversation-Id') returns null,
    # causing the frontend to never persist the conversation_id across turns.
    expose_headers=["X-Conversation-Id", "X-Nexia-Sources"],
)

# Include routes
app.include_router(chat.router)
app.include_router(memory.router)


class StatusResponse(BaseModel):
    status: str
    version: str
    llm_target: str
    ollama_reachable: bool


@app.get("/")
def root():
    return {"message": "Welcome to Nexia Agent Core API"}


@app.get("/api/health", response_model=StatusResponse)
async def health_check():
    core_online = await nexia_core.check_health()
    return StatusResponse(
        status="online",
        version="0.2.1",
        llm_target=f"Gemma 3 4B ({settings.ollama_model})",
        ollama_reachable=core_online,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

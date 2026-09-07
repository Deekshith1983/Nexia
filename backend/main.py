import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.routes import chat
from app.services.ollama_service import ollama_service

app = FastAPI(
    title="Nexia API Core",
    description="Backend API for Nexia grounded conversational agent",
    version="0.2.0",
)

# Enable CORS for local development (React frontend on localhost:5173 / localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat.router)


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
    ollama_online = await ollama_service.check_health()
    return StatusResponse(
        status="online",
        version="0.2.0",
        llm_target=f"Gemma 3 4B ({settings.ollama_model})",
        ollama_reachable=ollama_online,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

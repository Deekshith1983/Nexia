import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Nexia API Core",
    description="Backend API for Nexia grounded conversational agent",
    version="0.1.0",
)

# Enable CORS for local development (React frontend on localhost:5173 / localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StatusResponse(BaseModel):
    status: str
    version: str
    llm_target: str
    knowledge_source: str


@app.get("/")
def root():
    return {"message": "Welcome to Nexia Agent Core API"}


@app.get("/api/health", response_model=StatusResponse)
def health_check():
    return StatusResponse(
        status="online",
        version="0.1.0",
        llm_target="Qwen3 8B (via Ollama)",
        knowledge_source="Wikipedia REST API",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

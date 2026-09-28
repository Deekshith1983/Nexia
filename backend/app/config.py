import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "data/nexia_memory.db")

    class Config:
        env_file = ".env"

settings = Settings()

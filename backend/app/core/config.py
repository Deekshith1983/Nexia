"""Application configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Detect .env file location (check backend/.env, then current working dir/.env)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = _BASE_DIR / ".env" if (_BASE_DIR / ".env").exists() else ".env"


class Settings(BaseSettings):
    # Core API Settings
    PROJECT_NAME: str = "Nexia"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "0.1.0"

    # PostgreSQL Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:diki123456@localhost:5432/nexiav01"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_CONNECT_TIMEOUT: int = 10

    # Ollama Services
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embedding Settings (nomic-embed-text)
    EMBEDDING_MODEL: str = "nomic-embed-text"
    EMBEDDING_DIMENSIONS: int = 768
    EMBEDDING_BATCH_SIZE: int = 32

    # LLM Settings (gemma3:4b CPU Optimized)
    LLM_MODEL: str = "gemma3:4b"
    LLM_TIMEOUT_SECONDS: int = 360
    LLM_MAX_EVIDENCE_PASSAGES: int = 3
    LLM_TEMPERATURE: float = 0.1
    LLM_NUM_THREADS: int = 6       # Matches Ryzen 5 physical CPU cores
    LLM_NUM_CTX: int = 6144        # Expanded to hold list-type evidence passages fully
    LLM_NUM_PREDICT: int = 768     # Allow complete list-style answers without truncation
    LLM_KEEP_ALIVE: str = "30m"    # Keep model resident in RAM

    # Retrieval Tuning
    VECTOR_SEARCH_CANDIDATES: int = 20
    KEYWORD_SEARCH_CANDIDATES: int = 20
    HYBRID_TOP_K: int = 10

    # Cache Policy
    CACHE_FRESHNESS_DAYS: int = 7
    CACHE_STALE_ON_FAILURE: str = "use_existing"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

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
    LLM_MAX_EVIDENCE_PASSAGES: int = 9
    LLM_TEMPERATURE: float = 0.1
    LLM_NUM_THREADS: int = 6       # Matches Ryzen 5 physical CPU cores
    LLM_NUM_CTX: int = 6144        # Expanded to hold list-type evidence passages fully
    LLM_NUM_PREDICT: int = 768     # Allow complete list-style answers without truncation
    LLM_KEEP_ALIVE: str = "30m"    # Keep model resident in RAM

    # Retrieval Tuning
    VECTOR_SEARCH_CANDIDATES: int = 20
    KEYWORD_SEARCH_CANDIDATES: int = 20
    HYBRID_TOP_K: int = 10

    # Adaptive NLP & Information-Dense Query Settings
    ADAPTIVE_RETRIEVAL_ENABLED: bool = True
    MAX_SUBQUERIES_PER_COMPLEX_QUERY: int = 4
    COMPLEX_QUERY_TOP_K_PER_SUBQUERY: int = 4

    # Cache Policy
    CACHE_FRESHNESS_DAYS: int = 7
    CACHE_STALE_ON_FAILURE: str = "use_existing"

    # =========================================================================
    # Performance Optimization & Scalability Settings
    # =========================================================================

    # Rate Limiting (Token Bucket)
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BACKEND: str = "memory"  # "memory" | "redis"
    RATE_LIMIT_REDIS_URL: str = "redis://localhost:6379/0"
    RATE_LIMIT_FAIL_OPEN: bool = True
    RATE_LIMIT_ANSWER_CAPACITY: int = 2
    RATE_LIMIT_ANSWER_REFILL_SECONDS: float = 30.0
    RATE_LIMIT_CONV_CREATE_CAPACITY: int = 10
    RATE_LIMIT_CONV_CREATE_REFILL_SECONDS: float = 2.0
    RATE_LIMIT_CONV_READ_CAPACITY: int = 30
    RATE_LIMIT_CONV_READ_REFILL_SECONDS: float = 1.0

    # Inference Admission & Bounded Queue
    INFERENCE_MAX_ACTIVE: int = 1
    INFERENCE_MAX_QUEUE: int = 3
    INFERENCE_QUEUE_TIMEOUT: float = 30.0

    # PostgreSQL Connection Pool Tuning
    DB_POOL_RECYCLE: int = 1800
    DB_POOL_TIMEOUT: float = 15.0
    DB_PRE_PING: bool = True

    # Tiered Caching (InMemory / Redis)
    CACHE_ENABLED: bool = True
    CACHE_BACKEND: str = "memory"  # "memory" | "redis"
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_DEFAULT_TTL_SECONDS: int = 3600
    CACHE_WIKI_METADATA_TTL_SECONDS: int = 86400  # 24 hours
    CACHE_RETRIEVAL_TTL_SECONDS: int = 3600       # 1 hour
    CACHE_CONVERSATIONS_TTL_SECONDS: int = 300     # 5 minutes

    # Request Deduplication & Idempotency
    IDEMPOTENCY_ENABLED: bool = True
    IDEMPOTENCY_TTL_SECONDS: int = 300            # 5 minutes

    # Shared Async HTTP Client Pool
    HTTP_POOL_MAX_CONNECTIONS: int = 50
    HTTP_POOL_MAX_KEEPALIVE_CONNECTIONS: int = 10
    HTTP_POOL_KEEPALIVE_EXPIRY: float = 30.0

    # Observability & Metrics
    METRICS_ENABLED: bool = True
    CORRELATION_HEADER_NAME: str = "X-Request-ID"

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

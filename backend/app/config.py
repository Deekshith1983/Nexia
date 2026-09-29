"""
Nexia Centralized Production Configuration Architecture.
Validates all runtime boundaries, timeouts, limits, provider endpoints, and storage paths.
"""
import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Production-grade validated configuration for Nexia."""

    # Application Environment
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Model Provider Settings
    model_provider: str = os.getenv("MODEL_PROVIDER", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))
    request_timeout: float = float(os.getenv("REQUEST_TIMEOUT", "120.0"))

    # Knowledge Provider Settings
    knowledge_provider: str = os.getenv("KNOWLEDGE_PROVIDER", "wikipedia")
    wikipedia_timeout: float = float(os.getenv("WIKIPEDIA_TIMEOUT", "15.0"))
    wikipedia_max_retries: int = int(os.getenv("WIKIPEDIA_MAX_RETRIES", "2"))

    # Research & Reasoning Governance Limits
    research_action_budget: int = int(os.getenv("RESEARCH_ACTION_BUDGET", "8"))
    navigation_depth: int = int(os.getenv("NAVIGATION_DEPTH", "3"))
    decomposition_limit: int = int(os.getenv("DECOMPOSITION_LIMIT", "5"))
    reflection_repair_limit: int = int(os.getenv("REFLECTION_REPAIR_LIMIT", "2"))
    concurrency_limit: int = int(os.getenv("CONCURRENCY_LIMIT", "4"))

    # Cache & Optimization Configuration
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "500"))
    cache_ttl_seconds: float = float(os.getenv("CACHE_TTL_SECONDS", "3600.0"))

    # Centralized Persistence Paths
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "data/nexia_memory.db")
    sessions_db_path: str = os.getenv("SESSIONS_DB_PATH", "data/research_sessions.db")
    preferences_db_path: str = os.getenv("PREFERENCES_DB_PATH", "data/user_preferences.db")
    baselines_path: str = os.getenv("BASELINES_PATH", "data/evaluation_baselines.json")

    # Logging & Telemetry
    logging_level: str = os.getenv("LOGGING_LEVEL", "INFO")
    cors_origins: Union[str, List[str]] = os.getenv("CORS_ORIGINS", "*")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, v: str) -> str:
        allowed = ("development", "staging", "production", "test")
        v_clean = v.strip().lower()
        if v_clean not in allowed:
            raise ValueError(f"Invalid app_env '{v}'. Must be one of: {allowed}")
        return v_clean

    @field_validator("model_provider")
    @classmethod
    def validate_model_provider(cls, v: str) -> str:
        allowed = ("ollama", "mock", "test")
        v_clean = v.strip().lower()
        if v_clean not in allowed:
            raise ValueError(f"Invalid model_provider '{v}'. Must be one of: {allowed}")
        return v_clean

    @field_validator("ollama_timeout", "request_timeout", "wikipedia_timeout", "cache_ttl_seconds")
    @classmethod
    def validate_positive_float(cls, v: float, info) -> float:
        if v <= 0:
            raise ValueError(f"Configuration parameter '{info.field_name}' must be positive, got {v}")
        return v

    @field_validator("research_action_budget", "navigation_depth", "decomposition_limit", "concurrency_limit", "cache_max_size")
    @classmethod
    def validate_positive_int(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError(f"Configuration parameter '{info.field_name}' must be greater than 0, got {v}")
        return v

    @field_validator("reflection_repair_limit")
    @classmethod
    def validate_non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"reflection_repair_limit must be >= 0, got {v}")
        return v

    @field_validator("logging_level")
    @classmethod
    def validate_logging_level(cls, v: str) -> str:
        allowed = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        v_clean = v.strip().upper()
        if v_clean not in allowed:
            raise ValueError(f"Invalid logging_level '{v}'. Must be one of: {allowed}")
        return v_clean

    def get_cors_origins(self) -> List[str]:
        if isinstance(self.cors_origins, list):
            return self.cors_origins
        if isinstance(self.cors_origins, str):
            if self.cors_origins == "*":
                return ["*"]
            return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        return ["*"]


# Singleton application configuration instance
settings = Settings()

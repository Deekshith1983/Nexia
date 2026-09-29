"""
Nexia Language Model Provider Boundary.
Abstracts all LLM interactions behind a provider interface.
"""
from abc import ABC, abstractmethod
import asyncio
from typing import Any, AsyncGenerator, Dict, Optional

from app.config import settings
from app.core.errors import ModelTimeoutError, ModelUnavailableError
from app.core.llm_interface import LLMOfflineError, LLMServiceError


class LanguageModelProvider(ABC):
    """Abstract interface defining the language model provider boundary."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate complete text response from the model."""
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream response tokens asynchronously."""
        pass

    @abstractmethod
    async def health(self) -> bool:
        """Check if model provider infrastructure is healthy and reachable."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return configured model identifier."""
        pass


class OllamaModelProvider(LanguageModelProvider):
    """Production Ollama/Gemma language model provider implementation."""

    def __init__(self):
        from app.services.ollama_service import ollama_service
        self._service = ollama_service

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        try:
            return await self._service.generate_response(prompt)
        except LLMOfflineError as e:
            raise ModelUnavailableError(f"Ollama provider offline: {e}") from e
        except asyncio.TimeoutError as e:
            raise ModelTimeoutError(f"Ollama generation timed out: {e}") from e
        except LLMServiceError as e:
            raise ModelUnavailableError(f"Ollama service error: {e}") from e

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        try:
            async for chunk in self._service.stream_response(prompt):
                yield chunk
        except LLMOfflineError as e:
            raise ModelUnavailableError(f"Ollama provider offline: {e}") from e
        except asyncio.TimeoutError as e:
            raise ModelTimeoutError(f"Ollama stream timed out: {e}") from e
        except LLMServiceError as e:
            raise ModelUnavailableError(f"Ollama stream error: {e}") from e

    async def health(self) -> bool:
        try:
            return await self._service.check_health()
        except Exception:
            return False

    def get_model_name(self) -> str:
        return settings.ollama_model


class MockModelProvider(LanguageModelProvider):
    """Deterministic mock model provider for tests and benchmark isolation."""

    def __init__(self, canned_response: str = "Mock generation result."):
        self.canned_response = canned_response
        self._healthy = True

    def set_response(self, response: str) -> None:
        self.canned_response = response

    def set_healthy(self, healthy: bool) -> None:
        self._healthy = healthy

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        if not self._healthy:
            raise ModelUnavailableError("Mock model provider configured as unavailable.")
        return self.canned_response

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        if not self._healthy:
            raise ModelUnavailableError("Mock model provider configured as unavailable.")
        words = self.canned_response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    async def health(self) -> bool:
        return self._healthy

    def get_model_name(self) -> str:
        return "mock-gemma"


# Factory provider registry
_active_model_provider: Optional[LanguageModelProvider] = None


def get_model_provider() -> LanguageModelProvider:
    """Return the configured singleton model provider."""
    global _active_model_provider
    if _active_model_provider is None:
        if settings.model_provider == "mock":
            _active_model_provider = MockModelProvider()
        else:
            _active_model_provider = OllamaModelProvider()
    return _active_model_provider


def set_model_provider(provider: LanguageModelProvider) -> None:
    """Override active model provider (used in testing and evaluation)."""
    global _active_model_provider
    _active_model_provider = provider

from abc import ABC, abstractmethod
from typing import AsyncGenerator

class LLMServiceError(Exception):
    """Base exception for Language Model service errors."""
    pass

class LLMOfflineError(LLMServiceError):
    """Raised when Language Model service is unreachable."""
    pass

class BaseLanguageModel(ABC):
    """Abstract interface defining the contract for LLM provider implementations."""

    @abstractmethod
    async def check_health(self) -> bool:
        """Check if the LLM provider service is online and reachable."""
        pass

    @abstractmethod
    async def generate_response(self, prompt: str) -> str:
        """Generate a complete text response synchronously/non-streaming."""
        pass

    @abstractmethod
    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response text tokens asynchronously."""
        pass

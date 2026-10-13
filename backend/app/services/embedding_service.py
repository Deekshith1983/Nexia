"""Local embedding service adapter using Ollama nomic-embed-text."""

import logging
from typing import List, Optional
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(Exception):
    """Raised when the Ollama embedding service is unreachable or errors."""
    pass


class EmbeddingDimensionError(Exception):
    """Raised when the embedding vector dimension does not match the expected size."""
    pass


class EmbeddingService:
    """Provides async embeddings via Ollama nomic-embed-text."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        expected_dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL
        self.expected_dimension = expected_dimension or settings.EMBEDDING_DIMENSIONS
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.timeout = timeout

    async def check_health(self) -> bool:
        """Check if Ollama embedding endpoint is responding."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code != 200:
                    return False
                tags = res.json().get("models", [])
                return any(self.model in m.get("name", "") for m in tags)
        except Exception as e:
            logger.warning("Embedding service health check failed: %s", e)
            return False

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Process in batches with shared HTTP client session
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    payload = {
                        "model": self.model,
                        "input": batch,
                    }
                    try:
                        response = await client.post(
                            f"{self.base_url}/api/embed",
                            json=payload,
                        )
                    except httpx.RequestError as exc:
                        raise EmbeddingUnavailableError(
                            f"Failed to connect to embedding service at {self.base_url}: {exc}"
                        ) from exc

                    if response.status_code != 200:
                        raise EmbeddingUnavailableError(
                            f"Embedding service returned HTTP {response.status_code}: {response.text}"
                        )

                    data = response.json()
                    embeddings = data.get("embeddings")
                    if not embeddings or len(embeddings) != len(batch):
                        raise EmbeddingUnavailableError(
                            f"Expected {len(batch)} embeddings, received {len(embeddings) if embeddings else 0}"
                        )

                    for vec in embeddings:
                        if len(vec) != self.expected_dimension:
                            raise EmbeddingDimensionError(
                                f"Expected vector dimension {self.expected_dimension}, got {len(vec)}"
                            )
                        all_embeddings.append(vec)
        except httpx.RequestError as exc:
            raise EmbeddingUnavailableError(
                f"Failed to connect to embedding service at {self.base_url}: {exc}"
            ) from exc

        return all_embeddings

    async def embed_text(self, text: str) -> List[float]:
        """Embed a single text string."""
        results = await self.embed_batch([text])
        if not results:
            raise EmbeddingUnavailableError("No embedding was generated for the input text.")
        return results[0]

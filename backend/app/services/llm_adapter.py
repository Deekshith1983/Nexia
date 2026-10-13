"""Ollama LLM adapter for answer generation using local gemma3:4b."""

import logging
from typing import Optional, Tuple
import httpx
from pydantic import BaseModel

from app.core.config import settings
from app.services.prompts import SYSTEM_PROMPT, PromptContext, build_answer_prompt

logger = logging.getLogger(__name__)


class LLMUnavailableError(Exception):
    """Raised when Ollama LLM endpoint is unreachable."""
    pass


class LLMTimeoutError(Exception):
    """Raised when generation times out."""
    pass


class LLMOutputError(Exception):
    """Raised when generation returns empty or malformed output."""
    pass


class LLMGenerationResult(BaseModel):
    """Result returned by LLM adapter with profiling metadata."""

    answer_text: str
    status: str  # "ANSWERED" | "INSUFFICIENT_EVIDENCE" | "CONFLICTING_EVIDENCE" | "LLM_FAILED"
    model: str
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    generation_time_ms: float = 0.0
    load_duration_ms: Optional[float] = None
    prompt_eval_duration_ms: Optional[float] = None
    eval_duration_ms: Optional[float] = None
    tokens_per_second: Optional[float] = None


class OllamaLLMAdapter:
    """Manages prompt execution and response extraction via Ollama."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or float(settings.LLM_TIMEOUT_SECONDS)
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

    async def check_health(self) -> bool:
        """Verify if Ollama model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code != 200:
                    return False
                models = res.json().get("models", [])
                return any(self.model in m.get("name", "") for m in models)
        except Exception as e:
            logger.warning("LLM health check failed: %s", e)
            return False

    async def generate_answer(self, context: PromptContext) -> LLMGenerationResult:
        """Generate a grounded answer for the provided context."""
        prompt_text = build_answer_prompt(context)

        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": prompt_text,
            "stream": False,
            "options": {
                "num_thread": settings.LLM_NUM_THREADS,
                "num_ctx": settings.LLM_NUM_CTX,
                "num_predict": settings.LLM_NUM_PREDICT,
                "temperature": self.temperature,
            },
            "keep_alive": settings.LLM_KEEP_ALIVE,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"LLM request timed out after {self.timeout}s: {exc}") from exc
        except httpx.RequestError as exc:
            raise LLMUnavailableError(f"Failed to connect to Ollama at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise LLMOutputError(f"Ollama returned HTTP {response.status_code}: {response.text}")

        data = response.json()
        raw_answer = data.get("response", "").strip()

        if not raw_answer:
            raise LLMOutputError("LLM returned an empty response.")

        # Classify status based on answer content and context
        status = "ANSWERED"
        if raw_answer.startswith("INSUFFICIENT_EVIDENCE:") or "insufficient evidence" in raw_answer.lower():
            status = "INSUFFICIENT_EVIDENCE"
        elif context.conflicts and ("conflict" in raw_answer.lower() or "dispute" in raw_answer.lower()):
            status = "CONFLICTING_EVIDENCE"

        total_duration = data.get("total_duration", 0) / 1_000_000.0  # nanoseconds to ms
        load_duration = data.get("load_duration", 0) / 1_000_000.0
        prompt_eval_duration = data.get("prompt_eval_duration", 0) / 1_000_000.0
        eval_duration = data.get("eval_duration", 0) / 1_000_000.0
        eval_count = data.get("eval_count", 0)
        tokens_per_sec = (eval_count / (eval_duration / 1000.0)) if eval_duration > 0 else 0.0

        return LLMGenerationResult(
            answer_text=raw_answer,
            status=status,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            response_tokens=eval_count,
            generation_time_ms=round(total_duration, 2),
            load_duration_ms=round(load_duration, 2),
            prompt_eval_duration_ms=round(prompt_eval_duration, 2),
            eval_duration_ms=round(eval_duration, 2),
            tokens_per_second=round(tokens_per_sec, 2),
        )

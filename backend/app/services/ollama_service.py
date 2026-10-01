import json
import asyncio
import time
import httpx
from typing import AsyncGenerator
from app.config import settings
from app.core.llm_interface import BaseLanguageModel, LLMServiceError, LLMOfflineError
from app.optimization.cache import local_cache_manager
from app.optimization.telemetry import telemetry_tracker

class OllamaServiceError(LLMServiceError):
    """Base exception for Ollama service errors."""
    pass

class OllamaOfflineError(LLMOfflineError, OllamaServiceError):
    """Raised when Ollama server is unreachable."""
    pass

class OllamaService(BaseLanguageModel):
    def __init__(self, base_url: str = None, model: str = None, timeout: float = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self.model = model or settings.ollama_model
        self.timeout = float(timeout or settings.ollama_timeout)

    async def check_health(self, force_live: bool = False) -> bool:
        """Check if Ollama server is reachable, with lightweight caching."""
        if not force_live:
            cached_ready = local_cache_manager.get_model_readiness()
            if cached_ready is not None:
                return cached_ready

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                is_ready = (res.status_code == 200)
                local_cache_manager.set_model_readiness(is_ready, ttl_seconds=60)
                return is_ready
        except Exception:
            local_cache_manager.set_model_readiness(False, ttl_seconds=10)
            return False

    async def generate_response(self, prompt: str) -> str:
        """Send prompt to Ollama model and return generated text."""
        start_t = time.perf_counter()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                duration = time.perf_counter() - start_t
                telemetry_tracker.record_llm_call(duration, call_type="generate")
                
                if response.status_code == 404:
                    raise OllamaServiceError(f"Model '{self.model}' not found in Ollama. Please run `ollama pull {self.model}`.")
                elif response.status_code != 200:
                    raise OllamaServiceError(f"Ollama returned HTTP error {response.status_code}: {response.text}")
                
                data = response.json()
                return data.get("response", "").strip()

        except httpx.ConnectError:
            raise OllamaOfflineError(f"Cannot connect to Ollama server at {self.base_url}. Please ensure Ollama service is running.")
        except httpx.TimeoutException:
            raise OllamaServiceError("Timeout waiting for Ollama model inference response.")
        except OllamaServiceError:
            raise
        except Exception as e:
            raise OllamaServiceError(f"Unexpected error communicating with Ollama: {str(e)}")

    async def stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response chunks from Ollama model asynchronously."""
        start_t = time.perf_counter()
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "top_p": 0.8,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 404:
                        raise OllamaServiceError(f"Model '{self.model}' not found in Ollama. Please run `ollama pull {self.model}`.")
                    elif response.status_code != 200:
                        raise OllamaServiceError(f"Ollama returned HTTP error {response.status_code}")

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            duration = time.perf_counter() - start_t
            telemetry_tracker.record_llm_call(duration, call_type="stream")

        except asyncio.CancelledError:
            duration = time.perf_counter() - start_t
            telemetry_tracker.record_llm_call(duration, call_type="stream_cancelled")
            # Client cancelled request or disconnected. Exiting async with client.stream closes the socket connection to Ollama.
            print("[INFO] Ollama generation cancelled by client disconnect.")
            return
        except httpx.ConnectError:
            raise OllamaOfflineError(f"Cannot connect to Ollama server at {self.base_url}. Please ensure Ollama service is running.")
        except httpx.TimeoutException:
            raise OllamaServiceError("Timeout waiting for Ollama streaming response.")
        except OllamaServiceError:
            raise
        except Exception as e:
            raise OllamaServiceError(f"Unexpected error streaming from Ollama: {str(e)}")

ollama_service = OllamaService()

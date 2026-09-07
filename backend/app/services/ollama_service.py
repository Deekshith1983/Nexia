import httpx
from app.config import settings

class OllamaServiceError(Exception):
    """Base exception for Ollama service errors."""
    pass

class OllamaOfflineError(OllamaServiceError):
    """Raised when Ollama server is unreachable."""
    pass

class OllamaService:
    def __init__(self, base_url: str = None, model: str = None, timeout: float = None):
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        self.model = model or settings.ollama_model
        self.timeout = float(timeout or settings.ollama_timeout)

    async def check_health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def generate_response(self, prompt: str) -> str:
        """Send prompt to Ollama model and return generated text."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                
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

    async def stream_response(self, prompt: str):
        """Stream response chunks from Ollama model asynchronously."""
        import json
        import asyncio
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
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

        except asyncio.CancelledError:
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

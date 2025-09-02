# src/llm.py
import os, requests
from typing import List, Optional

_BASE_URL = "https://integrate.api.nvidia.com/v1"
_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"

class Client:
    """Minimal client that matches OpenAI's /chat/completions shape."""
    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: str = _BASE_URL,
                 model: str = _MODEL,
                 temperature: float = 0.0):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set NVIDIA_API_KEY in your environment.")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature

    def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

def create_client() -> Client:
    return Client()

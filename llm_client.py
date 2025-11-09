# llm_client.py
import json
import requests
from typing import Optional

from config import LLM_BASE_URL

def call_local_llm(model: str, prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    """
    Calls a local LLM endpoint compatible with Ollama's /api/generate.
    Adjust if your local endpoint differs.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    resp = requests.post(LLM_BASE_URL, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    # Ollama returns { "response": "...", ... }
    return data.get("response", "").strip()

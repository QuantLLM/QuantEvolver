from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenAICompatibleClient:
    """Tiny OpenAI-compatible chat client using stdlib only."""

    base_url: str | None = None
    api_key: str | None = None
    model: str = "gpt-4o-mini"
    timeout: int = 120
    max_retries: int = 3

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        default_base = "https://api.openai.com/v1" if os.environ.get("OPENAI_API_KEY") else ""
        self.base_url = (self.base_url or os.environ.get("OPENAI_BASE_URL") or default_base).rstrip("/")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is required when not using the default OpenAI endpoint")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY or DASHSCOPE_API_KEY is required")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(20, 2 ** attempt))
        assert last_exc is not None
        raise last_exc

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

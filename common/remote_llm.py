from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RemoteLLMConfig:
    base_url: str
    api_key: str
    model: str = ""
    timeout_sec: int = 60
    endpoint_path: str = ""


class RemoteLLMClient:
    """Клиент для удаленных LLM-сервисов с несколькими популярными форматами API."""

    def __init__(self, config: RemoteLLMConfig):
        self.config = config

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None, temperature: float = 0.2) -> str:
        errors = []
        candidates = self._build_candidates(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)

        for endpoint, payload in candidates:
            try:
                data = self._post_json(endpoint, payload)
                text = self._extract_text(data)
                if text:
                    return text
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")

        raise RuntimeError(
            "Не удалось получить ответ от удаленного LLM API. "
            "Проверьте путь endpoint (параметр --remote-path). Детали: " + " | ".join(errors)
        )

    def _build_candidates(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        temperature: float,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        openai_payload: Dict[str, Any] = {
            "messages": [
                {"role": "system", "content": "Отвечай на русском языке строго по инструкции пользователя."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if max_new_tokens is not None:
            openai_payload["max_tokens"] = max_new_tokens
        if self.config.model.strip():
            openai_payload["model"] = self.config.model.strip()

        generate_payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        if max_new_tokens is not None:
            generate_payload["max_new_tokens"] = max_new_tokens
        if self.config.model.strip():
            generate_payload["model"] = self.config.model.strip()

        ollama_payload: Dict[str, Any] = {
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_new_tokens is not None:
            ollama_payload["options"]["num_predict"] = max_new_tokens
        if self.config.model.strip():
            ollama_payload["model"] = self.config.model.strip()

        if self.config.endpoint_path.strip():
            endpoint = self._normalize_path(self.config.endpoint_path)
            return [(endpoint, openai_payload), (endpoint, generate_payload), (endpoint, ollama_payload)]

        return [
            ("/v1/chat/completions", openai_payload),
            ("/chat/completions", openai_payload),
            ("/api/v1/chat/completions", openai_payload),
            ("/openai/v1/chat/completions", openai_payload),
            ("/v1/completions", {
                "prompt": prompt,
                "temperature": temperature,
                **({"max_tokens": max_new_tokens} if max_new_tokens is not None else {}),
                **({"model": self.config.model.strip()} if self.config.model.strip() else {}),
            }),
            ("/generate", generate_payload),
            ("/v1/generate", generate_payload),
            ("/api/v1/generate", generate_payload),
            ("/api/generate", ollama_payload),
            ("/v1/text/generate", generate_payload),
        ]

    @staticmethod
    def _normalize_path(path: str) -> str:
        cleaned = path.strip()
        if not cleaned.startswith("/"):
            cleaned = "/" + cleaned
        return cleaned

    def _post_json(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        base = self.config.base_url.rstrip("/")
        url = f"{base}{endpoint}"
        request = urllib.request.Request(url=url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self.config.api_key}")
        request.add_header("X-API-Key", self.config.api_key)
        request.add_header("api-key", self.config.api_key)
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                raw = response.read().decode("utf-8", errors="ignore")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            details = body.strip() or str(exc)
            raise RuntimeError(f"HTTP {exc.code}: {details}") from exc

    @staticmethod
    def _extract_text(data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"].strip()
                if isinstance(first.get("text"), str):
                    return first["text"].strip()

        if isinstance(data.get("data"), dict):
            nested = RemoteLLMClient._extract_text(data["data"])
            if nested:
                return nested
        if isinstance(data.get("result"), dict):
            nested = RemoteLLMClient._extract_text(data["result"])
            if nested:
                return nested

        for key in ("generated_text", "text", "response", "output", "result"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

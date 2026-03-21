from __future__ import annotations

from typing import Optional

from openai import OpenAI

from backend.core.config import settings
from backend.utils.embeddings import normalize_base_url, resolve_api_key


def get_llm_client() -> OpenAI:
    llm = settings.llm
    api_key = resolve_api_key(
        llm.api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"]
    )
    if not api_key:
        raise RuntimeError("LLM API key required")
    return OpenAI(
        api_key=api_key,
        base_url=normalize_base_url(llm.base_url),
        timeout=180.0,
        max_retries=2,
    )


def generate_text(
    prompt: str,
    *,
    system: str = "Отвечай на русском языке.",
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    client = get_llm_client()
    resp = client.chat.completions.create(
        model=model or settings.llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = resp.choices[0]
    return (choice.message.content or "").strip()

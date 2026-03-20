"""
Сервис эмбеддингов.

Поддерживает два режима:
- local: fastembed (модели скачиваются локально)
- api: OpenAI-совместимый эндпоинт (remote embedder)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Утилиты ─────────────────────────────────────────────────


def normalize_openai_base_url(url: str) -> str:
    """Добавляет /v1 к base_url если отсутствует."""
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def resolve_api_key(
    primary: Optional[str],
    fallback_envs: Optional[List[str]] = None,
) -> Optional[str]:
    """Ищет API-ключ: сначала primary, потом env-переменные."""
    if primary:
        return primary
    for env_name in fallback_envs or []:
        value = os.getenv(env_name)
        if value:
            return value
    return None


# ── Фабрика эмбеддера ──────────────────────────────────────


def get_embedder(
    model_name: str,
    cache_dir: Path,
    embedder_url: Optional[str] = None,
    embedder_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Создаёт эмбеддер нужного типа.

    Args:
        model_name: имя модели (для fastembed или remote API).
        cache_dir: директория кэша моделей.
        embedder_url: URL remote эмбеддера (если None — используем local).
        embedder_api_key: API-ключ для remote эмбеддера.

    Returns:
        Словарь ``{"kind": "local"|"api", "client": ...}``.
    """
    if embedder_url:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        api_key = resolve_api_key(
            embedder_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"]
        )
        if not api_key:
            raise RuntimeError(
                "API key required for remote embedder. "
                "Set embedder_api_key or HACKAI_API_KEY / OPENAI_API_KEY env."
            )
        client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(embedder_url),
            timeout=120.0,
            max_retries=2,
        )
        return {"kind": "api", "client": client}

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)

    from fastembed import TextEmbedding  # pylint: disable=import-outside-toplevel

    return {
        "kind": "local",
        "client": TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir)
        ),
    }


# ── Подготовка текста (e5-специфичный префикс) ──────────────


def prepare_embedding_text(
    text: str,
    model_name: str,
    is_query: bool,
) -> str:
    """Добавляет префикс ``query:`` / ``passage:`` для e5-моделей."""
    if "e5" in model_name.lower():
        prefix = "query" if is_query else "passage"
        return f"{prefix}: {text}"
    return text


# ── Получение эмбеддингов ───────────────────────────────────

_MAX_API_RETRIES = 4
_RETRY_BASE_DELAY = 1.5


def embed_texts(
    embedder: Dict[str, Any],
    texts: List[str],
    model_name: str,
    is_query: bool,
) -> List[List[float]]:
    """
    Эмбеддит список текстов через выбранный embedder.

    Args:
        embedder: объект от ``get_embedder``.
        texts: тексты для эмбеддинга.
        model_name: имя модели (нужно для e5-префиксов).
        is_query: True для запросов, False для документов.

    Returns:
        Список векторов.
    """
    prepared = [
        prepare_embedding_text(t, model_name=model_name, is_query=is_query)
        for t in texts
    ]

    if embedder["kind"] == "api":
        return _embed_via_api(embedder["client"], prepared, model_name)

    return _embed_via_local(embedder["client"], prepared)


def _embed_via_api(
    client: Any,
    texts: List[str],
    model_name: str,
) -> List[List[float]]:
    """Эмбеддинг через OpenAI-совместимый API с ретраями."""
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_API_RETRIES):
        try:
            resp = client.embeddings.create(model=model_name, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            if attempt < _MAX_API_RETRIES - 1:
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
    raise RuntimeError(f"Embedder API failed after {_MAX_API_RETRIES} retries: {last_err}")


def _embed_via_local(
    client: Any,
    texts: List[str],
) -> List[List[float]]:
    """Эмбеддинг через локальный fastembed."""
    return [vec.tolist() for vec in client.embed(texts)]

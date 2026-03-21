from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_MAX_RETRIES = 4
_RETRY_DELAY = 1.5


def resolve_api_key(
    primary: Optional[str],
    fallback_envs: Optional[List[str]] = None,
) -> Optional[str]:
    if primary:
        return primary
    for name in fallback_envs or []:
        val = os.getenv(name)
        if val:
            return val
    return None


def normalize_base_url(url: str) -> str:
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def get_embedder(
    model_name: str,
    cache_dir: Path,
    embedder_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    if embedder_url:
        from openai import OpenAI  # pylint: disable=import-outside-toplevel

        resolved_key = resolve_api_key(
            api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"]
        )
        if not resolved_key:
            raise RuntimeError("API key required for remote embedder")
        client = OpenAI(
            api_key=resolved_key,
            base_url=normalize_base_url(embedder_url),
            timeout=120.0,
            max_retries=2,
        )
        return {"kind": "api", "client": client}

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    from fastembed import TextEmbedding  # pylint: disable=import-outside-toplevel

    return {
        "kind": "local",
        "client": TextEmbedding(model_name=model_name, cache_dir=str(cache_dir)),
    }


def embed_texts(
    embedder: Dict[str, Any],
    texts: List[str],
    model_name: str,
    *,
    is_query: bool = False,
) -> List[List[float]]:
    prepared = [_prepare(t, model_name, is_query) for t in texts]

    if embedder["kind"] == "api":
        return _embed_api(embedder["client"], prepared, model_name)
    return [vec.tolist() for vec in embedder["client"].embed(prepared)]


def _prepare(text: str, model_name: str, is_query: bool) -> str:
    if "e5" in model_name.lower():
        return f"{'query' if is_query else 'passage'}: {text}"
    return text


def _embed_api(
    client: Any, texts: List[str], model_name: str
) -> List[List[float]]:
    last_err: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(model=model_name, input=texts)
            return [item.embedding for item in resp.data]
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_DELAY * (2 ** attempt))
    raise RuntimeError(f"Embedder failed after {_MAX_RETRIES} retries: {last_err}")

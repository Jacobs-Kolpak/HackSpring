"""
Сервис поиска и генерации ответов (RAG pipeline).

Гибридный поиск: dense (Qdrant) + BM25 лексический ранкинг.
Опциональный API rerank и генерация ответа через LLM.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from backend.services.embeddings import (
    embed_texts,
    get_embedder,
    normalize_openai_base_url,
    resolve_api_key,
)
from backend.services.vector_store import (
    build_filter,
    get_client,
    search_vectors,
)


# ── Утилиты scoring ─────────────────────────────────────────


def tokenize_for_lexical(text: str) -> List[str]:
    """Токенизация для лексического ранкинга."""
    tokens = re.findall(r"[a-zа-яё0-9]+", text.lower())
    return [tok for tok in tokens if len(tok) > 1]


def minmax_normalize(values: List[float]) -> List[float]:
    """Min-max нормализация списка скоров."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-12:
        return [0.0] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def lexical_overlap_score(query: str, text: str) -> float:
    """Доля пересечения токенов запроса с документом."""
    q_tokens = set(tokenize_for_lexical(query))
    d_tokens = set(tokenize_for_lexical(text))
    if not q_tokens or not d_tokens:
        return 0.0
    return len(q_tokens & d_tokens) / max(1, len(q_tokens))


# ── Rerank: BM25 гибрид ────────────────────────────────────


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    dense_weight: float,
) -> List[Dict[str, Any]]:
    """
    Гибридный ранкинг: dense score + BM25.

    Args:
        query: текст запроса.
        candidates: результаты от vector store (с dense_score_raw).
        dense_weight: вес dense-компоненты (0..1).
    """
    dense_weight = min(1.0, max(0.0, dense_weight))
    bm25_weight = 1.0 - dense_weight

    tokenized_docs = [tokenize_for_lexical(c["text"]) for c in candidates]
    tokenized_query = tokenize_for_lexical(query)

    bm25_scores = [0.0] * len(candidates)
    if tokenized_query and any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(tokenized_query).tolist()

    dense_norm = minmax_normalize(
        [c["dense_score_raw"] for c in candidates]
    )
    bm25_norm = minmax_normalize(bm25_scores)
    local_lex = minmax_normalize(
        [lexical_overlap_score(query, c["text"]) for c in candidates]
    )

    reranked: List[Dict[str, Any]] = []
    for candidate, d_n, b_n, l_n in zip(
        candidates, dense_norm, bm25_norm, local_lex
    ):
        lexical_score = max(b_n, l_n)
        item = dict(candidate)
        item["dense_score"] = d_n
        item["bm25_score"] = lexical_score
        item["local_lexical_score"] = l_n
        item["score"] = dense_weight * d_n + bm25_weight * lexical_score
        reranked.append(item)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


# ── API Rerank (опциональный) ───────────────────────────────


def api_rerank_scores(
    rerank_url: str,
    rerank_api_key: str,
    rerank_model: Optional[str],
    query: str,
    documents: List[str],
) -> Optional[List[float]]:
    """Вызывает внешний rerank API и возвращает скоры."""
    import httpx  # pylint: disable=import-outside-toplevel

    url = f"{normalize_openai_base_url(rerank_url)}/rerank"
    headers = {"Authorization": f"Bearer {rerank_api_key}"}
    payload: Dict[str, Any] = {
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    if rerank_model:
        payload["model"] = rerank_model

    try:
        resp = httpx.post(url, headers=headers, json=payload, timeout=40.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception:  # pylint: disable=broad-except
        return None

    rows = data.get("results") or data.get("data") or []
    scores = [0.0] * len(documents)
    for row in rows:
        idx = row.get("index")
        if idx is None or not (0 <= int(idx) < len(documents)):
            continue
        score = row.get("relevance_score")
        if score is None:
            score = row.get("score", 0.0)
        scores[int(idx)] = float(score)
    return scores


# ── Основной pipeline поиска ────────────────────────────────


def retrieve_chunks(
    query: str,
    *,
    db_path: Optional[Path] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = "docs",
    embedding_model: str = "intfloat/multilingual-e5-large",
    embedding_cache: Optional[Path] = None,
    embedder_url: Optional[str] = None,
    embedder_api_key: Optional[str] = None,
    top_k: int = 5,
    fetch_k: int = 30,
    min_score: float = 0.20,
    dense_weight: float = 0.75,
    source_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    rerank_blend: float = 0.35,
    rerank_url: Optional[str] = None,
    rerank_api_key: Optional[str] = None,
    rerank_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Полный пайплайн: embed query -> Qdrant search -> hybrid rerank -> API rerank.

    Returns:
        Список результатов с полями: rank, score, chunk_id, text, source_name и др.
    """
    fetch_k = max(fetch_k, top_k)
    cache = embedding_cache or Path("data/embedding_cache")

    client = get_client(
        db_path=db_path, qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key
    )
    embedder = get_embedder(
        model_name=embedding_model,
        cache_dir=cache,
        embedder_url=embedder_url,
        embedder_api_key=embedder_api_key,
    )

    query_vector = embed_texts(
        embedder, [query], model_name=embedding_model, is_query=True
    )[0]

    query_filter = build_filter(
        source_name=source_name, metadata=metadata or {}
    )

    qdrant_candidates = search_vectors(
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        limit=fetch_k,
        query_filter=query_filter,
    )

    if not qdrant_candidates:
        return []

    # Гибридный rerank: dense + BM25
    reranked = rerank_candidates(
        query=query,
        candidates=qdrant_candidates,
        dense_weight=dense_weight,
    )

    # Инициализируем api_rerank_score
    for row in reranked:
        row["api_rerank_score"] = 0.0

    # Опциональный API rerank
    resolved_rerank_url = rerank_url or embedder_url
    resolved_rerank_key = resolve_api_key(
        rerank_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"]
    )
    if resolved_rerank_url and resolved_rerank_key and reranked:
        scores = api_rerank_scores(
            rerank_url=resolved_rerank_url,
            rerank_api_key=resolved_rerank_key,
            rerank_model=rerank_model,
            query=query,
            documents=[c["text"] for c in reranked],
        )
        if scores is not None:
            api_norm = minmax_normalize(scores)
            blend = min(1.0, max(0.0, rerank_blend))
            for item, a_n in zip(reranked, api_norm):
                item["api_rerank_score"] = a_n
                item["score"] = (
                    (1.0 - blend) * float(item["score"]) + blend * a_n
                )
            reranked.sort(key=lambda x: x["score"], reverse=True)

    # Фильтрация по min_score и обрезка до top_k
    results = [
        item for item in reranked if item["score"] >= min_score
    ][:top_k]
    for idx, item in enumerate(results, start=1):
        item["rank"] = idx

    return results


# ── Контекст + LLM ответ ───────────────────────────────────


def build_context(results: List[Dict[str, Any]]) -> str:
    """Собирает контекст для LLM из результатов поиска."""
    blocks = []
    for item in results:
        blocks.append(
            f"[Source: {item['source_name']} | "
            f"chunk={item['chunk_index']} | "
            f"score={item['score']:.4f}]\n"
            f"{item['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(
    query: str,
    context: str,
    model: str,
    llm_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
) -> str:
    """
    Генерирует ответ через OpenAI-совместимый LLM.

    Args:
        query: вопрос пользователя.
        context: контекст из RAG.
        model: имя модели.
        llm_url: base URL LLM API.
        llm_api_key: API-ключ.

    Returns:
        Текст ответа.
    """
    api_key = resolve_api_key(
        llm_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"]
    )
    if not api_key:
        raise RuntimeError(
            "API key required for LLM. "
            "Set llm_api_key or HACKAI_API_KEY / OPENAI_API_KEY env."
        )

    from openai import OpenAI  # pylint: disable=import-outside-toplevel

    if llm_url:
        client = OpenAI(
            api_key=api_key,
            base_url=normalize_openai_base_url(llm_url),
            timeout=180.0,
            max_retries=2,
        )
    else:
        client = OpenAI(api_key=api_key, timeout=180.0, max_retries=2)

    system_prompt = (
        "Ты RAG-ассистент. Отвечай строго на основе переданного контекста. "
        "Если данных не хватает, прямо скажи об этом."
    )
    user_prompt = (
        f"Вопрос:\n{query}\n\n"
        f"Контекст:\n{context}\n\n"
        "Дай краткий и точный ответ на русском языке."
    )

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    return response.output_text.strip()

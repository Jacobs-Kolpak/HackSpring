from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from backend.core.config import settings
from backend.utils.chunker import Chunk, build_chunks
from backend.utils.embeddings import (
    embed_texts,
    get_embedder,
    normalize_base_url,
    resolve_api_key,
)


def _get_client() -> QdrantClient:
    rag = settings.rag
    db_path = Path(rag.vector_store_path).resolve()
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def _ensure_collection(
    client: QdrantClient, name: str, vector_size: int
) -> None:
    if client.collection_exists(collection_name=name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=vector_size, distance=models.Distance.COSINE
        ),
    )


def _get_embedder():  # type: ignore[no-untyped-def]
    rag = settings.rag
    return get_embedder(
        model_name=rag.embeddings_model,
        cache_dir=Path("data/embedding_cache").resolve(),
        embedder_url=rag.embedder_url or None,
        api_key=rag.embedder_api_key or None,
    )


def ingest(
    paths: List[Path],
    collection: Optional[str] = None,
    chunk_size: int = 0,
    chunk_overlap: int = 0,
    source_name_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    rag = settings.rag
    col = collection or rag.collection
    size = chunk_size if chunk_size > 0 else rag.chunk_size
    overlap = chunk_overlap if chunk_overlap > 0 else rag.chunk_overlap

    chunks = build_chunks(
        paths,
        size=size,
        overlap=overlap,
        source_name_overrides=source_name_overrides,
    )
    if not chunks:
        return {"indexed_files": len(paths), "inserted_chunks": 0, "collection": col}

    embedder = _get_embedder()
    model = rag.embeddings_model

    sample = embed_texts(embedder, [chunks[0].text], model)[0]
    client = _get_client()
    _ensure_collection(client, col, len(sample))

    # Удаляем предыдущие версии этих же документов по нормализованному source_name.
    # Иначе при загрузке через временные файлы (`tmp*.pdf`) коллекция зарастает дубликатами.
    for src_name in {c.source_name for c in chunks}:
        client.delete(
            collection_name=col,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[
                    models.FieldCondition(
                        key="source_name",
                        match=models.MatchValue(value=src_name),
                    )
                ])
            ),
            wait=True,
        )

    vectors = embed_texts(embedder, [c.text for c in chunks], model)
    _upsert_batch(client, col, chunks, vectors)

    return {
        "indexed_files": len(paths),
        "inserted_chunks": len(chunks),
        "collection": col,
    }


def _upsert_batch(
    client: QdrantClient,
    collection: str,
    chunks: List[Chunk],
    vectors: List[List[float]],
    batch_size: int = 64,
) -> None:
    for i in range(0, len(chunks), batch_size):
        batch_c = chunks[i:i + batch_size]
        batch_v = vectors[i:i + batch_size]
        points = [
            models.PointStruct(
                id=c.chunk_id,
                vector=v,
                payload={
                    "source_path": c.source_path,
                    "source_name": c.source_name,
                    "chunk_index": c.chunk_index,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "metadata": c.metadata,
                },
            )
            for c, v in zip(batch_c, batch_v)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)


def _val(explicit: Optional[float], default: float) -> float:
    return explicit if explicit is not None else default


def _val_int(explicit: Optional[int], default: int) -> int:
    return explicit if explicit is not None else default


def retrieve(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    query: str,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    min_score: Optional[float] = None,
    dense_weight: Optional[float] = None,
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rag = settings.rag
    params = {
        "col": collection or rag.collection,
        "k": _val_int(top_k, rag.top_k),
        "fk": max(_val_int(fetch_k, rag.fetch_k), _val_int(top_k, rag.top_k)),
        "ms": _val(min_score, rag.min_score),
        "dw": _val(dense_weight, rag.dense_weight),
    }
    return _do_retrieve(query, source_name, params)


def _do_retrieve(
    query: str,
    source_name: Optional[str],
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rag = settings.rag
    embedder = _get_embedder()
    query_vec = embed_texts(
        embedder, [query], rag.embeddings_model, is_query=True
    )[0]

    client = _get_client()
    candidates = _search_qdrant(
        client, params["col"], query_vec, params["fk"], _build_filter(source_name)
    )
    if not candidates:
        return []

    reranked = _api_rerank(query, _hybrid_rerank(query, candidates, params["dw"]))

    filtered = [r for r in reranked if r["score"] >= params["ms"]]
    results = _dedupe_results(filtered, params["k"])
    for idx, item in enumerate(results, 1):
        item["rank"] = idx
    return results


def _build_filter(
    source_name: Optional[str],
) -> Optional[models.Filter]:
    if not source_name:
        return None
    return models.Filter(must=[
        models.FieldCondition(
            key="source_name",
            match=models.MatchValue(value=source_name),
        )
    ])


def _search_qdrant(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    limit: int,
    query_filter: Optional[models.Filter],
) -> List[Dict[str, Any]]:
    resp = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )
    results: List[Dict[str, Any]] = []
    for hit in resp.points:
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        if not text.strip():
            continue
        if _is_low_signal_chunk(text):
            continue
        results.append({
            "rank": 0,
            "dense_score_raw": float(hit.score),
            "chunk_id": payload.get("chunk_id", str(hit.id)),
            "source_name": payload.get("source_name", "unknown"),
            "source_path": payload.get("source_path", "unknown"),
            "chunk_index": payload.get("chunk_index", -1),
            "text": text,
        })
    return results


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zа-яё0-9]+", text.lower()) if len(t) > 1]


def _is_low_signal_chunk(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if len(clean) < 120:
        return True
    lines = [ln.strip() for ln in clean.splitlines() if ln.strip()]
    if not lines:
        return True

    toc_like = sum(1 for ln in lines if re.search(r"\.{4,}\s*\d{1,3}\s*$", ln))
    numeric_only = sum(1 for ln in lines if re.fullmatch(r"\d{1,3}", ln))
    if toc_like >= 2 or (toc_like + numeric_only) >= max(3, len(lines) // 2):
        return True

    weird_tokens = re.findall(r"[а-яёa-z]{1,3}э[а-яёa-z]{1,3}", clean.lower())
    if len(weird_tokens) >= 3 and len(_tokenize(clean)) < 80:
        return True
    return False


def _text_signature(text: str, max_chars: int = 800) -> str:
    clean = re.sub(r"\s+", " ", text.strip().lower())
    if len(clean) > max_chars:
        clean = clean[:max_chars]
    return clean


def _dedupe_results(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    seen_text: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for item in rows:
        sig = _text_signature(str(item.get("text", "")))
        if not sig or sig in seen_text:
            continue
        seen_text.add(sig)
        deduped.append(item)
        if len(deduped) >= limit:
            break
    return deduped


def _minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _hybrid_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    dense_weight: float,
) -> List[Dict[str, Any]]:
    dw = min(1.0, max(0.0, dense_weight))
    bw = 1.0 - dw

    docs = [_tokenize(c["text"]) for c in candidates]
    q_tok = _tokenize(query)

    bm25_raw = [0.0] * len(candidates)
    if q_tok and any(docs):
        bm25_raw = BM25Okapi(docs).get_scores(q_tok).tolist()

    dense_n = _minmax([c["dense_score_raw"] for c in candidates])
    bm25_n = _minmax(bm25_raw)

    result: List[Dict[str, Any]] = []
    for cand, dn, bn in zip(candidates, dense_n, bm25_n):
        item = dict(cand)
        item["dense_score"] = dn
        item["bm25_score"] = bn
        item["score"] = dw * dn + bw * bn
        item["api_rerank_score"] = 0.0
        result.append(item)

    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def _api_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rag = settings.rag
    url = rag.rerank_url or rag.embedder_url
    key = resolve_api_key(rag.rerank_api_key, ["HACKAI_API_KEY", "OPENAI_API_KEY"])

    if not url or not key or not candidates:
        return candidates

    import httpx  # pylint: disable=import-outside-toplevel

    try:
        resp = httpx.post(
            f"{normalize_base_url(url)}/rerank",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query,
                "documents": [c["text"] for c in candidates],
                "top_n": len(candidates),
                **({"model": rag.rerank_model} if rag.rerank_model else {}),
            },
            timeout=40.0,
        )
        resp.raise_for_status()
        rows = resp.json().get("results") or resp.json().get("data") or []
    except Exception:  # pylint: disable=broad-except
        return candidates

    scores = [0.0] * len(candidates)
    for row in rows:
        idx = row.get("index")
        if idx is not None and 0 <= int(idx) < len(candidates):
            scores[int(idx)] = float(row.get("relevance_score", row.get("score", 0)))

    blend = min(1.0, max(0.0, rag.rerank_blend))
    api_n = _minmax(scores)
    for item, an in zip(candidates, api_n):
        item["api_rerank_score"] = an
        item["score"] = (1.0 - blend) * item["score"] + blend * an

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def build_context(results: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{r['source_name']} chunk={r['chunk_index']}]\n{r['text']}"
        for r in results
    )


def ask(query: str, results: List[Dict[str, Any]], model: Optional[str] = None) -> str:
    from backend.utils.llm import generate_text  # pylint: disable=import-outside-toplevel

    context = build_context(results)
    return generate_text(
        prompt=(
            f"Вопрос:\n{query}\n\nКонтекст:\n{context}\n\n"
            "Дай краткий и точный ответ на русском языке."
        ),
        system=(
            "Ты RAG-ассистент. Отвечай строго на основе контекста. "
            "Если данных не хватает, скажи об этом."
        ),
        model=model,
        temperature=0.0,
    )

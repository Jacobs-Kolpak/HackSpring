"""
Сервис работы с векторным хранилищем (Qdrant).

Предоставляет функции для создания коллекций, upsert/delete точек
и поиска по вектору.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from backend.services.chunker import Chunk


# ── Клиент ──────────────────────────────────────────────────


def get_client(
    db_path: Optional[Path] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Создаёт Qdrant-клиент.

    Args:
        db_path: путь к локальному хранилищу (если qdrant_url не задан).
        qdrant_url: URL удалённого Qdrant.
        qdrant_api_key: API-ключ для удалённого Qdrant.

    Returns:
        Экземпляр QdrantClient.
    """
    if qdrant_url:
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    if not db_path:
        raise ValueError("db_path is required for local Qdrant mode")
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


# ── Управление коллекциями ──────────────────────────────────


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """Создаёт коллекцию, если она ещё не существует."""
    if client.collection_exists(collection_name=collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )


# ── Upsert / Delete ────────────────────────────────────────


def remove_sources(
    client: QdrantClient,
    collection_name: str,
    source_paths: List[str],
) -> None:
    """Удаляет все точки с указанными source_path."""
    for src in source_paths:
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source_path",
                            match=models.MatchValue(value=src),
                        )
                    ]
                )
            ),
            wait=True,
        )


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Chunk],
    vectors: List[List[float]],
    batch_size: int = 64,
) -> int:
    """
    Загружает чанки с векторами в Qdrant батчами.

    Returns:
        Количество загруженных точек.
    """
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "source_path": chunk.source_path,
                    "source_name": chunk.source_name,
                    "chunk_index": chunk.chunk_index,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(batch_chunks, batch_vectors)
        ]
        client.upsert(
            collection_name=collection_name, points=points, wait=True
        )
        total += len(points)

    return total


# ── Фильтры ────────────────────────────────────────────────


def build_filter(
    source_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[models.Filter]:
    """Строит Qdrant-фильтр по source_name и/или метаданным."""
    must: List[models.FieldCondition] = []

    if source_name:
        must.append(
            models.FieldCondition(
                key="source_name",
                match=models.MatchValue(value=source_name),
            )
        )

    for key, value in (metadata or {}).items():
        field_key = (
            key
            if key in {"source_name", "source_path", "chunk_id", "chunk_index"}
            else f"metadata.{key}"
        )
        must.append(
            models.FieldCondition(
                key=field_key,
                match=models.MatchValue(value=value),
            )
        )

    if not must:
        return None
    return models.Filter(must=must)


# ── Поиск ───────────────────────────────────────────────────


def search_vectors(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    query_filter: Optional[models.Filter] = None,
) -> List[Dict[str, Any]]:
    """
    Ищет ближайшие вектора в коллекции.

    Returns:
        Список словарей с полями: chunk_id, source_name, source_path,
        chunk_index, text, dense_score_raw.
    """
    hits_response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
        with_vectors=False,
    )

    candidates: List[Dict[str, Any]] = []
    for hit in hits_response.points:
        payload = hit.payload or {}
        text = str(payload.get("text", ""))
        if not text.strip():
            continue
        candidates.append(
            {
                "rank": 0,
                "dense_score_raw": float(hit.score),
                "chunk_id": payload.get("chunk_id", str(hit.id)),
                "source_name": payload.get("source_name", "unknown"),
                "source_path": payload.get("source_path", "unknown"),
                "chunk_index": payload.get("chunk_index", -1),
                "text": text,
            }
        )

    return candidates

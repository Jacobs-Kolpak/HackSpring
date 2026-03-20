"""
API роутер для RAG-системы.

Эндпоинты:
- POST /ingest   — загрузка и индексация документов
- POST /retrieve — поиск по запросу
- POST /ask      — поиск + генерация ответа через LLM
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.services.chunker import build_chunks
from backend.services.document_reader import SUPPORTED_EXTENSIONS
from backend.services.embeddings import embed_texts, get_embedder
from backend.services.retrieval import (
    build_context,
    generate_answer,
    retrieve_chunks,
)
from backend.services.vector_store import (
    ensure_collection,
    get_client,
    remove_sources,
    upsert_chunks,
)

router = APIRouter(prefix="/api/rag", tags=["RAG"])


# ── Schemas ─────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    """Запрос на поиск чанков."""

    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    fetch_k: Optional[int] = Field(default=None, ge=1, le=200)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dense_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskRequest(RetrieveRequest):
    """Запрос на поиск + генерацию ответа."""

    model: Optional[str] = None


class RetrieveResponse(BaseModel):
    """Ответ с результатами поиска."""

    query: str
    returned: int
    results: List[Dict[str, Any]]


class AskResponse(BaseModel):
    """Ответ с генерацией от LLM."""

    query: str
    answer: str
    model: str
    used_chunks: int
    results: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    """Ответ после индексации."""

    indexed_files: int
    inserted_chunks: int
    collection: str


# ── Helpers ─────────────────────────────────────────────────


def _retrieve_with_config(
    query: str,
    collection: Optional[str] = None,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    min_score: Optional[float] = None,
    dense_weight: Optional[float] = None,
    source_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Вызывает retrieve_chunks, подставляя дефолты из settings.rag."""
    rag = settings.rag

    return retrieve_chunks(
        query=query,
        db_path=Path(rag.vector_store_path).resolve(),
        collection_name=collection or rag.collection,
        embedding_model=rag.embeddings_model,
        embedding_cache=Path("data/embedding_cache").resolve(),
        embedder_url=rag.embedder_url or None,
        embedder_api_key=rag.embedder_api_key or None,
        top_k=top_k if top_k is not None else rag.top_k,
        fetch_k=fetch_k if fetch_k is not None else rag.fetch_k,
        min_score=min_score if min_score is not None else rag.min_score,
        dense_weight=dense_weight if dense_weight is not None else rag.dense_weight,
        source_name=source_name,
        metadata=metadata,
        rerank_blend=rag.rerank_blend,
        rerank_url=rag.rerank_url or None,
        rerank_api_key=rag.rerank_api_key or None,
        rerank_model=rag.rerank_model or None,
    )


# ── Endpoints ───────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest_files(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None),
    chunk_size: int = Form(0),
    chunk_overlap: int = Form(0),
) -> IngestResponse:
    """Загружает файлы, разбивает на чанки и индексирует в Qdrant."""
    rag = settings.rag

    if chunk_size <= 0:
        chunk_size = rag.chunk_size
    if chunk_overlap <= 0:
        chunk_overlap = rag.chunk_overlap
    col = collection or rag.collection

    temp_paths: List[Path] = []
    try:
        for upload in files:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file: {upload.filename}. "
                    f"Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                )
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=ext
            ) as tmp:
                data = await upload.read()
                tmp.write(data)
                temp_paths.append(Path(tmp.name))

        chunks = build_chunks(
            paths=temp_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from uploaded files",
            )

        db_path = Path(rag.vector_store_path).resolve()
        cache_dir = Path("data/embedding_cache").resolve()

        embedder = get_embedder(
            model_name=rag.embeddings_model,
            cache_dir=cache_dir,
            embedder_url=rag.embedder_url or None,
            embedder_api_key=rag.embedder_api_key or None,
        )

        # Определяем размер вектора по первому чанку
        sample_vec = embed_texts(
            embedder,
            [chunks[0].text],
            model_name=rag.embeddings_model,
            is_query=False,
        )[0]

        client = get_client(db_path=db_path)
        ensure_collection(
            client=client,
            collection_name=col,
            vector_size=len(sample_vec),
        )

        remove_sources(
            client, col, [str(p) for p in temp_paths]
        )

        all_texts = [c.text for c in chunks]
        all_vectors = embed_texts(
            embedder,
            all_texts,
            model_name=rag.embeddings_model,
            is_query=False,
        )

        total = upsert_chunks(
            client=client,
            collection_name=col,
            chunks=chunks,
            vectors=all_vectors,
        )

        return IngestResponse(
            indexed_files=len(temp_paths),
            inserted_chunks=total,
            collection=col,
        )

    finally:
        for tmp_path in temp_paths:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:  # pylint: disable=broad-except
                pass


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    """Поиск релевантных чанков по запросу."""
    results = _retrieve_with_config(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
        metadata=payload.metadata,
    )

    return RetrieveResponse(
        query=payload.query,
        returned=len(results),
        results=results,
    )


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    """Поиск + генерация ответа через LLM."""
    llm = settings.llm

    results = _retrieve_with_config(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
        metadata=payload.metadata,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found for this query",
        )

    model = payload.model or llm.model
    context = build_context(results)
    answer = generate_answer(
        query=payload.query,
        context=context,
        model=model,
        llm_url=llm.base_url,
        llm_api_key=llm.api_key,
    )

    return AskResponse(
        query=payload.query,
        answer=answer,
        model=model,
        used_chunks=len(results),
        results=results,
    )

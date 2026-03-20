"""
API роутер для генерации mind map.

Эндпоинты:
- POST /text  — mind map из текста
- POST /file  — mind map из загруженного файла
- POST /query — mind map из результатов RAG-поиска
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.config import settings
from backend.services.document_reader import SUPPORTED_EXTENSIONS, read_document
from backend.services.mindmap import build_graph_data
from backend.services.retrieval import retrieve_chunks

router = APIRouter(prefix="/api/mindmap", tags=["Mindmap"])


# ── Schemas ─────────────────────────────────────────────────


class MindmapTextRequest(BaseModel):
    """Запрос на генерацию mind map из текста."""

    text: str = Field(..., min_length=1)
    top_concepts: int = Field(default=14, ge=1, le=100)
    min_concept_freq: int = Field(default=2, ge=1)
    min_edge_weight: int = Field(default=1, ge=1)


class MindmapQueryRequest(BaseModel):
    """Запрос на генерацию mind map из RAG-результатов."""

    query: str = Field(..., min_length=1)
    collection: str = "docs"
    top_k: int = Field(default=8, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    top_concepts: int = Field(default=14, ge=1, le=100)
    min_concept_freq: int = Field(default=2, ge=1)
    min_edge_weight: int = Field(default=1, ge=1)


class MindmapResponse(BaseModel):
    """Ответ с данными графа."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: Dict[str, Any]


# ── Helpers ─────────────────────────────────────────────────


def _build_response(
    graph: Dict[str, List[Dict[str, Any]]],
    source: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> MindmapResponse:
    meta: Dict[str, Any] = {
        "source": source,
        "nodes_count": len(graph["nodes"]),
        "edges_count": len(graph["edges"]),
    }
    if extra_meta:
        meta.update(extra_meta)
    return MindmapResponse(
        nodes=graph["nodes"], edges=graph["edges"], meta=meta
    )


# ── Endpoints ───────────────────────────────────────────────


@router.post("/text", response_model=MindmapResponse)
async def mindmap_from_text(
    payload: MindmapTextRequest,
) -> MindmapResponse:
    """Генерирует mind map из переданного текста."""
    graph = build_graph_data(
        text=payload.text,
        top_n_concepts=payload.top_concepts,
        min_concept_freq=payload.min_concept_freq,
        min_edge_weight=payload.min_edge_weight,
    )
    return _build_response(graph, source="text")


@router.post("/file", response_model=MindmapResponse)
async def mindmap_from_file(
    file: UploadFile = File(...),
    top_concepts: int = Form(14),
    min_concept_freq: int = Form(2),
    min_edge_weight: int = Form(1),
) -> MindmapResponse:
    """Генерирует mind map из загруженного документа."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        data = await file.read()
        tmp.write(data)
        temp_path = Path(tmp.name)

    try:
        text = read_document(temp_path)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:  # pylint: disable=broad-except
            pass

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No extractable text found in file",
        )

    graph = build_graph_data(
        text=text,
        top_n_concepts=top_concepts,
        min_concept_freq=min_concept_freq,
        min_edge_weight=min_edge_weight,
    )
    return _build_response(
        graph, source="file", extra_meta={"filename": file.filename}
    )


@router.post("/query", response_model=MindmapResponse)
async def mindmap_from_query(
    payload: MindmapQueryRequest,
) -> MindmapResponse:
    """Генерирует mind map из результатов RAG-поиска."""
    rag = settings.rag

    results = retrieve_chunks(
        query=payload.query,
        db_path=Path(rag.vector_store_path).resolve(),
        collection_name=payload.collection or rag.collection,
        embedding_model=rag.embeddings_model,
        embedding_cache=Path("data/embedding_cache").resolve(),
        embedder_url=rag.embedder_url or None,
        embedder_api_key=rag.embedder_api_key or None,
        top_k=payload.top_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
        metadata=payload.metadata,
    )

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found",
        )

    text = "\n\n".join(item["text"] for item in results if item.get("text"))
    graph = build_graph_data(
        text=text,
        top_n_concepts=payload.top_concepts,
        min_concept_freq=payload.min_concept_freq,
        min_edge_weight=payload.min_edge_weight,
    )
    return _build_response(
        graph,
        source="query",
        extra_meta={"query": payload.query, "used_chunks": len(results)},
    )

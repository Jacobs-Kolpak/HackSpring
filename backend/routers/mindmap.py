"""Роутер mind map: из текста, файла, RAG-запроса."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.services import rag as rag_service
from backend.services.mindmap import build_graph_data
from backend.utils.document_reader import SUPPORTED_EXTENSIONS, read_document

router = APIRouter(prefix="/api/mindmap", tags=["Mindmap"])


# ── Schemas ─────────────────────────────────────────────────


class TextRequest(BaseModel):
    """Mind map из текста."""

    text: str = Field(..., min_length=1)
    top_concepts: int = Field(default=14, ge=1, le=100)
    min_freq: int = Field(default=2, ge=1)
    min_edge: int = Field(default=1, ge=1)


class QueryRequest(BaseModel):
    """Mind map из RAG-результатов."""

    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    top_concepts: int = Field(default=14, ge=1, le=100)
    min_freq: int = Field(default=2, ge=1)
    min_edge: int = Field(default=1, ge=1)


class MindmapResponse(BaseModel):
    """Данные графа."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: Dict[str, Any]


# ── Helpers ─────────────────────────────────────────────────


def _respond(
    graph: Dict[str, List[Dict]], source: str, **extra: Any
) -> MindmapResponse:
    meta: Dict[str, Any] = {
        "source": source,
        "nodes_count": len(graph["nodes"]),
        "edges_count": len(graph["edges"]),
        **extra,
    }
    return MindmapResponse(nodes=graph["nodes"], edges=graph["edges"], meta=meta)


# ── Endpoints ───────────────────────────────────────────────


@router.post("/text", response_model=MindmapResponse)
async def from_text(payload: TextRequest) -> MindmapResponse:
    """Mind map из текста."""
    graph = build_graph_data(
        payload.text,
        top_n=payload.top_concepts,
        min_freq=payload.min_freq,
        min_edge=payload.min_edge,
    )
    return _respond(graph, "text")


@router.post("/file", response_model=MindmapResponse)
async def from_file(
    file: UploadFile = File(...),
    top_concepts: int = Form(14),
    min_freq: int = Form(2),
    min_edge: int = Form(1),
) -> MindmapResponse:
    """Mind map из загруженного файла."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        text = read_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text in file")

    graph = build_graph_data(text, top_n=top_concepts, min_freq=min_freq, min_edge=min_edge)
    return _respond(graph, "file", filename=file.filename)


@router.post("/query", response_model=MindmapResponse)
async def from_query(payload: QueryRequest) -> MindmapResponse:
    """Mind map из результатов RAG-поиска."""
    results = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant chunks found",
        )

    text = "\n\n".join(r["text"] for r in results if r.get("text"))
    graph = build_graph_data(
        text, top_n=payload.top_concepts, min_freq=payload.min_freq, min_edge=payload.min_edge
    )
    return _respond(graph, "query", query=payload.query, used_chunks=len(results))

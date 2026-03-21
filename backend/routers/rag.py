from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.services import rag as rag_service
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/jacobs/rag", tags=["RAG"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    fetch_k: Optional[int] = Field(default=None, ge=1, le=200)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    dense_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    source_name: Optional[str] = None


class AskRequest(RetrieveRequest):
    model: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    returned: int
    results: List[Dict[str, Any]]


class AskResponse(BaseModel):
    query: str
    answer: str
    model: str
    used_chunks: int
    results: List[Dict[str, Any]]


class IngestResponse(BaseModel):
    indexed_files: int
    inserted_chunks: int
    collection: str


@router.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_files(
    files: List[UploadFile] = File(...),
    collection: Optional[str] = Form(None),
    chunk_size: int = Form(0),
    chunk_overlap: int = Form(0),
) -> IngestResponse:
    temp_paths: List[Path] = []
    source_name_overrides: Dict[str, str] = {}
    try:
        for upload in files:
            ext = Path(upload.filename or "").suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported: {upload.filename}",
                )
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(await upload.read())
                temp_path = Path(tmp.name)
                temp_paths.append(temp_path)
                original_name = Path(upload.filename or "").name.strip()
                if original_name:
                    source_name_overrides[str(temp_path.resolve())] = original_name

        result = rag_service.ingest(
            paths=temp_paths,
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_name_overrides=source_name_overrides,
        )
        if result["inserted_chunks"] == 0:
            raise HTTPException(status_code=400, detail="No text extracted")

        return IngestResponse(**result)
    finally:
        for p in temp_paths:
            p.unlink(missing_ok=True)


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    results = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    return RetrieveResponse(
        query=payload.query, returned=len(results), results=results
    )


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest) -> AskResponse:
    results = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        fetch_k=payload.fetch_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    model = payload.model or settings.llm.model
    answer = rag_service.ask(payload.query, results, model)

    return AskResponse(
        query=payload.query,
        answer=answer,
        model=model,
        used_chunks=len(results),
        results=results,
    )

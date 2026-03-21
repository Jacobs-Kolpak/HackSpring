from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services import rag as rag_service
from backend.services import summary as summary_service
from backend.utils.document_reader import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/api/summary", tags=["Summary"])


class TextSummaryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    topic: str = Field(default="Без названия")
    max_sentences: int = Field(default=10, ge=1, le=100)
    model: Optional[str] = None


class QuerySummaryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=50)
    min_score: float = Field(default=0.15, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    source_name: Optional[str] = None
    topic: str = Field(default="Без названия")
    max_sentences: int = Field(default=10, ge=1, le=100)
    model: Optional[str] = None


class SummaryResponse(BaseModel):
    summary: str
    source: str
    model: str
    meta: Dict[str, Any] = {}


@router.post("/text", response_model=SummaryResponse)
async def from_text(payload: TextSummaryRequest) -> SummaryResponse:
    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    model = payload.model or settings.llm.model
    result = summary_service.summarize(
        payload.text,
        topic=payload.topic,
        max_sentences=payload.max_sentences,
        model=model,
    )
    return SummaryResponse(summary=result, source="text", model=model)


@router.post("/file", response_model=SummaryResponse)
async def from_file(
    file: UploadFile = File(...),
    topic: str = Form("Без названия"),
    max_sentences: int = Form(10),
    model: Optional[str] = Form(None),
) -> SummaryResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        temp_path = Path(tmp.name)
    try:
        from backend.core.config import settings  # pylint: disable=import-outside-toplevel

        used_model = model or settings.llm.model
        result = summary_service.summarize_file(
            temp_path,
            topic=topic,
            max_sentences=max_sentences,
            model=used_model,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    return SummaryResponse(
        summary=result,
        source="file",
        model=used_model,
        meta={"filename": file.filename},
    )


@router.post("/query", response_model=SummaryResponse)
async def from_query(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    payload: QuerySummaryRequest,
) -> SummaryResponse:
    results: List[Dict[str, Any]] = rag_service.retrieve(
        query=payload.query,
        collection=payload.collection,
        top_k=payload.top_k,
        min_score=payload.min_score,
        dense_weight=payload.dense_weight,
        source_name=payload.source_name,
    )
    if not results:
        raise HTTPException(status_code=404, detail="Релевантных чанков не найдено")

    text = "\n\n".join(r["text"] for r in results if r.get("text"))

    from backend.core.config import settings  # pylint: disable=import-outside-toplevel

    used_model = payload.model or settings.llm.model
    summary = summary_service.summarize(
        text,
        topic=payload.topic,
        max_sentences=payload.max_sentences,
        model=used_model,
    )
    return SummaryResponse(
        summary=summary,
        source="query",
        model=used_model,
        meta={"query": payload.query, "used_chunks": len(results)},
    )

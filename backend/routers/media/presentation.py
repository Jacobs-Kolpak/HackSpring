from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.services.media import presentation as pres_service
from backend.services.rag import service as rag_service

router = APIRouter(prefix="/api/jacobs/presentation", tags=["Presentation"])


class PresentationRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    top_k: int = 8
    max_slides: Optional[int] = None
    model: Optional[str] = None


class PresentationResponse(BaseModel):
    title: str
    slides_count: int
    download_url: str
    meta: Dict[str, Any] = {}


class PresentationFromResultsRequest(BaseModel):
    query: str
    results: List[Dict[str, Any]] = Field(..., min_length=1)
    max_slides: Optional[int] = None
    model: Optional[str] = None


@router.post("/generate", response_model=PresentationResponse)
async def generate_from_rag(body: PresentationRequest) -> PresentationResponse:
    try:
        results = rag_service.retrieve(
            body.query,
            collection=body.collection,
            top_k=body.top_k,
        )
        if not results:
            raise HTTPException(
                status_code=404,
                detail="RAG не нашёл релевантных документов",
            )

        from backend.core.config import settings
        used_model = body.model or settings.llm.model

        path, meta = pres_service.generate_presentation(
            body.query,
            results,
            model=used_model,
            max_slides=body.max_slides,
        )

        return PresentationResponse(
            title=meta["title"],
            slides_count=meta["slides_count"],
            download_url=f"/api/jacobs/presentation/download/{path.name}",
            meta=meta,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/from-results", response_model=PresentationResponse)
async def generate_from_results(
    body: PresentationFromResultsRequest,
) -> PresentationResponse:
    try:
        from backend.core.config import settings
        used_model = body.model or settings.llm.model

        path, meta = pres_service.generate_presentation(
            body.query,
            body.results,
            model=used_model,
            max_slides=body.max_slides,
        )

        return PresentationResponse(
            title=meta["title"],
            slides_count=meta["slides_count"],
            download_url=f"/api/jacobs/presentation/download/{path.name}",
            meta=meta,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/download/{filename}")
async def download(filename: str) -> FileResponse:
    from backend.core.config import settings

    from pathlib import Path
    path = Path(settings.presentation.output_dir).resolve() / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
    )

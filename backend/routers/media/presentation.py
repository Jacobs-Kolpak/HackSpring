from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
    style: Optional[Dict[str, Any]] = None


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
    style: Optional[Dict[str, Any]] = None


def _parse_style_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None or not raw.strip():
        return None
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Поле style должно быть валидным JSON-объектом",
        ) from exc
    if not isinstance(loaded, dict):
        raise HTTPException(
            status_code=400,
            detail="Поле style должно быть JSON-объектом",
        )
    return loaded


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
            style=body.style,
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


@router.post("/generate-with-template", response_model=PresentationResponse)
async def generate_from_rag_with_template(
    query: str = Form(...),
    collection: Optional[str] = Form(None),
    top_k: int = Form(8),
    max_slides: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
    style: Optional[str] = Form(None),
    template: UploadFile = File(...),
) -> PresentationResponse:
    if not template.filename or not template.filename.lower().endswith(".pptx"):
        raise HTTPException(status_code=400, detail="Нужен файл шаблона .pptx")

    temp_path: Optional[Path] = None
    try:
        parsed_style = _parse_style_json(style)

        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp:
            content = await template.read()
            temp.write(content)
            temp.flush()
            temp_path = Path(temp.name)

        results = rag_service.retrieve(
            query,
            collection=collection,
            top_k=top_k,
        )
        if not results:
            raise HTTPException(
                status_code=404,
                detail="RAG не нашёл релевантных документов",
            )

        from backend.core.config import settings
        used_model = model or settings.llm.model

        path, meta = pres_service.generate_presentation(
            query,
            results,
            model=used_model,
            max_slides=max_slides,
            style=parsed_style,
            template_path=temp_path,
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
    finally:
        await template.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


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
            style=body.style,
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
    path = Path(settings.presentation.output_dir).resolve() / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=filename,
    )
